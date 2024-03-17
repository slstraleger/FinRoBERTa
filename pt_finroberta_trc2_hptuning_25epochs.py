import os
import datasets
import evaluate
import pandas as pd
import peft
from transformers import (RobertaForMaskedLM, AutoTokenizer, TrainerCallback, get_linear_schedule_with_warmup,
                          DataCollatorForLanguageModeling, TrainingArguments, Trainer, logging)
import torch
import optuna
import numpy as np
from sklearn.model_selection import train_test_split

# LOGGER
logging.set_verbosity_info()
# INPUTS
ROBERTA_VERSION = 'roberta-large'
ON_COLAB = False
TESTING = False
HYPERPARAMETER_TUNING = True
SAMPLE_DATA = False
SAMPLE_NR = 1000
N_TRIALS = 1
BATCH_SIZE = 128
FP16 = True
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
MAX_LENGTH = 64
METRIC = 'accuracy'
GRADUAL_UNFREEZING = True
DISCRIMINATIVE_LR = True
ADAM_BETA_ONE = 0.9
ADAM_BETA_TWO = 0.98
ADAM_EPSILON = 1e-6

# CALCULATED INPUTS
if ON_COLAB:
    DIR = '/content/drive/MyDrive/Colab_notebooks'
    CHECKPOINT_DIR = DIR + '/roberta-training/checkpoint/'
    MODEL_DIR = DIR + '/roberta-training/models/'
else:
    # os.chdir('../')
    DIR = os.getcwd()
    CHECKPOINT_DIR = DIR + '/roberta-training/checkpoint/'
    MODEL_DIR = DIR + '/roberta-training/models/'

print(torch.cuda.is_available())

tokenizer = AutoTokenizer.from_pretrained(ROBERTA_VERSION)


def get_pretraining_series():
    pretraining_series = pd.read_csv(DIR + '/data/trc2_pt_64.csv', index_col=False).dropna()
    if SAMPLE_DATA:
        pretraining_series = pretraining_series.sample(n=SAMPLE_NR, random_state=1, axis=0, ignore_index=True)
    if TESTING:
        pretraining_series = pretraining_series[:500]
    return pretraining_series


def encode_headline(sentence):
    return tokenizer(sentence['0'], padding='max_length', truncation=True, return_tensors='pt', max_length=MAX_LENGTH)


train_dataset, eval_dataset = train_test_split(get_pretraining_series(), test_size=0.1, random_state=1)
train_dataset = datasets.Dataset.from_pandas(train_dataset)
eval_dataset = datasets.Dataset.from_pandas(eval_dataset)
pretraining_ds = datasets.DatasetDict({'train': train_dataset, 'eval': eval_dataset})

pretraining_ds = pretraining_ds.map(encode_headline, batched=True,
                                    remove_columns=['Unnamed: 0', '0', '__index_level_0__'])

pretraining_ds.set_format(type='torch')

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15, return_tensors='pt')


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        if len(logits[0].shape) == 0:
            # PEFT LORA TEST
            logits = logits[1]
        else:
            logits = logits[0]
    return logits.argmax(dim=-1)


metric = evaluate.load(METRIC)


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics
    labels = labels.reshape(-1)
    preds = preds.reshape(-1)
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]
    return metric.compute(predictions=preds, references=labels)


def print_trainable_parameters(mod):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in mod.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || "
          f"trainable%: {100 * trainable_params / all_param:.2f}")


class EpochBeginCallback(TrainerCallback):

    def on_epoch_begin(self, args, state, control, **kwargs):
        if GRADUAL_UNFREEZING:
            epoch = int(state.epoch)
            total_epochs = state.num_train_epochs
            embeddings = kwargs['model'].roberta.embeddings
            all_layers = kwargs['model'].roberta.encoder.layer
            number_of_layers = len(all_layers)
            if number_of_layers - int(len(all_layers) * (epoch + 1) / (total_epochs - 1)) >= 0:
                layer_min_to_unfreeze = number_of_layers - int(len(all_layers) * (epoch + 1) / (total_epochs - 1))
                layer_max_to_unfreeze = number_of_layers - int(len(all_layers) * epoch / (total_epochs - 1))
                unfrozen_layers = all_layers[layer_min_to_unfreeze:layer_max_to_unfreeze]
                for named_layer in unfrozen_layers.named_parameters():
                    if not any(i in named_layer[0] for i in ['base_layer.weight']):
                        named_layer[1].requires_grad = True
                print(f'\nUnfroze layers {layer_min_to_unfreeze} to {layer_max_to_unfreeze}')
            if epoch + 1 == total_epochs:
                for emb_params in embeddings.named_parameters():
                    if not any(i in emb_params[0] for i in ['token_type_embeddings', 'word_embeddings']):
                        emb_params[1].requires_grad = True
                print(f'\nUnfroze embedding layer')
            print_trainable_parameters(kwargs['model'])
        print('EpochBeginCallback completed')


best_score = -np.inf


def objective(trial, train, test):
    print(f'Starting trial {trial.number} with parameters:', trial.params)
    train.set_format('torch')
    test.set_format('torch')
    # Parameters
    params = {"learning_rate": trial.suggest_categorical("learning_rate", [1e-6, 1e-5, 1e-4, 1e-3]),
              "batch_size": trial.suggest_categorical("batch_size", [BATCH_SIZE]),
              "num_train_epochs": trial.suggest_categorical("num_train_epochs", [10, 14, 18]),
              "warmup_ratio": trial.suggest_categorical("warmup_ratio", [0, 0.1]),
              "lora_r": trial.suggest_categorical("lora_r", [8, 16, 32]),
              "lora_dropout": trial.suggest_categorical("lora_dropout", [0.05, 0.1]),
              "disc_lr": trial.suggest_categorical("disc_lr", [1.05, 1.2, 2.6])}
    training_arguments = TrainingArguments(output_dir=CHECKPOINT_DIR, overwrite_output_dir=True,
                                           do_train=True, evaluation_strategy='epoch',
                                           logging_strategy='no', save_strategy='no',
                                           num_train_epochs=25,
                                           report_to=['tensorboard'],
                                           per_device_train_batch_size=128,
                                           per_device_eval_batch_size=128,
                                           learning_rate=1e-4, adam_beta1=ADAM_BETA_ONE,
                                           adam_beta2=ADAM_BETA_TWO, adam_epsilon=ADAM_EPSILON,
                                           warmup_ratio=0.1,
                                           remove_unused_columns=False, fp16=FP16,
                                           label_names=['labels'])

    def get_trainer(lora_r, lora_dropout, train_data, eval_data):
        def get_model(model, r, dropout):
            for lora_layer in model.roberta.encoder.layer:
                lora_layer.attention.self.query = peft.tuners.lora.Linear(
                    lora_layer.attention.self.query, 'default', r=r, lora_alpha=r, lora_dropout=dropout)
                lora_layer.attention.self.value = peft.tuners.lora.Linear(
                    lora_layer.attention.self.value, 'default', r=r, lora_alpha=r, lora_dropout=dropout)
            return model

        roberta_trainer = Trainer(
            model=get_model(RobertaForMaskedLM.from_pretrained(ROBERTA_VERSION), lora_r, lora_dropout),
            args=training_arguments,
            data_collator=data_collator,
            train_dataset=train_data, eval_dataset=eval_data,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            callbacks=[EpochBeginCallback])

        def set_discriminative_lr():
            lr = training_arguments.learning_rate
            roberta = roberta_trainer.model.roberta
            num_layers = len(roberta.encoder.layer)
            dft_rate = 1.2
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            original = 'original_module'
            encoder_params = []
            for i in range(num_layers):
                encoder_decay = {'params': [p for n, p in list(roberta.encoder.layer[i].named_parameters()) if
                                            not any(nd in n for nd in no_decay) and original not in n],
                                 'weight_decay': 0.01,
                                 'lr': lr / (dft_rate ** (num_layers - i))}
                encoder_nodecay = {'params': [p for n, p in list(roberta.encoder.layer[i].named_parameters()) if
                                              any(nd in n for nd in no_decay) and original not in n],
                                   'weight_decay': 0.0,
                                   'lr': lr / (dft_rate ** (num_layers - i))}
                if [x for x in encoder_decay['params'] if (x.sum() == torch.tensor(0))]:
                    zero_tensors = [x for x in encoder_decay['params'] if (x.sum() == torch.tensor(0))]
                    encoder_decay['params'] = [x for x in encoder_decay['params'] if not (x.sum() == torch.tensor(0))]
                    for zero_tensor in zero_tensors:
                        lora_encoder_decay = {
                            'params': [zero_tensor], 'weight_decay': 0.01,
                            'lr': lr / (dft_rate ** (num_layers - i))}
                        encoder_params.append(lora_encoder_decay)
                encoder_params.append(encoder_decay)
                encoder_params.append(encoder_nodecay)

            optimizer_grouped_parameters = [{
                    'params': [p for n, p in list(roberta.embeddings.named_parameters())
                               if (not any(nd in n for nd in no_decay) and n != 'word_embeddings.weight'
                                   and original not in n)],
                    'weight_decay': 0.01, 'lr': lr / (dft_rate ** (num_layers + 1))}, {
                    'params': [p for n, p in list(roberta.embeddings.named_parameters())
                               if any(nd in n for nd in no_decay) and original not in n],
                    'weight_decay': 0.0, 'lr': lr / (dft_rate ** (num_layers + 1))}, {
                    'params': [p for n, p in list(roberta_trainer.model.lm_head.named_parameters())
                               if not any(nd in n for nd in no_decay) and original not in n],
                    'weight_decay': 0.01, 'lr': lr}, {
                    'params': [p for n, p in list(roberta_trainer.model.lm_head.named_parameters())
                               if any(nd in n for nd in no_decay) and original not in n],
                    'weight_decay': 0.0, 'lr': lr}
            ]

            optimizer_grouped_parameters.extend(encoder_params)
            num_train_optimization_steps = (int(len(train_data) / training_arguments.per_device_train_batch_size
                                                / training_arguments.gradient_accumulation_steps)
                                            * training_arguments.num_train_epochs)
            training_arguments.warmup_steps = int(float(num_train_optimization_steps) * training_arguments.warmup_ratio)
            roberta_trainer.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr,
                                                          betas=(ADAM_BETA_ONE, ADAM_BETA_TWO),
                                                          eps=ADAM_EPSILON, fused=True)
            roberta_trainer.lr_scheduler = get_linear_schedule_with_warmup(
                roberta_trainer.optimizer,
                num_warmup_steps=training_arguments.warmup_steps,
                num_training_steps=num_train_optimization_steps)

        if DISCRIMINATIVE_LR:
            set_discriminative_lr()
        print_trainable_parameters(roberta_trainer.model)
        return roberta_trainer

    trainer = get_trainer(16, 0.1, train, test)

    def freeze_layers():
        for all_params in trainer.model.parameters():
            all_params.requires_grad = False
        for lm_head_param in trainer.model.lm_head.parameters():
            lm_head_param.requires_grad = True
        for original_layers in trainer.model.roberta.encoder.named_parameters():
            if any(i in original_layers[0] for i in ['query.weight', 'value.weight']):
                original_layers[1].requires_grad = False
        print(f'Entire model frozen, lm_head unfrozen')

    if GRADUAL_UNFREEZING:
        freeze_layers()
    trainer.train()
    avg_score = trainer.evaluate()['eval_accuracy']
    # Save the model is the avg score > current best score
    avg_accuracy = avg_score
    global best_score
    if avg_accuracy > best_score:
        best_score = avg_accuracy
        for layer in trainer.model.roberta.encoder.layer:
            layer.attention.self.query.merge()
            layer.attention.self.value.merge()
            layer.attention.self.query = layer.attention.self.query.base_layer
            layer.attention.self.value = layer.attention.self.value.base_layer
        trainer.model.save_pretrained(MODEL_DIR + 'peft_model_trc2')
    # Clean up
    print(f"Average result {avg_score} and the best score {best_score}")
    return avg_score


# Train the model and find the optimal parameters using optuna library
def train_model_with_optuna(train, test):
    # # Create a study to find the optimal hyper-parameters
    study = optuna.create_study(direction='maximize')  # Resume the existing study
    # Set up the timeout to avoid runing out of quote
    # n_jobs =-1 is CPU bounded
    study.optimize(lambda trial: objective(trial, train, test), n_jobs=1, n_trials=N_TRIALS, show_progress_bar=True,
                   gc_after_trial=True)
    # Print out the experiment results
    return study


kf_study = train_model_with_optuna(pretraining_ds['train'], pretraining_ds['eval'])
kf_study.trials_dataframe().to_csv(MODEL_DIR + 'peft_model_trc2/trials.csv')
print(f"Best parameters: {kf_study.best_params}\n"
      f"Number of finished trials: {len(kf_study.trials)}\n"
      f"Best trial:{kf_study.best_trial}\n")
