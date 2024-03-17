import gc
import json
import os
from datetime import datetime
import time

import numpy as np
import optuna
import pandas as pd
import peft
import torch
from torch import nn
from torch.nn import MSELoss
from torch.utils.tensorboard import SummaryWriter
from transformers import RobertaForMaskedLM, get_linear_schedule_with_warmup
from hiagm_save_models import ModifiedHiAGM
from HiAGM.data_modules.dataset import ClassificationDataset
from HiAGM.helper.configure import Configure
from HiAGM.data_modules.vocab import Vocab
from HiAGM.helper.utils import load_checkpoint
import HiAGM.helper.logger as logger
from torch.utils.data import DataLoader

INPUT_DATA_DIR = 'data/HiAGM/data'
HIAGM_MODEL_LOC = 'scripts_used_for_modelling/HiAGM-RoBERTa/best_HiAGM-RoBERTa.pt'
N_LABELS = 28
BATCH_SIZE = 1
MAX_LENGTH = 64
N_DATA_W = 0
N_TRIALS = 10
K_FOLD_SPLITS = 5
ADAM_BETA_ONE = 0.9
ADAM_BETA_TWO = 0.98
ADAM_EPSILON = 1e-6
DISCRIMINATIVE_LR = True
GRADUAL_UNFREEZING = True


class Collator(object):
    def __init__(self, config):
        """
        Collator object for the collator_fn in data_modules.data_loader
        :param config: helper.configure, Configure Object
        :param vocab: data_modules.vocab, Vocab Object
        """
        super(Collator, self).__init__()
        self.device = config.train.device_setting.device
        # self.label_size = len(vocab.v2i['label'].keys())
        self.label_size = N_LABELS

    def _multi_hot(self, batch_labels):
        """
        :param batch_labels: label idx list of one batch, List[List[int]], e.g.  [[1,2],[0,1,3,4]]
        :return: multi-hot value for classification -> List[List[int]], e.g. [[0,1,1,0,0],[1,1,0,1,1]
        """
        '''
        batch_size = len(batch_labels)
        max_length = max([len(sample) for sample in batch_labels])
        aligned_batch_labels = []
        for sample_label in batch_labels:
            aligned_batch_labels.append(sample_label + (max_length - len(sample_label)) * [sample_label[0]])
        '''
        aligned_batch_labels = torch.Tensor(batch_labels).long()
        batch_labels_multi_hot = nn.functional.one_hot(aligned_batch_labels, N_LABELS)
        return batch_labels_multi_hot

    def __call__(self, batch):
        """
        transform data for training
        :param batch: Dict{'token': List[List[int]],
                           'label': List[List[int]],
                            'token_len': List[int]}
        :return: batch -> Dict{'token': torch.FloatTensor,
                               'label': torch.FloatTensor,
                               'token_len': torch.FloatTensor,
                               'label_list': List[List[int]]}
        """
        batch_token = []
        batch_label = []
        batch_doc_len = []
        batch_sentiment = []
        batch_att_mask = []
        for sample in batch:
            batch_token.append(sample['token'])
            batch_label.append(sample['label'])
            batch_doc_len.append(sample['token_len'])
            batch_sentiment.append(sample['sentiment_score'])
            batch_att_mask.append(sample['attention_mask'])

        batch_token = torch.tensor(batch_token)
        batch_multi_hot_label = self._multi_hot(batch_label)
        batch_doc_len = torch.FloatTensor(batch_doc_len)
        batch_sentiment = torch.FloatTensor(batch_sentiment)
        batch_att_mask = torch.tensor(batch_att_mask)

        return {
            'token': batch_token,
            'label': batch_multi_hot_label,
            'token_len': batch_doc_len,
            'label_list': batch_label,
            'sentiment_score': batch_sentiment,
            'attention_mask': batch_att_mask
        }


class NewModifiedDataset(ClassificationDataset):
    def __init__(self, config, vocab, stage='TRAIN', on_memory=False, corpus_lines=None):
        super(NewModifiedDataset, self).__init__(config, vocab, stage, on_memory, corpus_lines)
        self.new_v2i = {
            'ta_stock': 0, 'price action': 1, 'coverage': 2, 'rumors': 3, 'ma': 4, 'options': 5, 'strategy': 6,
            'dividend policy': 7, 'sales': 8, 'market_2': 9, 'volatility': 10, 'financial': 11, 'appointment': 12,
            'legal': 13, 'risks': 14, 'signal': 15, 'central banks': 16, 'regulatory': 17, 'ta_corp': 18,
            'fundamentals': 19, 'company communication': 20, 'ipo': 21, 'conditions': 22, 'currency': 23,
            'reputation': 24, 'insider activity': 25, 'trade': 26, 'buyside': 27}
        self.new_i2v = {
            0: 'ta_stock', 1: 'price action', 2: 'coverage', 3: 'rumors', 4: 'ma', 5: 'options', 6: 'strategy',
            7: 'dividend policy', 8: 'sales', 9: 'market_2', 10: 'volatility', 11: 'financial', 12: 'appointment',
            13: 'legal', 14: 'risks', 15: 'signal', 16: 'central banks', 17: 'regulatory', 18: 'ta_corp', 19:
                'fundamentals', 20: 'company communication', 21: 'ipo', 22: 'conditions', 23: 'currency',
            24: 'reputation', 25: 'insider activity', 26: 'trade', 27: 'buyside'}
        if on_memory:
            for data_point in self.data:
                data_point['label'] = self.new_v2i[data_point['label'][1]]

    def _preprocess_sample(self, sample_str):
        """
        preprocess each sample with the limitation of maximum length and pad each sample to maximum length
        :param sample_str: Str format of json data, "Dict{'token': List[Str], 'label': List[Str]}"
        :return: sample -> Dict{'token': List[int], 'label': List[int], 'token_len': int}
        """
        if not self.on_memory:
            raw_sample = json.loads(sample_str)
            sample = {'token': [], 'label': []}
            for k in raw_sample.keys():
                if k == 'token':
                    sample[k] = raw_sample[k]
                else:
                    sample[k] = []
                    for v in raw_sample[k]:
                        if v not in self.vocab.v2i[k].keys():
                            logger.warning('Vocab not in ' + k + ' ' + v)
                        else:
                            sample[k].append(self.vocab.v2i[k][v])
        else:
            sample = sample_str
        if not sample['token']:
            sample['token'].append(self.vocab.padding_index)
        if self.mode == 'TRAIN':
            assert isinstance(sample['label'], int), 'Label is empty'
        else:
            sample['label'] = [0]
        sample['token_len'] = min(len(sample['token']), self.max_input_length)
        padding = [self.vocab.padding_index for _ in range(0, self.max_input_length - len(sample['token']))]
        sample['token'] += padding
        sample['token'] = sample['token'][:self.max_input_length]
        return sample


def data_loaders(config, vocab, data=None):
    """
    get data loaders for training and evaluation
    :param config: helper.configure, Configure Object
    :param vocab: data_modules.vocab, Vocab Object
    :param data: on-memory data, Dict{'train': List[str] or None, ...}
    :return: -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader)
    """
    if data is None:
        data = {'train': None, 'val': None, 'test': None}
    on_memory = data['train'] is not None
    collate_fn = Collator(config)
    train_dataset = NewModifiedDataset(config, vocab, stage='TRAIN', on_memory=on_memory, corpus_lines=data['train'])
    train_loader = DataLoader(train_dataset,
                              batch_size=config.train.batch_size,
                              shuffle=True,
                              num_workers=config.train.device_setting.num_workers,
                              collate_fn=collate_fn,
                              pin_memory=True)

    val_dataset = NewModifiedDataset(config, vocab, stage='VAL', on_memory=on_memory, corpus_lines=data['val'])
    val_loader = DataLoader(val_dataset,
                            batch_size=config.eval.batch_size,
                            shuffle=True,
                            num_workers=config.train.device_setting.num_workers,
                            collate_fn=collate_fn,
                            pin_memory=True)

    test_dataset = NewModifiedDataset(config, vocab, stage='TEST', on_memory=on_memory, corpus_lines=data['test'])
    test_loader = DataLoader(test_dataset,
                             batch_size=config.eval.batch_size,
                             shuffle=True,
                             num_workers=config.train.device_setting.num_workers,
                             collate_fn=collate_fn,
                             pin_memory=True)

    return train_loader, val_loader, test_loader


def prepare_lora(model, lora_r, lora_dropout):
    for lora_layer in model.encoder.layer:
        lora_layer.attention.self.query = peft.tuners.lora.Linear(
            lora_layer.attention.self.query, 'default',
            r=lora_r, lora_alpha=lora_r, lora_dropout=lora_dropout)
        lora_layer.attention.self.value = peft.tuners.lora.Linear(
            lora_layer.attention.self.value, 'default',
            r=lora_r, lora_alpha=lora_r, lora_dropout=lora_dropout)


class RobertaHiAGM(nn.Module):

    def set_discriminative_lr(self, lr, dft_rate, warmup_ratio, len_training, n_epochs, gradient_accumulation_steps=1):
        num_layers = len(self.roberta_model.encoder.layer)
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        original = 'original_module'
        encoder_params = []
        for i in range(num_layers):
            encoder_decay = {'params': [p for n, p in list(self.roberta_model.encoder.layer[i].named_parameters()) if
                                        not any(nd in n for nd in no_decay) and original not in n],
                             'weight_decay': 0.01,
                             'lr': lr / (dft_rate ** (num_layers - i))}
            encoder_nodecay = {'params': [p for n, p in list(self.roberta_model.encoder.layer[i].named_parameters()) if
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
            'params': [p for n, p in list(self.roberta_model.embeddings.named_parameters())
                       if (not any(nd in n for nd in no_decay) and n != 'word_embeddings.weight'
                           and original not in n)],
            'weight_decay': 0.01, 'lr': lr / (dft_rate ** (num_layers + 1))}, {
            'params': [p for n, p in list(self.roberta_model.embeddings.named_parameters())
                       if any(nd in n for nd in no_decay) and original not in n],
            'weight_decay': 0.0, 'lr': lr / (dft_rate ** (num_layers + 1))}, {
            'params': [p for n, p in list(self.linear.named_parameters())
                       if not any(nd in n for nd in no_decay) and original not in n],
            'weight_decay': 0.01, 'lr': lr}, {
            'params': [p for n, p in list(self.linear.named_parameters())
                       if any(nd in n for nd in no_decay) and original not in n],
            'weight_decay': 0.0, 'lr': lr}, {
            'params': [p for n, p in list(self.asp_cat_emb.named_parameters())
                       if not any(nd in n for nd in no_decay) and original not in n],
            'weight_decay': 0.01, 'lr': lr}, {
            # 'params': [p for n, p in list(self.asp_cat_emb.named_parameters())
            #            if any(nd in n for nd in no_decay) and original not in n],
            # 'weight_decay': 0.0, 'lr': lr},
            # {
            'params': [p for n, p in list(self.linear2.named_parameters())
                       if not any(nd in n for nd in no_decay) and original not in n],
            'weight_decay': 0.01, 'lr': lr}, {
            'params': [p for n, p in list(self.linear2.named_parameters())
                       if any(nd in n for nd in no_decay) and original not in n],
            'weight_decay': 0.0, 'lr': lr}
        ]

        optimizer_grouped_parameters.extend(encoder_params)
        num_train_optimization_steps = (int(len_training / BATCH_SIZE / gradient_accumulation_steps) * n_epochs)
        warmup_steps = int(float(num_train_optimization_steps) * warmup_ratio)
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr, betas=(ADAM_BETA_ONE, ADAM_BETA_TWO),
                                           eps=ADAM_EPSILON, fused=True)
        self.lr_scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps,
                                                            num_training_steps=num_train_optimization_steps)

    def __init__(self, roberta_model_loc, emb_dims, cls_dropout,
                 asp_cat_dropout, linear_mid_dropout, lora_r, lora_dropout):
        super().__init__()
        self.lr_scheduler = None
        self.optimizer = None
        self.roberta_model = RobertaForMaskedLM.from_pretrained(roberta_model_loc).roberta.half()
        prepare_lora(self.roberta_model, lora_r, lora_dropout)
        self.asp_cat_emb = nn.Embedding(N_LABELS, emb_dims)
        self.cls_dropout = nn.Dropout(cls_dropout)
        self.asp_cat_dropout = nn.Dropout(asp_cat_dropout)
        self.linear = nn.Linear(emb_dims, 1)
        self.linear_mid_dropout = nn.Dropout(linear_mid_dropout)
        self.linear2 = nn.Linear(N_LABELS + int(self.roberta_model.config.hidden_size / emb_dims), 1)
        self.sigmoid = nn.Sigmoid()
        self.new_v2i = {
            'ta_stock': 0, 'price action': 1, 'coverage': 2, 'rumors': 3, 'ma': 4, 'options': 5, 'strategy': 6,
            'dividend policy': 7, 'sales': 8, 'market_2': 9, 'volatility': 10, 'financial': 11, 'appointment': 12,
            'legal': 13, 'risks': 14, 'signal': 15, 'central banks': 16, 'regulatory': 17, 'ta_corp': 18,
            'fundamentals': 19, 'company communication': 20, 'ipo': 21, 'conditions': 22, 'currency': 23,
            'reputation': 24, 'insider activity': 25, 'trade': 26, 'buyside': 27}

    def set_train(self):
        self.roberta_model.train()
        self.asp_cat_emb.train()
        self.linear.train()
        self.linear2.train()

    def set_eval(self):
        self.roberta_model.eval()
        self.asp_cat_emb.eval()
        self.linear.eval()
        self.linear2.eval()

    def forward(self, inputs, training):
        if training:
            asp_cat = self.asp_cat_emb(inputs['label'])
        else:
            asp_cat = self.asp_cat_emb(inputs['one_hot_l2_pred'])
        asp_cat = self.asp_cat_dropout(asp_cat)
        cls_embed = self.roberta_model(inputs['token'],
                                       inputs['attention_mask'])[0][:, 0, :].view(1, -1, asp_cat.shape[2])
        cls_embed = self.cls_dropout(cls_embed)
        output = self.linear(torch.cat([cls_embed, asp_cat], 1)).squeeze(dim=2)
        output = self.linear_mid_dropout(output)
        output = self.linear2(output).view(1)
        output = self.sigmoid(output)
        return output


def objective(trial, embedding_model_loc, output_name, hiagm_params):
    params = {"emb_dims": trial.suggest_categorical("emb_dims", [32, 64, 128]),
              "cls_dropout": trial.suggest_categorical("cls_dropout", [0.05, 0.1, 0.2, 0.5]),
              "asp_cat_dropout": trial.suggest_categorical("asp_cat_dropout", [0.05, 0.1, 0.2, 0.5]),
              "linear_mid_dropout": trial.suggest_categorical("linear_mid_dropout", [0.05, 0.1, 0.2, 0.5]),
              "lora_r": trial.suggest_categorical("lora_r", [8, 16, 32]),
              "lora_dropout": trial.suggest_categorical("lora_dropout", [0.05, 0.1]),
              "warmup_ratio": trial.suggest_categorical("warmup_ratio", [0, 0.1]),
              "learning_rate": trial.suggest_categorical("learning_rate", [1e-6, 1e-5, 1e-4, 1e-3]),
              "disc_lr": trial.suggest_categorical("disc_lr", [1.05, 1.2, 2.6]),
              "num_train_epochs": trial.suggest_categorical("num_train_epochs", [13, 25, 49])
              }

    NUM_KERNEL = hiagm_params['text_dim'] // len(hiagm_params['kernel_size'])

    print(f'Starting training final model ({output_name}), trial {trial.number} with parameters:', params)

    fold_score = np.zeros((K_FOLD_SPLITS, params['num_train_epochs']))
    for fold in range(K_FOLD_SPLITS):
        config = Configure(config={
            "data": {"dataset": "fiqa", "data_dir": "data/HiAGM/r-h_data/", "train_file": f"fiqa_train_{fold}.json",
                     "val_file": f"fiqa_val_{fold}.json", "test_file": "fiqa_test.json",
                     "prob_json": "fiqa_prob.json", "hierarchy": "fiqa.taxonomy"},
            "vocabulary": {"dir": "data/HiAGM/vocab", "vocab_dict": "word.dict", "max_token_vocab": 60000,
                           "label_dict": "label.dict"},
            "embedding": {"token": {"dimension": 1024, "type": "pretrain",
                                    "pretrained_file": "data/HiAGM/vocab/roberta_vocab.txt",
                                    "dropout": hiagm_params['fully_connected_dropout'],
                                    "init_type": "uniform"},
                          "label": {"dimension": hiagm_params['label_dim'], "type": "random",
                                    "dropout": hiagm_params['fully_connected_dropout'],
                                    "init_type": "kaiming_uniform"}},
            "text_encoder": {"max_length": MAX_LENGTH, "RNN": {"bidirectional": True, "num_layers": 1, "type": "GRU",
                                                               "hidden_dimension": hiagm_params['rnn_hidden_dim'],
                                                               "dropout": hiagm_params['gru_dropout']},
                             "CNN": {"kernel_size": hiagm_params['kernel_size'], "num_kernel": NUM_KERNEL},
                             "topK_max_pooling": 1},
            "structure_encoder": {"type": "GCN", "node": {"type": "text", "dimension": hiagm_params['label_dim'],
                                                          "dropout": hiagm_params['structure_encoder_dropout']}},
            "model": {"type": "HiAGM-TP",
                      "linear_transformation": {"text_dimension": hiagm_params['text_dim'],
                                                "node_dimension": hiagm_params['label_dim'],
                                                "dropout": hiagm_params['fully_connected_dropout']},
                      "classifier": {"num_layer": 1, "dropout": hiagm_params['fully_connected_dropout']}},
            "train": {"optimizer": {"type": hiagm_params['optimizer'], "learning_rate": hiagm_params['learning_rate'],
                                    "lr_decay": 0.8, "lr_patience": 1, "early_stopping": 4},
                      "batch_size": BATCH_SIZE, "start_epoch": 0, "end_epoch": params['num_train_epochs'],
                      "loss": {"classification": "BCEWithLogitsLoss",
                               "recursive_regularization": {"flag": True,
                                                            "penalty": hiagm_params['recursive_regularization']}},
                      "device_setting": {"device": "cuda", "visible_device_list": "0", "num_workers": N_DATA_W},
                      "checkpoint": {"dir": f"HiAGM_output/{output_name}/checkpoint", "max_number": 0,
                                     "save_best": ["f1"]}},
            "eval": {"batch_size": BATCH_SIZE, "threshold": 0.5},
            "test": {"best_checkpoint": "best_HiAGM-TP", "batch_size": BATCH_SIZE},
            "log": {"level": "info", "filename": f"RoBERTa-HiAGM_output/{output_name}/logs/log_t{trial.number}.log"}})
        # loading corpus and generate vocabulary
        corpus_vocab = Vocab(config, special_token=['1', '3'])

        def get_preloaded_data():
            return {
                'train': [json.loads(line) for line in
                          open(os.path.join(config.data.data_dir, config.data.train_file))],
                'val': [json.loads(line) for line in open(os.path.join(config.data.data_dir, config.data.val_file))],
                'test': [json.loads(line) for line in open(os.path.join(config.data.data_dir, config.data.test_file))]}

        train_loader, val_loader, test_loader = data_loaders(config, corpus_vocab, get_preloaded_data())
        best_epoch = -1
        best_performance = np.inf
        now = datetime.now().strftime("%m-%d_%H-%M-%S")
        writer = SummaryWriter(f"RoBERTa-HiAGM_output/{output_name}/runs/t{trial.number}_f{fold}__{now}/")

        hiagm = ModifiedHiAGM(config, corpus_vocab, model_type=config.model.type, model_mode='TRAIN',
                              emb=RobertaForMaskedLM.from_pretrained(embedding_model_loc).roberta.embeddings)
        load_checkpoint(HIAGM_MODEL_LOC, hiagm, config)
        config.train.start_epoch = 0
        hiagm.eval()
        for param in hiagm.parameters():
            param.requires_grad = False
        hiagm.to(config.train.device_setting.device)

        roberta_hiagm = RobertaHiAGM(embedding_model_loc,
                                     emb_dims=params['emb_dims'], cls_dropout=params['cls_dropout'],
                                     asp_cat_dropout=params['asp_cat_dropout'],
                                     linear_mid_dropout=params['linear_mid_dropout'], lora_r=params['lora_r'],
                                     lora_dropout=params['lora_dropout'])
        roberta_hiagm.to(config.train.device_setting.device)
        if DISCRIMINATIVE_LR:
            roberta_hiagm.set_discriminative_lr(
                params['learning_rate'], params['disc_lr'], params['warmup_ratio'], params['num_train_epochs'],
                len(train_loader))
        if GRADUAL_UNFREEZING:
            for all_params in roberta_hiagm.roberta_model.parameters():
                all_params.requires_grad = False
            print(f'RoBERTa layers frozen')

        def get_one_hot_l2_pred(inputs):
            with torch.no_grad():
                hiagm_output = torch.sigmoid(hiagm(inputs)[0]).cpu().tolist()
                all_level2_labels = set(sum(
                    [v for k, v in hiagm.structure_encoder.hierarchical_label_dict.items() if k != -1], []))
                l2_pred = {i: hiagm_output[i] for i in all_level2_labels}
                l2_pred = sorted(l2_pred, key=l2_pred.get, reverse=True)[0]
                l2_pred = roberta_hiagm.new_v2i[hiagm.vocab.i2v['label'][l2_pred]]
                one_hot_l2_pred = np.zeros(len(inputs['label'][0]))
                one_hot_l2_pred[l2_pred] = 1
                one_hot_l2_pred = torch.as_tensor(one_hot_l2_pred, dtype=torch.int64).unsqueeze(0)
                return one_hot_l2_pred

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

        def unfreeze_layers(model, current_epoch, total_epochs):
            embeddings = model.roberta_model.embeddings
            all_layers = model.roberta_model.encoder.layer
            number_of_layers = len(all_layers)
            if number_of_layers - int(number_of_layers * (current_epoch + 1) / (total_epochs - 1)) >= 0:
                layer_min_to_unfreeze = number_of_layers - int(number_of_layers * (current_epoch + 1) / (total_epochs - 1))
                layer_max_to_unfreeze = number_of_layers - int(number_of_layers * current_epoch / (total_epochs - 1))
                unfrozen_layers = all_layers[layer_min_to_unfreeze:layer_max_to_unfreeze]
                for named_layer in unfrozen_layers.named_parameters():
                    if not any(i in named_layer[0] for i in ['base_layer.weight']):
                        named_layer[1].requires_grad = True
                print(f'\nUnfroze layers {layer_min_to_unfreeze} to {layer_max_to_unfreeze}')
            if current_epoch + 1 == total_epochs:
                for emb_params in embeddings.named_parameters():
                    if not any(i in emb_params[0] for i in ['token_type_embeddings', 'word_embeddings']):
                        emb_params[1].requires_grad = True
                print(f'\nUnfroze embedding layer')
            print_trainable_parameters(model)

        # set origin log
        model_checkpoint = config.train.checkpoint.dir
        model_name = f'{config.model.type}_{trial.number}'
        wait = 0

        if not os.path.isdir(model_checkpoint):
            os.mkdir(model_checkpoint)

        gc.collect()

        def training_loop(loader, training, backward):
            total_loss = []
            for index, batch in enumerate(loader):
                batch['one_hot_l2_pred'] = get_one_hot_l2_pred(batch)
                for x in batch:
                    if x in ['token', 'label', 'attention_mask', 'sentiment_score', 'one_hot_l2_pred']:
                        batch[x] = batch[x].to(config.train.device_setting.device)
                train_real = batch['sentiment_score'] * 2 - 1
                train_pred = roberta_hiagm(batch, training=training) * 2 - 1
                loss_fc = MSELoss()
                iter_loss = loss_fc(train_pred, train_real)
                if backward:
                    iter_loss.backward()
                    roberta_hiagm.optimizer.step()
                    roberta_hiagm.lr_scheduler.step()
                    roberta_hiagm.optimizer.zero_grad()
                total_loss.append(iter_loss.item())
            return np.mean(total_loss)

        # train
        for epoch in range(config.train.start_epoch, config.train.end_epoch):
            torch.cuda.empty_cache()
            start_time = time.time()
            roberta_hiagm.set_train()
            if GRADUAL_UNFREEZING:
                unfreeze_layers(roberta_hiagm, epoch, params['num_train_epochs'])
            training_loop(train_loader, training=True, backward=True)

            with torch.no_grad():
                roberta_hiagm.set_eval()
                loss = training_loop(train_loader, training=False, backward=False)
                writer.add_scalar("mse/train", loss, epoch)

                loss = training_loop(val_loader, training=False, backward=False)
                writer.add_scalar("mse/val", loss, epoch)
                fold_score[fold][epoch] = loss

            # saving best model and check model
            '''
            if not loss <= best_performance:
                wait += 1
                if wait % config.train.optimizer.lr_patience == 0:
                    
                    logger.warning("Performance has not been improved for {} epochs, updating learning rate".format(wait))
                    logger.warning('Learning rate update {}--->{}'.format(
                        trainer.optimizer.param_groups[0]['lr'],
                        trainer.optimizer.param_groups[0]['lr'] * trainer.config.train.optimizer.lr_decay))
                    for param in trainer.optimizer.param_groups:
                        param['lr'] = (param['lr'] * trainer.config.train.optimizer.lr_decay)
                    
                if wait == config.train.optimizer.early_stopping:
                    logger.warning(
                        "Performance has not been improved for {} epochs, stopping train with early stopping".format(wait))
                    break
            '''
            if loss < best_performance:
                wait = 0
                logger.info('Improve MSE loss {} --> {}'.format(best_performance, loss))
                best_performance = loss
                '''
                best_epoch = epoch
                save_checkpoint({'epoch': epoch, 'model_type': config.model.type, 'state_dict': hiagm.state_dict(),
                                 'best_performance': best_performance, 'optimizer': optimize.state_dict()},
                                os.path.join(model_checkpoint, 'best_' + model_name + '.pt'))
                '''
            logger.info('Epoch {} Time Cost {} secs.'.format(epoch, time.time() - start_time))
        writer.close()
    print(f'Average performance of hyperparameters: {np.min(np.mean(fold_score, axis=1))}')
    return np.min(np.mean(fold_score, axis=1))


if __name__ == '__main__':
    def kf_train(emb_model_loc, hiagm_name, hp_orig):
        def train_model_with_optuna():
            # # Create a study to find the optimal hyper-parameters
            study = optuna.create_study(direction='minimize')  # Resume the existing study
            # Set up the timeout to avoid runing out of quote
            # n_jobs =-1 is CPU bounded

            study.optimize(lambda trial: objective(trial, emb_model_loc, hiagm_name, hiagm_params=hp_orig),
                           n_trials=N_TRIALS, show_progress_bar=True, gc_after_trial=True)
            # Print out the experiment results
            return study

        kf_studies = []
        kf_studies.append(train_model_with_optuna())
        for kf_study in kf_studies:
            print(f"Best parameters: {kf_study.best_params}\n"
                  f"Number of finished trials: {len(kf_study.trials)}\n"
                  f"Best trial:{kf_study.best_trial}\n")

        kf_total_trials_df = pd.concat([x.trials_dataframe() for x in kf_studies])

        kf_total_trials_df.to_csv(f'HiAGM_output/test_trials_{hiagm_name}.csv')

        # train(dict_config)
        print(f'{hiagm_name} training finished')
    kf_train('roberta-large', 'HiAGM-RoBERTa',
             {"learning_rate": 5e-4, "optimizer": "Adam", "structure_encoder_dropout": 0.2,
              "gru_dropout": 0.1, "fully_connected_dropout": 0.5, "recursive_regularization": 1e-3,
              "rnn_hidden_dim": 64, "label_dim": 128, "text_dim": 64, "kernel_size": [2, 3, 4, 5]})
    kf_train('scripts_used_for_modelling/RoBERTa-TRC2/roberta-training/models/peft_model_trc2',
             'HiAGM-RoBERTa-TRC2',
             {"learning_rate": 1e-4, "optimizer": "Adam", "structure_encoder_dropout": 0.3,
              "gru_dropout": 0.2, "fully_connected_dropout": 0.5, "recursive_regularization": 1e-3,
              "rnn_hidden_dim": 64, "label_dim": 128, "text_dim": 64, "kernel_size": [2, 3, 4, 5]})
    kf_train('scripts_used_for_modelling/RoBERTa-TRC2-FPB/roberta-training/models/peft_model_fpb',
             'HiAGM-RoBERTa-TRC2-FPB',
             {"learning_rate": 1e-4, "optimizer": "Adam", "structure_encoder_dropout": 0.3,
              "gru_dropout": 0.2, "fully_connected_dropout": 0.5, "recursive_regularization": 1e-3,
              "rnn_hidden_dim": 64, "label_dim": 128, "text_dim": 64, "kernel_size": [2, 3, 4, 5]})
