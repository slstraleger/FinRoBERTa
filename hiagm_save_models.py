import sys
from datetime import datetime

import numpy as np
import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

sys.path += ['HiAGM']

from torch import nn
from torch.utils.tensorboard import SummaryWriter
import os
import json
from torch.utils.data import DataLoader
from transformers import RobertaForMaskedLM
import torch
import HiAGM.helper.logger as logger
from HiAGM.helper.configure import Configure
from HiAGM.data_modules.vocab import Vocab
from HiAGM.train_modules.criterions import ClassificationLoss
from HiAGM.train_modules.trainer import Trainer
from HiAGM.helper.utils import save_checkpoint
import time

from HiAGM.data_modules.collator import Collator
from HiAGM.data_modules.dataset import ClassificationDataset

MAX_LENGTH = 64
BATCH_SIZE = 1
N_DATA_W = 0
LOAD_CHECKPOINTS = False


def custom_evaluate(epoch_predicts, epoch_labels, loss, vocab, train_or_test):
    l2_true = [x[1] for x in epoch_labels]
    l2_pred = []

    for sample_predict in epoch_predicts:
        np_sample_predict = np.array(sample_predict, dtype=np.float32)
        sample_predict_descent_idx = np.argsort(-np_sample_predict)
        l2_pred.append([i for i in sample_predict_descent_idx if i in set(l2_true)][0])

    precision = precision_score(l2_true, l2_pred, average='weighted', zero_division=0)
    recall = recall_score(l2_true, l2_pred, average='weighted', zero_division=0)
    f1 = f1_score(l2_true, l2_pred, average='weighted', zero_division=0)

    if train_or_test == 'TEST' and (f1 > best_performance):
        conf_matrix = confusion_matrix(l2_true, l2_pred, labels=list(set(l2_true)))
        display_labels = [vocab.i2v['label'][x].title() for x in set(l2_true)]
        rename_dict = {'Ma': 'M&A', 'Market_2': 'Market', 'Company Communication': 'Company Comm.'}
        for index, data in enumerate(display_labels):
            for key, value in rename_dict.items():
                if key in data:
                    display_labels[index] = data.replace(key, rename_dict[key])
        conf_matrix_to_save = ConfusionMatrixDisplay(conf_matrix,
                                                     display_labels=display_labels)

        fig, ax = plt.subplots(figsize=(8, 8))
        conf_matrix_to_save.plot(ax=ax, colorbar=False, cmap=plt.cm.Blues)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center')
        ax.set_title(f'{OUTPUT_NAME} Bottom Level Aspect Category')
        plt.subplots_adjust(bottom=0.2, left=0.2)
        plt.savefig(f'HiAGM_output/confusion_matrix_{OUTPUT_NAME}.svg', format='svg', dpi=1200)

    metrics = {'precision': precision, 'recall': recall, 'f1': f1, 'loss': loss}

    return metrics


class CustomTrainer(Trainer):
    def __init__(self, model, criterion, optimizer, vocab, config):
        super().__init__(model, criterion, optimizer, vocab, config)

    def run(self, data_loader, epoch, stage, mode='TRAIN'):
        """
        training epoch
        :param data_loader: Iteration[Dict{'token':tensor, 'label':tensor, }]
        :param epoch: int, log results of this epoch
        :param stage: str, e.g. 'TRAIN'/'DEV'/'TEST', figure out the corpus
        :param mode: str, ['TRAIN', 'EVAL'], train with backward pass while eval without it
        :return: metrics -> {'precision': 0.xx, 'recall': 0.xx, 'micro-f1': 0.xx, 'macro-f1': 0.xx}
        """
        predict_probs = []
        target_labels = []
        total_loss = 0.0
        num_batch = data_loader.__len__()

        for batch in tqdm.tqdm(data_loader):
            logits = self.model(batch)
            if self.config.train.loss.recursive_regularization.flag:
                recursive_constrained_params = self.model.hiagm.linear.weight
            else:
                recursive_constrained_params = None
            loss = self.criterion(logits,
                                  batch['label'].to(self.config.train.device_setting.device),
                                  recursive_constrained_params)
            total_loss += loss.item()

            if mode == 'TRAIN':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            predict_results = torch.sigmoid(logits).cpu().tolist()
            predict_probs.extend(predict_results)
            target_labels.extend(batch['label_list'])
        total_loss = total_loss / num_batch
        if mode == 'EVAL':
            metrics = custom_evaluate(predict_probs, target_labels, total_loss, self.vocab, stage)
            logger.info("%s performance at epoch %d --- Precision: %f, "
                        "Recall: %f, F1: %f, Loss: %f.\n"
                        % (stage, epoch,
                           metrics['precision'], metrics['recall'], metrics['f1'],
                           total_loss))
            return metrics


def set_optimizer(config, model):
    """
    :param config: helper.configure, Configure Object
    :param model: computational graph
    :return: torch.optim
    """
    params = model.optimize_params_dict()
    match config.train.optimizer.type:
        case "AdamW":
            return torch.optim.AdamW(lr=config.train.optimizer.learning_rate, params=params, fused=True,
                                     weight_decay=1e-4)
        case "SGD":
            return torch.optim.SGD(lr=config.train.optimizer.learning_rate, params=params, momentum=0.9)
        case "RMSprop":
            return torch.optim.RMSprop(lr=config.train.optimizer.learning_rate, params=params, momentum=0.9)
        case "Adagrad":
            return torch.optim.Adagrad(lr=config.train.optimizer.learning_rate, params=params)
        case "Adam":
            return torch.optim.Adam(lr=config.train.optimizer.learning_rate, params=params, fused=True,
                                    weight_decay=1e-4)


class ModifiedHiAGM(nn.Module):
    def __init__(self, config, vocab, model_type, model_mode='TRAIN', emb=None):
        """
        Hierarchy-Aware Global Model class
        :param config: helper.configure, Configure Object
        :param vocab: data_modules.vocab, Vocab Object
        :param model_type: Str, ('HiAGM-TP' for the serial variant of text propagation,
                                 'HiAGM-LA' for the parallel variant of multi-label soft attention,
                                 'Origin' without hierarchy-aware module)
        :param model_mode: Str, ('TRAIN', 'EVAL'), initialize with the pretrained word embedding if value is 'TRAIN'
        """
        super(ModifiedHiAGM, self).__init__()
        self.config = config
        self.vocab = vocab
        self.device = config.train.device_setting.device

        self.token_map, self.label_map = vocab.v2i['token'], vocab.v2i['label']
        if emb is not None:
            self.token_embedding = emb
            self.token_embedding.requires_grad = False
            for emb_layer in self.token_embedding.parameters():
                emb_layer.requires_grad = False
        else:
            from HiAGM.models.embedding_layer import EmbeddingLayer
            self.token_embedding = EmbeddingLayer(
                vocab_map=self.token_map,
                embedding_dim=config.embedding.token.dimension,
                vocab_name='token',
                config=config,
                padding_index=vocab.padding_index,
                pretrained_dir=config.embedding.token.pretrained_file,
                model_mode=model_mode,
                initial_type=config.embedding.token.init_type
            )

        DATAFLOW_TYPE = {
            'HiAGM-TP': 'serial',
            'HiAGM-LA': 'parallel',
            'Origin': 'origin'
        }

        self.dataflow_type = DATAFLOW_TYPE[model_type]

        from HiAGM.models.text_encoder import TextEncoder
        self.text_encoder = TextEncoder(config)
        from HiAGM.models.structure_model.structure_encoder import StructureEncoder
        self.structure_encoder = StructureEncoder(config=config,
                                                  label_map=vocab.v2i['label'],
                                                  device=self.device,
                                                  graph_model_type=config.structure_encoder.type)

        if self.dataflow_type == 'serial':
            from HiAGM.models.text_feature_propagation import HiAGMTP
            self.hiagm = HiAGMTP(config=config,
                                 device=self.device,
                                 graph_model=self.structure_encoder,
                                 label_map=self.label_map)
        elif self.dataflow_type == 'parallel':
            from HiAGM.models.multi_label_attention import HiAGMLA
            self.hiagm = HiAGMLA(config=config,
                                 device=self.device,
                                 graph_model=self.structure_encoder,
                                 label_map=self.label_map,
                                 model_mode=model_mode)
        else:
            from HiAGM.models.origin import Classifier
            self.hiagm = Classifier(config=config,
                                    vocab=vocab,
                                    device=self.device)

    def optimize_params_dict(self):
        """
        get parameters of the overall model
        :return: List[Dict{'params': Iteration[torch.Tensor],
                           'lr': Float (predefined learning rate for specified module,
                                        which is different from the others)
                          }]
        """
        params = list()
        params.append({'params': self.text_encoder.parameters()})
        params.append({'params': self.token_embedding.parameters()})
        params.append({'params': self.hiagm.parameters()})
        return params

    def forward(self, batch):
        embedding = self.token_embedding(batch['token'].to(self.config.train.device_setting.device))

        # set padded token tensors to zero as attention layers normally handle padded tokens
        padding_boolean = (batch['token'] != 1) * (batch['token'] != 0) * (batch['token'] != 2)
        embedding[~padding_boolean] = 0
        embedding = torch.roll(embedding, -1, 1)

        # get the length of sequences for dynamic rnn, (batch_size, 1)
        seq_len = batch['token_len']

        token_output = self.text_encoder(embedding, seq_len)

        logits = self.hiagm(token_output)

        return logits


class ModifiedDataset(ClassificationDataset):
    def __init__(self, config, vocab, stage='TRAIN', on_memory=False, corpus_lines=None):
        super(ModifiedDataset, self).__init__(config, vocab, stage, on_memory, corpus_lines)
        if on_memory:
            for data_point in self.data:
                label_list = []
                for label in data_point['label']:
                    label_list.append(self.vocab.v2i['label'][label])
                data_point['label'] = label_list

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
            assert sample['label'], 'Label is empty'
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
    collate_fn = Collator(config, vocab)
    train_dataset = ModifiedDataset(config, vocab, stage='TRAIN', on_memory=on_memory, corpus_lines=data['train'])
    train_loader = DataLoader(train_dataset,
                              batch_size=config.train.batch_size,
                              shuffle=True,
                              num_workers=config.train.device_setting.num_workers,
                              collate_fn=collate_fn,
                              pin_memory=True)

    val_dataset = ModifiedDataset(config, vocab, stage='VAL', on_memory=on_memory, corpus_lines=data['val'])
    val_loader = DataLoader(val_dataset,
                            batch_size=config.eval.batch_size,
                            shuffle=True,
                            num_workers=config.train.device_setting.num_workers,
                            collate_fn=collate_fn,
                            pin_memory=True)

    test_dataset = ModifiedDataset(config, vocab, stage='TEST', on_memory=on_memory, corpus_lines=data['test'])
    test_loader = DataLoader(test_dataset,
                             batch_size=config.eval.batch_size,
                             shuffle=True,
                             num_workers=config.train.device_setting.num_workers,
                             collate_fn=collate_fn,
                             pin_memory=True)

    return train_loader, val_loader, test_loader


def final_train(embedding_model_loc, output_name, end_epoch, params):
    NUM_KERNEL = params['text_dim'] // len(params['kernel_size'])

    print(f'Starting training final model ({output_name}) with parameters:', params)
    config = Configure(config={
        "data": {"dataset": "fiqa", "data_dir": "data/HiAGM/data/", "train_file": f"fiqa_train.json",
                 "val_file": f"fiqa_test.json", "test_file": "fiqa_test.json",
                 "prob_json": "fiqa_prob.json", "hierarchy": "fiqa.taxonomy"},
        "vocabulary": {"dir": "data/HiAGM/vocab", "vocab_dict": "word.dict", "max_token_vocab": 60000,
                       "label_dict": "label.dict"},
        "embedding": {"token": {"dimension": 1024, "type": "pretrain",
                                "pretrained_file": "data/HiAGM/vocab/roberta_vocab.txt",
                                "dropout": params['fully_connected_dropout'],
                                "init_type": "uniform"},
                      "label": {"dimension": params['label_dim'], "type": "random",
                                "dropout": params['fully_connected_dropout'],
                                "init_type": "kaiming_uniform"}},
        "text_encoder": {"max_length": MAX_LENGTH, "RNN": {"bidirectional": True, "num_layers": 1, "type": "GRU",
                                                           "hidden_dimension": params['rnn_hidden_dim'],
                                                           "dropout": params['gru_dropout']},
                         "CNN": {"kernel_size": params['kernel_size'], "num_kernel": NUM_KERNEL},
                         "topK_max_pooling": 1},
        "structure_encoder": {"type": "GCN", "node": {"type": "text", "dimension": params['label_dim'],
                                                      "dropout": params['structure_encoder_dropout']}},
        "model": {"type": "HiAGM-TP",
                  "linear_transformation": {"text_dimension": params['text_dim'],
                                            "node_dimension": params['label_dim'],
                                            "dropout": params['fully_connected_dropout']},
                  "classifier": {"num_layer": 1, "dropout": params['fully_connected_dropout']}},
        "train": {"optimizer": {"type": params['optimizer'], "learning_rate": params['learning_rate'],
                                "lr_decay": 0.8, "lr_patience": 1, "early_stopping": 4},
                  "batch_size": BATCH_SIZE, "start_epoch": 0, "end_epoch": end_epoch,
                  "loss": {"classification": "BCEWithLogitsLoss",
                           "recursive_regularization": {"flag": True,
                                                        "penalty": params['recursive_regularization']}},
                  "device_setting": {"device": "cuda", "visible_device_list": "0", "num_workers": N_DATA_W},
                  "checkpoint": {"dir": f"HiAGM_output/final/checkpoint/{output_name}/", "max_number": 0,
                                 "save_best": ["f1"]}},
        "eval": {"batch_size": BATCH_SIZE, "threshold": 0.5},
        "test": {"best_checkpoint": "best_HiAGM-TP", "batch_size": BATCH_SIZE},
        "log": {"level": "info", "filename": f"HiAGM_output/final/{output_name}/logs/log.log"}})
    # loading corpus and generate vocabulary
    corpus_vocab = Vocab(config, special_token=['1', '3'])

    # get data
    def get_preloaded_data():
        return {
            'train': [json.loads(line) for line in open(os.path.join(config.data.data_dir, config.data.train_file))],
            'val': [json.loads(line) for line in open(os.path.join(config.data.data_dir, config.data.val_file))],
            'test': [json.loads(line) for line in open(os.path.join(config.data.data_dir, config.data.test_file))]}

    train_loader, val_loader, test_loader = data_loaders(config, corpus_vocab, get_preloaded_data())
    global best_performance
    best_performance = 0
    now = datetime.now().strftime("%m-%d_%H-%M-%S")
    writer = SummaryWriter(f"HiAGM_output/final/runs/{output_name}_{now}")

    # build up model
    hiagm = ModifiedHiAGM(config, corpus_vocab, model_type=config.model.type, model_mode='TRAIN',
                          emb=RobertaForMaskedLM.from_pretrained(embedding_model_loc).roberta.embeddings)
    hiagm.to(config.train.device_setting.device)
    # define training objective & optimizer
    criterion = ClassificationLoss(os.path.join(config.data.data_dir, config.data.hierarchy), corpus_vocab.v2i['label'],
                                   recursive_penalty=config.train.loss.recursive_regularization.penalty,
                                   recursive_constraint=config.train.loss.recursive_regularization.flag)
    optimize = set_optimizer(config, hiagm)

    # get epoch trainer
    trainer = CustomTrainer(model=hiagm, criterion=criterion, optimizer=optimize, vocab=corpus_vocab, config=config)

    # set origin log
    model_checkpoint = config.train.checkpoint.dir
    wait = 0

    if not os.path.isdir(model_checkpoint):
        os.mkdir(model_checkpoint)

    from torchinfo import summary
    print(summary(hiagm))

    # train
    for epoch in range(config.train.start_epoch, config.train.end_epoch):
        start_time = time.time()
        trainer.train(train_loader, epoch)
        train_perf = trainer.eval(train_loader, epoch, 'TRAIN')
        writer.add_scalar("precision/train", train_perf['precision'], epoch)
        writer.add_scalar("recall/train", train_perf['recall'], epoch)
        writer.add_scalar("f1/train", train_perf['f1'], epoch)
        writer.add_scalar("loss/train", train_perf['loss'], epoch)
        performance = trainer.eval(test_loader, epoch, 'TEST')
        writer.add_scalar("precision/test", performance['precision'], epoch)
        writer.add_scalar("recall/test", performance['recall'], epoch)
        writer.add_scalar("f1/test", performance['f1'], epoch)
        writer.add_scalar("loss/test", performance['loss'], epoch)
        # saving best model and check model
        '''
        if not performance['f1'] >= best_performance:
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

        if performance['f1'] > best_performance:
            wait = 0
            logger.info('Improve weighted F1 {} --> {}'.format(best_performance, performance['f1']))
            best_performance = performance['f1']
            save_checkpoint({'epoch': epoch, 'model_type': config.model.type, 'state_dict': hiagm.state_dict(),
                             'best_performance': best_performance, 'optimizer': optimize.state_dict()},
                            os.path.join(model_checkpoint, 'best_' + output_name + '.pt'))
            # save_confusion_matrix(os.path.join(model_checkpoint, 'conf_matrix_best_' + output_name + '.pt'))
        logger.info('Epoch {} Time Cost {} secs.'.format(epoch, time.time() - start_time))

    writer.close()


if __name__ == '__main__':
    best_performance = 0
    OUTPUT_NAME = 'HiAGM-RoBERTa'
    hyper_pars_1 = {"learning_rate": 5e-4, "optimizer": "Adam", "structure_encoder_dropout": 0.2, "gru_dropout": 0.1,
                    "fully_connected_dropout": 0.5, "recursive_regularization": 1e-3, "rnn_hidden_dim": 64,
                    "label_dim": 128, "text_dim": 64, "kernel_size": [2, 3, 4, 5]}
    final_train('roberta-large', OUTPUT_NAME, 50, hyper_pars_1)
    best_performance = 0
    OUTPUT_NAME = 'HiAGM-RoBERTa-TRC2'
    hyper_pars_2 = {"learning_rate": 1e-4, "optimizer": "Adam", "structure_encoder_dropout": 0.3, "gru_dropout": 0.2,
                    "fully_connected_dropout": 0.5, "recursive_regularization": 1e-3, "rnn_hidden_dim": 64,
                    "label_dim": 128, "text_dim": 64, "kernel_size": [2, 3, 4, 5]}
    final_train('scripts_used_for_modelling/RoBERTa-TRC2/roberta-training/models/peft_model_trc2',
                OUTPUT_NAME, 50, hyper_pars_2)
    best_performance = 0
    OUTPUT_NAME = 'HiAGM-RoBERTa-TRC2-FPB'
    hyper_pars_3 = {"learning_rate": 1e-4, "optimizer": "Adam", "structure_encoder_dropout": 0.3, "gru_dropout": 0.2,
                    "fully_connected_dropout": 0.5, "recursive_regularization": 1e-3, "rnn_hidden_dim": 64,
                    "label_dim": 128, "text_dim": 64, "kernel_size": [2, 3, 4, 5]}
    final_train('scripts_used_for_modelling/RoBERTa-TRC2-FPB/roberta-training/models/peft_model_fpb',
                OUTPUT_NAME, 50, hyper_pars_3)
    print('HiAGM saving finished')
