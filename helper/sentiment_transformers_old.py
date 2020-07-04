""" self-contained sentiment analysis model finetuning on hugginface.transformers
- checkpoint managers: different ckpt id will be given to different configuration
- dataset: sst/imdb dataset will be automatically fetched from source and compile as DataLoader
- multiGPU support
- command line interface for testing inference
- see https://huggingface.co/transformers/_modules/transformers/optimization.html#AdamW for AdamW and linear scheduler
"""

import math
import copy
import traceback
import argparse
import os
import random
import string
import json
import logging
import shutil
import transformers
import torchtext
import torch
import numpy as np
from torch import optim
from torch import nn
from torch.autograd import detect_anomaly
from torch.utils.tensorboard import SummaryWriter
from glob import glob
from logging.config import dictConfig

dictConfig(
    dict(
        version=1,
        formatters={'f': {'format': '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'}},
        handlers={'h': {'class': 'logging.StreamHandler', 'formatter': 'f', 'level': logging.DEBUG}},
        root={'handlers': ['h'], 'level': logging.DEBUG}
    )
)
LOGGER = logging.getLogger()
NUM_WORKER = int(os.getenv("NUM_WORKER", '4'))
PROGRESS_INTERVAL = int(os.getenv("PROGRESS_INTERVAL", '100'))
CACHE_DIR = os.getenv("CACHE_DIR", './cache')
CKPT_DIR = os.getenv("CKPT_DIR", './ckpt')
VALID_TRANSFORMER_SEQUENCE_CLASSIFICATION = {
    'xlm-roberta-large': transformers.XLMRobertaForSequenceClassification,
    'xlm-roberta-base': transformers.XLMRobertaForSequenceClassification,
    'bert-base-multilingual-cased': transformers.BertForSequenceClassification}
VALID_TOKENIZER = {
    'xlm-roberta-large': transformers.XLMRobertaTokenizer,
    'xlm-roberta-base': transformers.XLMRobertaTokenizer,
    'bert-base-multilingual-cased': transformers.BertTokenizer}
assert set(VALID_TRANSFORMER_SEQUENCE_CLASSIFICATION.keys()) == set(VALID_TOKENIZER.keys())


def get_dataset(data_name: str = 'sst',
                get_label_dict_only: bool = False):
    """ download dataset file and return dictionary including training/validation split """
    if get_label_dict_only:
        return json.load(open(os.path.join(CACHE_DIR, data_name, 'label.json')))

    def decode_data(iterator, file_prefix, label_dict: dict):
        if os.path.exists(file_prefix + '.text') and os.path.exists(file_prefix + '.label'):
            list_of_text = open(file_prefix + '.text', 'r').read().split('\n')
            list_of_label = [int(l) for l in open(file_prefix + '.label', 'r').read().split('\n')]
            assert len(list_of_label) == len(list_of_text)
            return label_dict, (list_of_text, list_of_label)

        list_text = []
        list_label = []
        for i in iterator:
            if data_name == 'sst' and i.label == 'neutral':
                continue
            if i.label not in label_dict.keys():
                label_dict[i.label] = len(label_dict)
            list_text.append(' '.join(i.text))
            list_label.append(str(label_dict[i.label]))

        with open(file_prefix + '.text', 'w') as f_writer:
            f_writer.write('\n'.join(list_text))

        with open(file_prefix + '.label', 'w') as f_writer:
            f_writer.write('\n'.join(list_label))

        return label_dict, None

    data_field, label_field = torchtext.data.Field(sequential=True), torchtext.data.Field(sequential=False)
    if data_name == 'imdb':
        iterator_split = torchtext.datasets.IMDB.splits(data_field, root=CACHE_DIR, label_field=label_field)
    elif data_name == 'sst':
        iterator_split = torchtext.datasets.SST.splits(data_field, root=CACHE_DIR, label_field=label_field)
    else:
        raise ValueError('unknown dataset: %s' % data_name)

    data_split, data = list(), None
    if os.path.exists(os.path.join(CACHE_DIR, data_name, 'label.json')):
        label_dictionary = json.load(open(os.path.join(CACHE_DIR, data_name, 'label.json')))
    else:
        label_dictionary = dict()

    for name, it in zip(['train', 'valid', 'test'], iterator_split):
        _file_prefix = os.path.join(CACHE_DIR, data_name, name)
        label_dictionary, data = decode_data(it, file_prefix=_file_prefix, label_dict=label_dictionary)
        if data is None:
            _, data = decode_data(it, file_prefix=_file_prefix, label_dict=label_dictionary)
        data_split.append(data)
        LOGGER.info('dataset %s/%s: %i' % (data_name, name, len(data[0])))
    return data_split, label_dictionary


class TokenEncoder:
    """ Token encoder with transformers tokenizer """

    def __init__(self,
                 transformer: str,
                 max_seq_length: int = None):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(transformer, cache_dir=CACHE_DIR)
        if max_seq_length and max_seq_length > self.tokenizer.max_len:
            raise ValueError('`max_seq_length should be less than %i' % self.tokenizer.max_len)
        self.max_seq_length = max_seq_length if max_seq_length else self.tokenizer.max_len
        LOGGER.info('max_sequence_length: %i' % self.max_seq_length)

    def __call__(self, text):
        # tokens_dict = self.tokenizer.encode_plus(text, max_length=self.max_seq_length, pad_to_max_length=True)
        return self.tokenizer.encode_plus(text, max_length=self.max_seq_length, pad_to_max_length=True)
        # token_ids = tokens_dict['input_ids']
        # attention_mask = tokens_dict['attention_mask']
        # return token_ids, attention_mask


class Dataset(torch.utils.data.Dataset):
    """ torch.utils.data.Dataset instance """

    def __init__(self,
                 data: list,
                 token_encoder,
                 label: list=None):
        self.data = data
        if label is None:
            self.label = None
        else:
            self.label = [int(l) for l in label]
        self.token_encoder = token_encoder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        encode = self.token_encoder(self.data[idx])
        if self.label is not None:
            encode['labels'] = self.label[idx]
        float_list = ['attention_mask']
        encode = {k: torch.tensor(v, dtype=torch.float32) if k in float_list else torch.tensor(v, dtype=torch.long)
                  for k, v in encode.items()}
        return encode
        # out_data = torch.tensor(token_ids, dtype=torch.long)
        # attention_mask = torch.tensor(attention_mask, dtype=torch.float32)
        # if self.label is None:
        #     return out_data, attention_mask
        # else:
        #     out_label = torch.tensor(self.label[idx], dtype=torch.long)
        #     return out_data, attention_mask, out_label


# class Dataset(torch.utils.data.Dataset):
#     """ torch.utils.data.Dataset with transformer tokenizer """
#
#     def __init__(self, data: list, transformer_tokenizer,
#                  max_seq_length: int = None, label: list = None, pad_to_max_length: bool = True):
#         self.data = data  # list of half-space split tokens
#         if label is None:
#             self.label = None
#         else:
#             self.label = [int(l) for l in label]
#         self.pad_to_max_length = pad_to_max_length
#         self.tokenizer = transformer_tokenizer
#         if max_seq_length and max_seq_length > self.tokenizer.max_len:
#             raise ValueError('`max_seq_length should be less than %i' % self.tokenizer.max_len)
#         self.max_seq_length = max_seq_length if max_seq_length else self.tokenizer.max_len
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         encode = self.tokenizer.encode_plus(self.data[idx], max_length=self.max_seq_length, pad_to_max_length=self.pad_to_max_length)
#         if self.label is not None:
#             encode['labels'] = self.label[idx]
#         float_list = ['attention_mask']
#         encode = {k: torch.tensor(v, dtype=torch.float32) if k in float_list else torch.tensor(v, dtype=torch.long)
#                   for k, v in encode.items()}
#         return encode


class ParameterManager:
    """ Parameter manager for model training """

    def __init__(self,
                 prefix: str = None,
                 checkpoint: str = None,
                 **kwargs):

        """ Parameter manager for model training
         Parameter
        -------------------
        prefix: prefix to filename
        checkpoint: existing checkpoint name if you want to load
        kwargs: model parameters
        """
        self.checkpoint_dir, self.parameter = self.__versioning(kwargs, checkpoint, prefix)
        LOGGER.info('checkpoint: %s' % self.checkpoint_dir)
        for k, v in self.parameter.items():
            LOGGER.info(' - [param] %s: %s' % (k, str(v)))

    def __call__(self, key):
        """ retrieve a parameter """
        if key not in self.parameter.keys():
            raise ValueError('unknown parameter %s' % key)
        return self.parameter[key]

    def remove_ckpt(self):
        shutil.rmtree(self.checkpoint_dir)

    @staticmethod
    def __versioning(parameter: dict = None, checkpoint: str = None, prefix: str = None):
        """ Checkpoint versioner: Either of `config` or `checkpoint` need to be specified (`config` has priority)
         Parameter
        ---------------------
        parameter: parameter configuration to find same setting checkpoint
        checkpoint: existing checkpoint to be loaded
         Return
        --------------------
        path_to_checkpoint: path to new checkpoint dir
        parameter: parameter
        """

        def random_string(string_length=10, exceptions: list = None):
            """ Generate a random string of fixed length """
            while True:
                letters = string.ascii_lowercase
                random_letters = ''.join(random.choice(letters) for i in range(string_length))
                if exceptions is None or random_letters not in exceptions:
                    break
            return random_letters

        if checkpoint is None and parameter is None:
            raise ValueError('either of `checkpoint` or `parameter` is needed.')

        if checkpoint is None:
            LOGGER.info('issue new checkpoint id')
            # check if there are any checkpoints with same hyperparameters
            version_name = []
            for parameter_path in glob(os.path.join(CKPT_DIR, '*/hyperparameters.json')):
                _dir = parameter_path.replace('/hyperparameters.json', '')
                _dict = json.load(open(parameter_path))
                version_name.append(_dir.split('/')[-1])
                if parameter == _dict:
                    inp = input('found a checkpoint with same configuration\n'
                                'enter to delete the existing checkpoint %s\n'
                                'or exit by type anything but not empty' % _dir)
                    if inp == '':
                        shutil.rmtree(_dir)
                    else:
                        exit()

            new_checkpoint = random_string(exceptions=version_name)
            if prefix:
                new_checkpoint = '_'.join([prefix, new_checkpoint])
            new_checkpoint_path = os.path.join(CKPT_DIR, new_checkpoint)
            os.makedirs(new_checkpoint_path, exist_ok=True)
            with open(os.path.join(new_checkpoint_path, 'hyperparameters.json'), 'w') as _f:
                json.dump(parameter, _f)
            return new_checkpoint_path, parameter

        else:
            LOGGER.info('load existing checkpoint')
            checkpoints = glob(os.path.join(CKPT_DIR, checkpoint, 'hyperparameters.json'))
            if len(checkpoints) >= 2:
                raise ValueError('Checkpoints are duplicated: %s' % str(checkpoints))
            elif len(checkpoints) == 0:
                raise ValueError('No checkpoint: %s' % os.path.join(CKPT_DIR, checkpoint))
            else:
                parameter = json.load(open(checkpoints[0]))
                target_checkpoints_path = checkpoints[0].replace('/hyperparameters.json', '')
                return target_checkpoints_path, parameter


class TransformerSequenceClassifier:

    def __init__(self,
                 dataset: str,
                 batch_size_validation: int = None,
                 checkpoint: str = None,
                 inference_mode: bool = False,
                 **kwargs):

        self.inference_mode = inference_mode
        if self.inference_mode:
            LOGGER.info('*** initialize network (INFERENCE MODE) ***')
        else:
            LOGGER.info('*** initialize network ***')

        # checkpoint versioning
        self.param = ParameterManager(prefix=dataset, checkpoint=checkpoint, dataset=dataset, **kwargs)
        self.checkpoint_model = os.path.join(self.param.checkpoint_dir, 'model.pt')
        self.batch_size_validation = batch_size_validation if batch_size_validation else self.param('batch_size')

        # fix random seed
        random.seed(self.param('random_seed'))
        np.random.seed(self.param('random_seed'))
        torch.manual_seed(self.param('random_seed'))

        # model/dataset setup
        if self.inference_mode:
            label_dict = json.load(open(os.path.join(CACHE_DIR, self.param('dataset'), 'label.json')))
            self.dataset_split = None
        else:
            self.dataset_split, label_dict = get_dataset(self.param('dataset'))
            with open(os.path.join(CACHE_DIR, self.param('dataset'), 'label.json'), 'w') as f:
                json.dump(label_dict, f)
        self.id_to_label = dict([(str(v), str(k)) for k, v in label_dict.items()])

        # self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.param('transformer'), cache_dir=CACHE_DIR)
        self.token_encoder = TokenEncoder(self.param('transformer'), self.param('max_seq_length'))
        self.config = transformers.AutoConfig.from_pretrained(
            self.param('transformer'),
            num_labels=len(self.id_to_label),
            id2label=self.id_to_label,
            label2id=label_dict,
            cache_dir=CACHE_DIR,
        )
        self.model_seq_cls = transformers.AutoModelForSequenceClassification.from_pretrained(
            self.param('transformer'), config=self.config
        )

        # GPU allocation
        self.n_gpu = torch.cuda.device_count()
        self.data_parallel = False
        self.device = 'cuda' if self.n_gpu > 0 else 'cpu'
        self.model_seq_cls.to(self.device)
        if self.n_gpu > 1:
            self.data_parallel = True
            self.model_seq_cls = torch.nn.DataParallel(self.model_seq_cls.cuda())
            LOGGER.info('WARNING: torch.nn.DataParallel is not tested')
        else:
            self.n_gpu = 0
        LOGGER.info('running on %i GPUs' % self.n_gpu)

        # optimizer
        if self.inference_mode:
            self.optimizer = self.scheduler = None
        else:
            if self.param("optimizer") == 'adamw':
                self.optimizer = transformers.AdamW(
                    self.model_seq_cls.parameters(), lr=self.param('lr'), weight_decay=self.param('weight_decay'))
            elif self.param("optimizer") == 'adam':
                self.optimizer = optim.Adam(
                    self.model_seq_cls.parameters(), lr=self.param('lr'), weight_decay=self.param('weight_decay'))
            elif self.param("optimizer") == 'sgd':
                self.optimizer = optim.SGD(
                    self.model_seq_cls.parameters(), lr=self.param('lr'), weight_decay=self.param('weight_decay'))
            else:
                raise ValueError('bad optimizer: %s' % self.param("optimizer"))

            # scheduler
            if self.param('scheduler') == 'constant':
                self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1, last_epoch=-1)
            elif self.param('scheduler') == 'linear':
                self.scheduler = transformers.get_linear_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=self.param('warmup_step'),
                    num_training_steps=self.param('total_step'))
            else:
                raise ValueError('bad scheduler: %s' % self.param('scheduler'))

        # load ckpt
        if os.path.exists(self.checkpoint_model):
            ckpt = torch.load(self.checkpoint_model, map_location='cpu')
            self.__step = ckpt['step']  # num of training step
            self.__epoch = ckpt['epoch']  # num of epoch
            self.__best_val_loss = ckpt['best_val_loss']
            self.__best_val_loss_step = ckpt['best_val_loss_step']
            self.__best_model_wts = ckpt['best_model_state_dict']
            LOGGER.info('load ckpt from %s' % self.checkpoint_model)
            if self.inference_mode:
                self.model_seq_cls.load_state_dict(ckpt['best_model_state_dict'])
                LOGGER.info('load best ckpt from step %i / %i' % (self.__best_val_loss_step, self.__step))
            else:
                self.model_seq_cls.load_state_dict(ckpt['model_state_dict'])
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        else:
            self.__step = 0
            self.__epoch = 0
            self.__best_val_loss = None
            self.__best_val_loss_step = None
            self.__best_model_wts = None

        # log
        if self.inference_mode:
            self.writer = None
        else:
            self.writer = SummaryWriter(log_dir=self.param.checkpoint_dir)

    def release_cache(self):
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def predict(self,
                x: list,
                batch_size: int = 1):
        """ model inference
        :param x: list of input
        :param batch_size: batch size for inference
        :return: (prediction, prob)
            prediction is a list of predicted label, and prob is a list of dictionary with each probability
        """
        self.model_seq_cls.eval()
        data_loader = torch.utils.data.DataLoader(
            Dataset(x, token_encoder=self.token_encoder), batch_size=min(batch_size, len(x)))
        prediction, prob = [], []
        for inputs, attn_mask in data_loader:
            inputs = inputs.to(self.device)
            attn_mask = attn_mask.to(self.device)
            outputs = self.model_seq_cls(inputs, attention_mask=attn_mask)
            logit = outputs[0]
            _, _pred = torch.max(logit, dim=1)
            _pred_list = _pred.cpu().tolist()
            _prob_list = torch.nn.functional.softmax(logit, dim=1).cpu().tolist()
            prediction += [self.id_to_label[str(_p)] for _p in _pred_list]
            prob += [dict(
                [(self.id_to_label[str(i)], float(pr))
                 for i, pr in enumerate(_p)]
            ) for _p in _prob_list]
        return prediction, prob

    def train(self):
        if self.inference_mode:
            raise ValueError('model is on an inference mode')

        # setup data loader
        LOGGER.info('setup dataset')
        dataset_split, _ = get_dataset(self.param('dataset'))
        data_loader_train = torch.utils.data.DataLoader(
            # Dataset(dataset_split[0][0], label=dataset_split[0][1], transformer_tokenizer=self.tokenizer),
            Dataset(dataset_split[0][0], label=dataset_split[0][1], token_encoder=self.token_encoder),
            batch_size=self.param('batch_size'),
            shuffle=True,
            num_workers=NUM_WORKER,
            drop_last=True)
        data_loader_valid = torch.utils.data.DataLoader(
            Dataset(dataset_split[1][0], label=dataset_split[1][1], token_encoder=self.token_encoder),
            batch_size=self.batch_size_validation,
            num_workers=NUM_WORKER)
        if len(dataset_split) > 2:
            data_loader_test = torch.utils.data.DataLoader(
                Dataset(dataset_split[2][0], label=dataset_split[2][1], token_encoder=self.token_encoder),
                batch_size=self.param('batch_size'))
        else:
            data_loader_test = None

        LOGGER.info('start training from step %i (epoch: %i)' % (self.__step, self.__epoch))
        try:
            with detect_anomaly():
                while True:

                    if_training_finish = self.__epoch_train(data_loader_train)
                    self.release_cache()

                    if_early_stop = self.__epoch_valid(data_loader_valid, prefix='valid')
                    self.release_cache()

                    if if_training_finish or if_early_stop:
                        if data_loader_test:
                            self.__epoch_valid(data_loader_valid, prefix='test')
                        break
                    self.__epoch += 1

        except RuntimeError:
            LOGGER.info(traceback.format_exc())
            LOGGER.info('*** RuntimeError (NaN found, see above log in detail) ***')

        except KeyboardInterrupt:
            LOGGER.info('*** KeyboardInterrupt ***')

        if self.__best_val_loss is None:
            self.param.remove_ckpt()
            exit('nothing to be saved')

        LOGGER.info('[training completed] best model: valid loss %0.3f at step %i'
                    % (self.__best_val_loss, self.__best_val_loss_step))

        if self.data_parallel:
            model_wts = self.model_seq_cls.module.state_dict()
        else:
            model_wts = self.model_seq_cls.state_dict()

        torch.save({
            'model_state_dict': model_wts,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'step': self.__step,
            'epoch': self.__epoch,
            'best_val_loss': self.__best_val_loss,
            'best_val_loss_step': self.__best_val_loss_step,
            'best_model_state_dict': self.__best_model_wts
        }, self.checkpoint_model)
        self.writer.close()
        LOGGER.info('ckpt saved at %s' % self.checkpoint_model)

    def __epoch_train(self, data_loader):
        """ train on single epoch return flag which is True if training has been completed """
        self.model_seq_cls.train()

        for i, encode in enumerate(data_loader, 1):

            encode = {k: v.to(self.device) for k, v in encode.items()}

            # zero the parameter gradients
            self.optimizer.zero_grad()
            # forward: output prediction and get loss
            model_outputs = self.model_seq_cls(**encode)
            loss, logit = model_outputs[0:2]

            if self.data_parallel:
                loss = torch.mean(loss)

            _, pred = torch.max(logit, 1)
            # backward: calculate gradient
            loss.backward()
            # gradient clip
            if self.param('clip') is not None:
                nn.utils.clip_grad_norm_(self.model_seq_cls.parameters(), self.param('clip'))
            # optimizer and scheduler step
            self.optimizer.step()
            self.scheduler.step()
            # instantaneous accuracy, loss, and learning rate
            inst_accuracy = ((pred == encode['labels']).cpu().float().mean()).item()
            inst_loss = loss.cpu().item()
            inst_lr = self.optimizer.param_groups[0]['lr']
            # log
            self.writer.add_scalar('train/loss', inst_loss, self.__step)
            self.writer.add_scalar('train/accuracy', inst_accuracy, self.__step)
            self.writer.add_scalar('learning_rate', inst_lr, self.__step)

            if self.__step % PROGRESS_INTERVAL == 0:
                LOGGER.info(' * (step %i) accuracy: %.3f, loss: %.3f, lr: %0.8f'
                            % (self.__step, inst_accuracy, inst_loss, inst_lr))

            self.__step += 1

            if self.__step >= self.param('total_step'):
                LOGGER.info('reached total step')
                return True

        return False

    def __epoch_valid(self, data_loader, prefix: str='valid'):
        """ validation/test """
        self.model_seq_cls.eval()
        list_accuracy, list_loss = [], []

        for inputs, attn_mask, outputs in data_loader:

            inputs = inputs.to(self.device)
            attn_mask = attn_mask.to(self.device)
            outputs = outputs.to(self.device)

            model_outputs = self.model_seq_cls(inputs, attention_mask=attn_mask, labels=outputs)
            loss, logit = model_outputs[0:2]
            if self.data_parallel:
                loss = torch.mean(loss)
            _, pred = torch.max(logit, 1)
            list_accuracy.append(((pred == outputs).cpu().float().mean()).item())
            list_loss.append(loss.cpu().item())

        accuracy, loss = float(np.mean(list_accuracy)), float(np.mean(list_loss))

        self.writer.add_scalar('%s/accuracy' % prefix, accuracy, self.__epoch)
        self.writer.add_scalar('%s/loss' % prefix, loss, self.__epoch)
        LOGGER.info('[epoch %i] (%s) accuracy: %.3f, loss: %.3f' % (self.__epoch, prefix, accuracy, loss))

        if self.__best_val_loss is None or loss < self.__best_val_loss:
            self.__best_val_loss = loss
            self.__best_val_loss_step = self.__step
            if self.data_parallel:
                self.__best_model_wts = copy.deepcopy(self.model_seq_cls.module.state_dict())
            else:
                self.__best_model_wts = copy.deepcopy(self.model_seq_cls.state_dict())
        else:
            loss_margin = loss - self.__best_val_loss
            if self.param('tolerance') is not None and self.param('tolerance') < loss_margin:
                LOGGER.info('early stop as loss exceeds tolerance: %0.2f < %0.2f '
                            % (self.param('tolerance'), loss_margin))
                return True
        return False


def get_options():
    parser = argparse.ArgumentParser(
        description='finetune transformers to sentiment analysis',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--data', help='data (imdb/sst)', default='sst', type=str)
    parser.add_argument('--transformer',
                        help='language model (%s)' % str(VALID_TRANSFORMER_SEQUENCE_CLASSIFICATION.keys()),
                        default='xlm-roberta-base',
                        type=str)
    parser.add_argument('--max-seq-length',
                        help='max sequence length (use same length as used in pre-training if not provided)',
                        default=None,
                        type=int)
    parser.add_argument('--random-seed', help='random seed', default=1234, type=int)
    parser.add_argument('--lr', help='learning rate', default=2e-5, type=float)
    parser.add_argument('--clip', help='gradient clip', default=None, type=float)
    parser.add_argument('--optimizer', help='optimizer', default='adam', type=str)
    parser.add_argument('--scheduler', help='scheduler', default='linear', type=str)
    parser.add_argument('--total-step', help='total training step', default=13000, type=int)
    parser.add_argument('--batch-size', help='batch size', default=16, type=int)
    parser.add_argument('--batch-size-validation',
                        help='batch size for validation (smaller size to save memory)',
                        default=4,
                        type=int)
    parser.add_argument('--warmup-step', help='warmup step (6 percent of total is recommended)', default=700, type=int)
    parser.add_argument('--weight-decay', help='weight decay', default=0.0, type=float)
    parser.add_argument('--tolerance', help='early stop tolerance in terms of valid loss', default=None, type=float)
    parser.add_argument('--checkpoint', help='checkpoint to load', default=None, type=str)
    parser.add_argument('--inference-mode', help='inference mode', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    opt = get_options()
    classifier = TransformerSequenceClassifier(
        batch_size_validation=opt.batch_size_validation,
        checkpoint=opt.checkpoint,
        dataset=opt.data,
        transformer=opt.transformer,
        random_seed=opt.random_seed,
        lr=opt.lr,
        clip=opt.clip,
        optimizer=opt.optimizer,
        scheduler=opt.scheduler,
        total_step=opt.total_step,
        warmup_step=opt.warmup_step,
        tolerance=opt.tolerance,
        weight_decay=opt.weight_decay,
        batch_size=opt.batch_size,
        max_seq_length=opt.max_seq_length,
        inference_mode=opt.inference_mode
    )
    if classifier.inference_mode:
        while True:
            _inp = input('input sentence >>>')
            if _inp == 'q':
                break
            elif _inp == '':
                continue
            else:
                predictions, probs = classifier.predict([_inp])
                print(predictions)
                print(probs)

    else:
        classifier.train()