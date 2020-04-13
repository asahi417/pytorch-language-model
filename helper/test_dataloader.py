""" sentiment analysis model finetuning on hugginface.transformers

- checkpoint managers: different ckpt id will be given to different configuration
- dataset: sst/imdb dataset will be automatically fetched from source and compile as DataLoader
- multiGPU support
- command line interface for testing inference
"""

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

import util_hf_optimizer

dictConfig(
    dict(
        version=1,
        formatters={
            'f': {'format': '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'}
        },
        handlers={
            'h': {'class': 'logging.StreamHandler',
                  'formatter': 'f',
                  'level': logging.DEBUG}
        },
        root={
            'handlers': ['h'],
            'level': logging.DEBUG,
        }
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


def get_dataset(data_name: str = 'sst'):
    """ download dataset file and return dictionary including training/validation split """

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

    with open(os.path.join(CACHE_DIR, data_name, 'label.json'), 'w') as f:
        json.dump(label_dictionary, f)
    num_labels = len(list(label_dictionary.keys()))
    return data_split, num_labels


class TokenEncoder:
    """ Token encoder with transformers tokenizer """

    def __init__(self,
                 transformer: str,
                 max_seq_length: int = None):
        self.tokenizer = VALID_TOKENIZER[transformer].from_pretrained(transformer, cache_dir=CACHE_DIR)
        if max_seq_length and max_seq_length > self.tokenizer.max_len:
            raise ValueError('`max_seq_length should be less than %i' % self.tokenizer.max_len)
        self.max_seq_length = max_seq_length if max_seq_length else self.tokenizer.max_len
        LOGGER.info('max_sequence_length: %i' % self.max_seq_length)

    def __call__(self, text):
        # token_ids = self.tokenizer.encode(text)
        tokens_dict = self.tokenizer.encode_plus(text, max_length=self.max_seq_length, pad_to_max_length=True)
        token_ids = tokens_dict['input_ids']
        attention_mask = tokens_dict['attention_mask']
        return token_ids, attention_mask


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
        token_ids, attention_mask = self.token_encoder(self.data[idx])
        out_data = torch.tensor(token_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.float32)
        if self.label is None:
            return out_data, attention_mask
        else:
            out_label = torch.tensor(self.label[idx], dtype=torch.long)
            return out_data, attention_mask, out_label


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
                 **kwargs):
        LOGGER.info('*** initialize network ***')

        # checkpoint versioning
        self.param = ParameterManager(prefix=dataset, checkpoint=checkpoint, dataset=dataset, **kwargs)
        self.checkpoint_model = os.path.join(self.param.checkpoint_dir, 'model.pt')
        self.batch_size_validation = batch_size_validation if batch_size_validation else self.param('batch_size')

        # fix random seed
        random.seed(self.param('random_seed'))
        np.random.seed(self.param('random_seed'))
        torch.manual_seed(self.param('random_seed'))

        # model setup
        _, num_labels = get_dataset(self.param('dataset'))
        self.token_encoder = TokenEncoder(self.param('transformer'), self.param('max_seq_length'))
        self.param.remove_ckpt()

    def train(self):

        # setup data loader
        LOGGER.info('setup dataset')
        dataset_split, _ = get_dataset(self.param('dataset'))
        data_loader_train = torch.utils.data.DataLoader(
            Dataset(dataset_split[0][0], label=dataset_split[0][1], token_encoder=self.token_encoder),
            batch_size=self.param('batch_size'),
            shuffle=True,
            num_workers=NUM_WORKER,
            drop_last=True)

        try:
            with detect_anomaly():
                while True:
                    self.__epoch_train(data_loader_train)

        except RuntimeError:
            LOGGER.info(traceback.format_exc())
            LOGGER.info('*** RuntimeError (NaN found, see above log in detail) ***')

        except KeyboardInterrupt:
            LOGGER.info('*** KeyboardInterrupt ***')

            exit('nothing to be saved')

    def __epoch_train(self, data_loader):
        """ train on single epoch return flag which is True if training has been completed """

        for i, (inputs, attn_mask, outputs) in enumerate(data_loader, 1):

            print(inputs)
            print(attn_mask)
            print(outputs)
            exit()


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
    parser.add_argument('--batch-size', help='batch size', default=2, type=int)
    parser.add_argument('--batch-size-validation', help='batch size for validation', default=4, type=int)
    parser.add_argument('--warmup-step', help='warmup step', default=700, type=int)  # 6% of total step recommended
    parser.add_argument('--weight-decay', help='weight decay', default=0.0, type=float)
    parser.add_argument('--tolerance', help='tolerance for valid loss', default=None, type=float)
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
        max_seq_length=opt.max_seq_length
    )
    if not opt.inference_mode:
        classifier.train()
    else:
        while True:
            _inp = input('input sentence >>>')
            if _inp == 'q':
                break
            elif _inp == '':
                continue
            else:
                print(classifier.predict([_inp]))

