""" self-contained NER finetuning on hugginface.transformers (conll_2003/wnut_17)

- checkpoint managers: different ckpt id will be given to different configuration
- dataset: sst/imdb dataset will be automatically fetched from source and compile as DataLoader
- multiGPU support
- command line interface for testing inference
- see https://huggingface.co/transformers/_modules/transformers/optimization.html#AdamW for AdamW and linear scheduler
"""

import traceback
import argparse
import os
import random
import string
import json
import logging
import shutil
import transformers
import torch
from time import time
from torch import optim
from torch import nn
from torch.autograd import detect_anomaly
from torch.utils.tensorboard import SummaryWriter
from glob import glob
from logging.config import dictConfig
from itertools import chain
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report, accuracy_score


class ParameterManager:
    """ Parameter manager for model training """

    def __init__(self, prefix: str = None, checkpoint: str = None, **kwargs):

        """ Parameter manager for model training

         Parameter
        -------------------
        prefix: prefix to filename
        checkpoint: existing checkpoint name if you want to load
        kwargs: model parameters
        """
        self.checkpoint_dir, self.parameter = self.__versioning(kwargs, checkpoint, prefix)
        self.__dict__.update(self.parameter)

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


def get_options():
    parser = argparse.ArgumentParser(
        description='finetune transformers to sentiment analysis',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-c', '--checkpoint', help='checkpoint to load', default=None, type=str)
    parser.add_argument('-d', '--data', help='data conll_2003/wnut_17', default='wnut_17', type=str)
    parser.add_argument('-t', '--transformer', help='pretrained language model', default='xlm-roberta-base', type=str)
    parser.add_argument('-m', '--max-seq-length',
                        help='max sequence length (use same length as used in pre-training if not provided)',
                        default=128,
                        type=int)
    parser.add_argument('-b', '--batch-size', help='batch size', default=16, type=int)
    parser.add_argument('--random-seed', help='random seed', default=1234, type=int)
    parser.add_argument('--lr', help='learning rate', default=2e-5, type=float)
    parser.add_argument('--optimizer', help='optimizer', default='adam', type=str)
    parser.add_argument('--scheduler', help='scheduler', default='linear', type=str)
    parser.add_argument('--total-step', help='total training step', default=13000, type=int)
    parser.add_argument('--batch-size-validation',
                        help='batch size for validation (smaller size to save memory)',
                        default=2,
                        type=int)
    parser.add_argument('--warmup-step', help='warmup step (6 percent of total is recommended)', default=700, type=int)
    parser.add_argument('--weight-decay', help='weight decay', default=0.0, type=float)
    parser.add_argument('--early-stop', help='value of accuracy drop for early stop', default=0.1, type=float)
    parser.add_argument('--inference-mode', help='inference mode', action='store_true')
    parser.add_argument('--fp16', help='fp16', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    opt = get_options()
    classifier = TransformerTokenClassification(
        batch_size_validation=opt.batch_size_validation,
        checkpoint=opt.checkpoint,
        dataset=opt.data,
        transformer=opt.transformer,
        random_seed=opt.random_seed,
        lr=opt.lr,
        optimizer=opt.optimizer,
        scheduler=opt.scheduler,
        total_step=opt.total_step,
        warmup_step=opt.warmup_step,
        weight_decay=opt.weight_decay,
        batch_size=opt.batch_size,
        max_seq_length=opt.max_seq_length,
        inference_mode=opt.inference_mode,
        early_stop=opt.early_stop,
        fp16=opt.fp16
    )
    if classifier.inference_mode:

        predictions = classifier.predict(['I live in London', '東京は今日も暑いです'])
        print(predictions)

        # while True:
        #     _inp = input('input sentence >>>')
        #     if _inp == 'q':
        #         break
        #     elif _inp == '':
        #         continue
        #     else:
        #         predictions = classifier.predict([_inp])
        #         print(predictions)

    else:
        classifier.train()

