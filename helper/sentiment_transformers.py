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
    'bert-base-multilingual-cased': transformers.BertForSequenceClassification
}
VALID_TOKENIZER = {
    'xlm-roberta-large': transformers.XLMRobertaTokenizer,
    'xlm-roberta-base': transformers.XLMRobertaTokenizer,
    'bert-base-multilingual-cased': transformers.BertTokenizer
}


def get_dataset(data_name: str = 'sst', label_to_id: dict = None):
    """ download dataset file and return dictionary including training/validation split """
    label_to_id = dict() if label_to_id is None else label_to_id

    def decode_data(iterator, file_prefix, _label_to_id):
        if not os.path.exists(file_prefix + '.text') or not os.path.exists(file_prefix + '.label'):
            list_text = []
            list_label = []
            for i in iterator:
                if data_name == 'sst' and i.label == 'neutral':
                    continue
                # if i.label not in _label_to_id.keys():
                #     _label_to_id[i.label] = len(_label_to_id)
                list_text.append(' '.join(i.text))
                list_label.append(i.label)

            with open(file_prefix + '.text', 'w') as f_writer:
                f_writer.write('\n'.join(list_text))

            with open(file_prefix + '.label', 'w') as f_writer:
                f_writer.write('\n'.join(list_label))

        list_of_text = open(file_prefix + '.text', 'r').read().split('\n')
        list_of_label_raw = open(file_prefix + '.label', 'r').read().split('\n')
        for unique_label in list(set(list_of_label_raw)):
            if unique_label not in _label_to_id.keys():
                _label_to_id[unique_label] = len(_label_to_id)
        list_of_label = [int(_label_to_id[l]) for l in list_of_label_raw]
        assert len(list_of_label) == len(list_of_text)
        return _label_to_id, (list_of_text, list_of_label)

    data_field, label_field = torchtext.data.Field(sequential=True), torchtext.data.Field(sequential=False)
    if data_name == 'imdb':
        iterator_split = torchtext.datasets.IMDB.splits(data_field, root=CACHE_DIR, label_field=label_field)
    elif data_name == 'sst':
        iterator_split = torchtext.datasets.SST.splits(data_field, root=CACHE_DIR, label_field=label_field)
    else:
        raise ValueError('unknown dataset: %s' % data_name)

    data_split, data = list(), None

    for name, it in zip(['train', 'valid', 'test'], iterator_split):
        _file_prefix = os.path.join(CACHE_DIR, data_name, name)
        label_to_id, data = decode_data(it, file_prefix=_file_prefix, _label_to_id=label_to_id)
        data_split.append(data)
        LOGGER.info('dataset %s/%s: %i' % (data_name, name, len(data[0])))
    return data_split, label_to_id


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


class AdamW(torch.optim.Optimizer):
    """ Implements Adam algorithm with weight decay fix.

    Parameters:
        lr (float): learning rate. Default 1e-3.
        betas (tuple of 2 floats): Adams beta parameters (b1, b2). Default: (0.9, 0.999)
        eps (float): Adams epsilon. Default: 1e-6
        weight_decay (float): Weight decay. Default: 0.0
        correct_bias (bool): can be set to False to avoid correcting bias in Adam (e.g. like in Bert TF repository). Default True.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0, correct_bias=True):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(1.0 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1.0 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.data.add_(-group["lr"] * group["weight_decay"], p.data)

        return loss


class TokenEncoder:
    """ Token encoder with transformers tokenizer """

    def __init__(self,
                 transformer: str,
                 max_seq_length: int = None):
        self.tokenizer = VALID_TOKENIZER[transformer].from_pretrained(transformer, cache_dir=CACHE_DIR)
        if max_seq_length and max_seq_length > self.tokenizer.max_len:
            raise ValueError('`max_seq_length should be less than %i' % self.tokenizer.max_len)
        self.max_seq_length = max_seq_length if max_seq_length else self.tokenizer.max_len
        LOGGER.info('max_sequence_length (LM max_sequence_length): %i (%i)'
                    % (self.max_seq_length, self.tokenizer.max_len))

    def __call__(self, text):
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
        self.label = label
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
    """ finetune transformers on text classification """

    def __init__(self,
                 dataset: str,
                 batch_size_validation: int = None,
                 checkpoint: str = None,
                 inference_mode: bool = False,
                 **kwargs):
        """ finetune transformers on text classification """
        self.inference_mode = inference_mode
        if self.inference_mode:
            LOGGER.info('*** initialize network (INFERENCE MODE) ***')
        else:
            LOGGER.info('*** initialize network ***')

        # checkpoint versioning
        self.param = ParameterManager(prefix=dataset, checkpoint=checkpoint, dataset=dataset, **kwargs)
        self.batch_size_validation = batch_size_validation if batch_size_validation else self.param('batch_size')

        # fix random seed
        random.seed(self.param('random_seed'))
        np.random.seed(self.param('random_seed'))
        torch.manual_seed(self.param('random_seed'))

        # model/dataset setup
        stats, label_to_id = self.load_ckpt()
        if self.inference_mode:
            if stats is None or label_to_id is None:
                raise ValueError('As no checkpoints found, unable to perform inference.')
            self.dataset_split, self.label_to_id = None, label_to_id
            self.token_encoder = TokenEncoder(self.param('transformer'))
            self.writer = None
        else:
            self.dataset_split, self.label_to_id = get_dataset(self.param('dataset'), label_to_id=label_to_id)
            self.token_encoder = TokenEncoder(self.param('transformer'), max_seq_length=self.param('max_seq_length'))
            self.writer = SummaryWriter(log_dir=self.param.checkpoint_dir)

        self.id_to_label = dict([(str(v), str(k)) for k, v in self.label_to_id.items()])
        self.model_seq_cls = VALID_TRANSFORMER_SEQUENCE_CLASSIFICATION[self.param('transformer')].from_pretrained(
            self.param('transformer'),
            cache_dir=CACHE_DIR,
            num_labels=len(list(self.id_to_label.keys())),
            output_hidden_states=True
        )

        # load checkpoint
        if stats is not None:
            self.__step = stats['step']  # num of training step
            self.__epoch = stats['epoch']  # num of epoch
            self.__best_val_accuracy = stats['best_val_accuracy']
            self.__best_val_accuracy_step = stats['best_val_accuracy_step']
            if self.inference_mode:
                self.__best_model_wts = None
                self.model_seq_cls.load_state_dict(stats['best_model_state_dict'])
                LOGGER.info('use best ckpt from step %i / %i' % (self.__best_val_accuracy_step, self.__step))
            else:
                self.__best_model_wts = stats['best_model_state_dict']
                self.model_seq_cls.load_state_dict(stats['model_state_dict'])
            del stats
        else:
            self.__step = 0
            self.__epoch = 0
            self.__best_val_accuracy = None
            self.__best_val_accuracy_step = None
            self.__best_model_wts = None

        # optimizer
        if self.inference_mode:
            self.optimizer = self.scheduler = None
        else:
            if self.param("optimizer") == 'adamw':
                self.optimizer = AdamW(
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
                self.scheduler = get_linear_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=self.param('warmup_step'),
                    num_training_steps=self.param('total_step'))
            else:
                raise ValueError('bad scheduler: %s' % self.param('scheduler'))

            # apply checkpoint statistics to optimizer/scheduler
            if stats is not None:
                self.optimizer.load_state_dict(stats['optimizer_state_dict'])
                self.scheduler.load_state_dict(stats['scheduler_state_dict'])

        # GPU allocation
        self.n_gpu = torch.cuda.device_count()
        self.data_parallel = False
        if self.n_gpu == 1:
            self.model_seq_cls = self.model_seq_cls.cuda()
        elif self.n_gpu > 1:  # TODO: test multi-GPUs
            self.data_parallel = True
            self.model_seq_cls = torch.nn.DataParallel(self.model_seq_cls.cuda())
            LOGGER.info('WARNING: torch.nn.DataParallel is not tested')
        else:
            self.n_gpu = 0
        LOGGER.info('running on %i GPUs' % self.n_gpu)
        self.device = 'cuda' if self.n_gpu > 0 else 'cpu'

    def load_ckpt(self):
        checkpoint_file = os.path.join(self.param.checkpoint_dir, 'model.pt')
        label_id_file = os.path.join(self.param.checkpoint_dir, 'label_to_id.json')
        if os.path.exists(checkpoint_file):
            assert os.path.exists(label_id_file)
            LOGGER.info('load ckpt from %s' % checkpoint_file)
            ckpt = torch.load(checkpoint_file, map_location='cpu')
            ckpt_dict = dict(
                step=ckpt['step'],
                epoch=ckpt['epoch'],
                model_state_dict=ckpt['model_state_dict'],
                best_val_accuracy=ckpt['best_val_accuracy'],
                best_val_accuracy_step=ckpt['best_val_accuracy_step'],
                best_model_wts=ckpt['best_model_state_dict'],
                best_model_state_dict=ckpt['best_model_state_dict'],
                optimizer_state_dict=ckpt['optimizer_state_dict'],
                scheduler_state_dict=ckpt['scheduler_state_dict'],
            )
            label_to_id = json.load(open(label_id_file, 'r'))
            return ckpt_dict, label_to_id
        else:
            return None, None

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
        data_loader_train = torch.utils.data.DataLoader(
            Dataset(self.dataset_split[0][0], label=self.dataset_split[0][1], token_encoder=self.token_encoder),
            batch_size=self.param('batch_size'),
            shuffle=True,
            num_workers=NUM_WORKER,
            drop_last=True)
        data_loader_valid = torch.utils.data.DataLoader(
            Dataset(self.dataset_split[1][0], label=self.dataset_split[1][1], token_encoder=self.token_encoder),
            batch_size=self.batch_size_validation,
            num_workers=NUM_WORKER)
        if len(self.dataset_split) > 2:
            data_loader_test = torch.utils.data.DataLoader(
                Dataset(self.dataset_split[2][0], label=self.dataset_split[2][1], token_encoder=self.token_encoder),
                batch_size=self.batch_size_validation,
                num_workers=NUM_WORKER
            )
        else:
            data_loader_test = None

        LOGGER.info('start training from step %i (epoch: %i)' % (self.__step, self.__epoch))
        try:
            with detect_anomaly():
                while True:
                    if_training_finish = self.__epoch_train(data_loader_train)
                    if_early_stop = self.__epoch_valid(data_loader_valid, prefix='valid')

                    if if_training_finish or if_early_stop:
                        if data_loader_test:
                            self.__epoch_valid(data_loader_test, prefix='test')
                        break
                    self.__epoch += 1

        except RuntimeError:
            LOGGER.info(traceback.format_exc())
            LOGGER.info('*** RuntimeError (NaN found, see above log in detail) ***')

        except KeyboardInterrupt:
            LOGGER.info('*** KeyboardInterrupt ***')

        if self.__best_val_accuracy is None:
            self.param.remove_ckpt()
            exit('nothing to be saved')

        LOGGER.info('[training completed] best model: valid loss %0.3f at step %i'
                    % (self.__best_val_accuracy, self.__best_val_accuracy_step))
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
            'best_val_accuracy': self.__best_val_accuracy,
            'best_val_accuracy_step': self.__best_val_accuracy_step,
            'best_model_state_dict': self.__best_model_wts
        }, os.path.join(self.param.checkpoint_dir, 'model.pt'))
        with open(os.path.join(self.param.checkpoint_dir, 'label_to_id.json'), 'w') as f:
            json.dump(self.label_to_id, f)
        self.writer.close()
        LOGGER.info('ckpt saved at %s' % self.param.checkpoint_dir)

    def __epoch_train(self, data_loader):
        """ train on single epoch return flag which is True if training has been completed """
        self.model_seq_cls.train()

        for i, (inputs, attn_mask, outputs) in enumerate(data_loader, 1):

            inputs = inputs.to(self.device)
            attn_mask = attn_mask.to(self.device)
            outputs = outputs.to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()
            # forward: output prediction and get loss
            model_outputs = self.model_seq_cls(inputs, attention_mask=attn_mask, labels=outputs)
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
            inst_accuracy = ((pred == outputs).cpu().float().mean()).item()
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

        if self.__best_val_accuracy is None or accuracy > self.__best_val_accuracy:
            self.__best_val_accuracy = accuracy
            self.__best_val_accuracy_step = self.__step
            if self.data_parallel:
                self.__best_model_wts = copy.deepcopy(self.model_seq_cls.module.state_dict())
            else:
                self.__best_model_wts = copy.deepcopy(self.model_seq_cls.state_dict())
        elif self.param('tolerance') is not None:
            if self.param('tolerance') < self.__best_val_accuracy - accuracy:
                LOGGER.info('early stop:\n - best accuracy: %0.3f \n - current accuracy: %0.3f'
                            % (self.__best_val_accuracy, accuracy))
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
                        default=128,
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
    parser.add_argument('--tolerance', help='early stop tolerance in terms of valid accuracy', default=None, type=float)
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

