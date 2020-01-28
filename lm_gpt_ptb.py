""" pytorch GPT implementation, train on PTB """

# for model
import copy
import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import random
from torch.utils.tensorboard import SummaryWriter

from transformer_module import BaseGPT2
from util import create_log, ParameterManager
from huggingface_optimizer import AdamW, get_linear_schedule_with_warmup, get_constant_schedule


class BatchFeeder:
    """ Pytorch batch feeding iterator for language model training """

    def __init__(self,
                 batch_size,
                 num_steps,
                 sequence):
        """ Pytorch batch feeding iterator for language model training

         Parameter
        -------------------
        batch_size: int
            batch size
        num_steps: int
            sequence truncation size
        sequence: list
            integer token id sequence to feed
        """
        self._index = 0
        self.batch_size = batch_size
        self.num_steps = num_steps
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        seq = torch.tensor(sequence, dtype=torch.long, device=device)
        self.data_size = seq.size(0)

        n_batch = self.data_size // self.batch_size
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        seq = seq.narrow(0, 0, n_batch * self.batch_size)
        # Evenly divide the data across the bsz batches.
        self._data = seq.view(self.batch_size, -1).contiguous()

    def __len__(self):
        return (self.data_size // self.batch_size - 1) // self.num_steps

    def __iter__(self):
        return self

    def __next__(self):
        """ next batch for train data (size is `self._batch_size`) loop for self._iteration_number

         Return
        -----------------
        (inputs, outputs): list (batch_size, num_steps)
        """
        if (self._index + 1) * self.num_steps + 1 > self._data.size(1):
            self._index = 0
            raise StopIteration
        x = self._data[:, self._index * self.num_steps:(self._index + 1) * self.num_steps].contiguous()
        y = self._data[:, self._index * self.num_steps + 1:(self._index + 1) * self.num_steps + 1].contiguous()
        self._index += 1
        return x, y


class GPT2:
    """ GPT2 language model """

    def __init__(self,
                 checkpoint: str = None,
                 checkpoint_dir: str = None,
                 default_parameter: str = None,
                 **kwargs):
        """ GPT2 language model """
        self.__logger = create_log()
        self.__logger.debug('initialize network: *** GPT2 ***')
        # setup parameter
        self.__param = ParameterManager(
            checkpoint=checkpoint,
            checkpoint_dir=checkpoint_dir,
            default_parameter=default_parameter,
            **kwargs)
        self.__checkpoint_model = os.path.join(self.__param.checkpoint_dir, 'model.pt')
        # build network
        self.__net = BaseGPT2(
            n_layer=self.__param("n_layer"),
            n_embedding=self.__param("n_embedding"),
            n_state_ffn=self.__param("n_state_ffn"),
            n_head=self.__param("n_head"),
            n_context=self.__param("n_context"),
            max_cache_size=self.__param("max_cache_size"),
            residual_dropout=self.__param("residual_dropout"),
            attention_dropout=self.__param("attention_dropout"),
            embedding_dropout=self.__param("embedding_dropout"),
            vocab_size=self.__param("vocab_size")
        )
        # GPU allocation
        if torch.cuda.device_count() == 1:
            self.__logger.debug('running on single GPU')
            self.__net = self.__net.cuda()
            self.n_gpu = 1
        elif torch.cuda.device_count() > 1:
            self.__logger.debug('running on %i GPUs' % torch.cuda.device_count())
            self.__net = torch.nn.DataParallel(self.__net.cuda())
            self.n_gpu = torch.cuda.device_count()
        else:
            self.__logger.debug('no GPUs found')
            self.n_gpu = 0

        # optimizer
        self.__optimizer = AdamW(
            self.__net.parameters(),
            lr=self.__param('lr'),
            weight_decay=self.__param('weight_decay'))
        if self.__param('scheduler') == 'constant':
            self.__scheduler = get_constant_schedule(self.__optimizer)
        elif self.__param('scheduler') == 'linear':
            self.__scheduler = get_linear_schedule_with_warmup(
                self.__optimizer,
                num_warmup_steps=self.__param('warmup_steps'),
                num_training_steps=self.__param('total_steps'))
        else:
            raise ValueError('bad scheduler: %s' % self.__param('scheduler'))

        # loss definition (CrossEntropyLoss includes softmax inside)
        self.__loss = nn.CrossEntropyLoss()

        # load pre-trained ckpt
        if os.path.exists(self.__checkpoint_model):
            ckpt = torch.load(self.__checkpoint_model)
            self.__net.load_state_dict(ckpt['model_state_dict'])
            self.__optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.__scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            self.__training_step = ckpt['training_step']  # num of training step
            self.__epoch = ckpt['epoch']
            self.__best_epoch = ckpt['best_epoch']
            self.__best_val_ppl = ckpt['best_val_ppl']
            self.__logger.debug('load ckpt from %s' % self.__checkpoint_model)
            self.__logger.debug(' - epoch (best): %s (%s) ' % (str(ckpt['epoch']), str(ckpt['best_epoch'])))
            self.__logger.debug(' - ppl (best)  : %s (%s) ' % (str(ckpt['val_ppl']), str(ckpt['best_val_ppl'])))
        else:
            self.__training_step = 0
            self.__epoch = 0
            self.__best_epoch = 0
            self.__best_val_ppl = None

        # log
        self.__writer = SummaryWriter(log_dir=self.__param.checkpoint_dir)
        self.__sanity_check(self.__param('random_seed'))

    @property
    def hyperparameters(self):
        return self.__param

    def __sanity_check(self, seed: int = 1234):
        """ sanity check as logging model size """
        self.__logger.debug('trainable variables')
        model_size = 0
        for name, param in self.__net.named_parameters():
            if param.requires_grad:
                __shape = list(param.data.shape)
                model_size += np.prod(__shape)
                self.__logger.debug(' - [weight size] %s: %s' % (name, str(__shape)))
        self.__logger.debug(' - %i variables in total' % model_size)
        self.__logger.debug('hyperparameters')
        for k, v in self.__param.parameter.items():
            self.__logger.debug(' - [param] %s: %s' % (k, str(v)))

        # fix random seed
        if seed:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if self.n_gpu > 0:
                torch.cuda.manual_seed_all(seed)

    def evaluate(self, data_valid, data_test=None):
        """ evaluate model """
        batch_param = dict(batch_size=self.__param('batch_size'), num_steps=self.__param('n_context'))
        loss, ppl = self.__epoch_valid(BatchFeeder(sequence=data_valid, **batch_param))
        self.__logger.debug('(val)  loss: %.5f, ppl: %.5f' % (loss, ppl))
        if data_test:
            loss, ppl = self.__epoch_valid(BatchFeeder(sequence=data_test, **batch_param), is_test=True)
            self.__logger.debug('(test) loss: %.5f, ppl: %.5f' % (loss, ppl))

    def train(self,
              data_train: list,
              data_valid: list,
              data_test: list = None,
              progress_interval: int = 100):
        """ train model """
        best_model_wts = None
        val_ppl = None

        self.__logger.debug('initialize batch feeder')
        batch_param = dict(batch_size=self.__param('batch_size'), num_steps=self.__param('n_context'))
        loader_train = BatchFeeder(sequence=data_train, **batch_param)
        loader_valid = BatchFeeder(sequence=data_valid, **batch_param)

        try:
            while True:
                loss, ppl = self.__epoch_train(loader_train, progress_interval=progress_interval)
                val_loss, val_ppl = self.__epoch_valid(loader_valid)
                self.__logger.debug('[epoch %i] (train) loss: %.3f, ppl: %.3f (valid) loss: %.3f, ppl: %.3f'
                                    % (self.__epoch, loss, ppl, val_loss, val_ppl))

                if self.__best_val_ppl is None or val_ppl < self.__best_val_ppl:
                    best_model_wts = copy.deepcopy(self.__net.state_dict())
                    self.__best_epoch = self.__epoch
                    self.__best_val_ppl = val_ppl

                if self.__training_step > self.__param('total_steps'):
                    if data_test:
                        loader_test = BatchFeeder(sequence=data_test, **batch_param)
                        loss, ppl = self.__epoch_valid(loader_test, is_test=True)
                        self.__logger.debug('(test) loss: %.3f, ppl: %.3f' % (loss, ppl))
                    break

        except KeyboardInterrupt:
            self.__logger.info('*** KeyboardInterrupt ***')

        self.__logger.debug('[training completed] best model: valid ppt %0.3f at epoch %i'
                            % (self.__best_val_ppl, self.__best_epoch))
        torch.save({
            'model_state_dict': self.__net.state_dict(),
            'optimizer_state_dict': self.__optimizer.state_dict(),
            'scheduler_state_dict': self.__scheduler.state_dict(),
            'training_step': self.__training_step,
            'epoch': self.__epoch,
            'val_ppl': val_ppl,
            'best_epoch': self.__best_epoch,
            'best_val_ppl': self.__best_val_ppl,
            'best_model_state_dict': best_model_wts,
        }, self.__checkpoint_model)
        self.__writer.close()
        self.__logger.info('ckpt saved at %s' % self.__checkpoint_model)

    def __epoch_train(self, data_loader, progress_interval: int = 1000):
        """ training on single epoch """
        self.__net.train()
        perplexity = None
        mean_loss = None
        full_seq_length = 0
        full_loss = 0

        for data in data_loader:

            # get the inputs (data is a list of [inputs, labels])
            inputs, outputs = data
            # zero the parameter gradients
            self.__optimizer.zero_grad()
            # forward: output prediction and get loss
            (logit, prob, pred), _ = self.__net(inputs)
            # backward: calculate gradient
            logit = logit.view(-1, logit.size(-1))
            outputs = outputs.view(-1)
            tmp_loss = self.__loss(logit, outputs)
            tmp_loss.backward()
            # optimize
            self.__optimizer.step()
            self.__scheduler.step()
            # log
            full_loss += len(outputs) * tmp_loss.cpu().item()
            full_seq_length += len(outputs)
            perplexity = np.exp(min(30, full_loss / full_seq_length))
            mean_loss = full_loss / full_seq_length
            lr = self.__optimizer.param_groups[0]['lr']
            self.__writer.add_scalar('train/loss', mean_loss, self.__training_step)
            self.__writer.add_scalar('train/perplexity', perplexity, self.__training_step)
            self.__writer.add_scalar('learning_rate', lr, self.__training_step)

            if self.__training_step % progress_interval == 0:
                self.__logger.debug(' * (step %i) ppl: %.3f, lr: %0.6f' % (self.__training_step, perplexity, lr))

            self.__training_step += 1
        self.__epoch += 1
        return mean_loss, perplexity

    def __epoch_valid(self, data_loader, is_test: bool=False):
        """ validation/test """
        self.__net.eval()
        full_seq_length = 0
        full_loss = 0
        for data in data_loader:
            inputs, outputs = data
            (logit, prob, pred), _ = self.__net(inputs)
            print(inputs[0, -3:], outputs[0, -3:], pred[0, -3:])
            logit = logit.view(-1, logit.size(-1))
            outputs = outputs.view(-1)
            full_loss += len(outputs) * self.__loss(logit, outputs).cpu().item()
            full_seq_length += len(outputs)
        mean_loss = full_loss / full_seq_length
        perplexity = np.exp(min(30, full_loss / full_seq_length))
        if not is_test:
            self.__writer.add_scalar('valid/perplexity', perplexity, self.__epoch)
            self.__writer.add_scalar('valid/loss', mean_loss, self.__epoch)
        return mean_loss, perplexity


def get_options():
    parser = argparse.ArgumentParser(description='Train tokenizer', formatter_class=argparse.RawTextHelpFormatter)
    _p = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('-c', '--ckpt', help='pre-trained model ckpt', default=None, type=str, **_p)
    parser.add_argument('-e', '--evaluate', help='evaluation', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    with open('./data/penn-treebank/ptb.train.eos.id.txt', 'r') as f:
        _data_train = [int(i) for i in f.read().split()]
    with open('./data/penn-treebank/ptb.valid.eos.id.txt', 'r') as f:
        _data_valid = [int(i) for i in f.read().split()]
    with open('./data/penn-treebank/ptb.test.eos.id.txt', 'r') as f:
        _data_test = [int(i) for i in f.read().split()]

    arguments = get_options()

    _model = GPT2(checkpoint=arguments.ckpt,
                  checkpoint_dir='./ckpt/lm_gpt_ptb',
                  default_parameter='./parameters/lm_gpt_ptb.toml')
    if arguments.evaluate:
        _model.evaluate(data_valid=_data_valid, data_test=_data_test)
    else:
        _model.train(data_train=_data_train,
                     data_valid=_data_valid,
                     data_test=_data_test,
                     progress_interval=20)

