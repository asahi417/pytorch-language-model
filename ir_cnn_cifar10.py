""" pytorch sample of image recognition on CIFAIR10

* logging instance loss
* checkpoint manager
* save checkpoints
* tensorboard
"""

# for model
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# for parameter manager
import json
import os
import random
import string
import toml
from glob import glob

# for logger
import logging
from logging.config import dictConfig


def create_log():
    """ simple Logger
    Usage
    -------------------
    logger.info(message)
    logger.error(error)
    """
    logging_config = dict(
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
        },
    )
    dictConfig(logging_config)
    logger = logging.getLogger()
    return logger


class ParameterManager:
    """ Parameter manager for model training """

    def __init__(self,
                 checkpoint: str = None,
                 checkpoint_dir: str = None,
                 default_parameter: str = None,
                 **kwargs):

        """ Parameter manager for model training
        - loading model: {checkpoint_dir}/{checkpoint}
        - new model: {checkpoint_dir}/new_unique_checkpoint_id

         Parameter
        -------------------
        checkpoint: existing checkpoint name if you want to load
        checkpoint_dir: checkpoint dir
        default_parameter: path to toml file containing default parameters
        kwargs: model parameters
        """
        self.__logger = create_log()
        if checkpoint_dir is None:
            checkpoint_dir = os.path.dirname(os.path.abspath(__file__))
        self.__logger.debug('checkpoint_dir: %s' % checkpoint_dir)
        if default_parameter is None:
            self.__logger.debug('no default parameter')
            parameter = kwargs
        else:
            self.__logger.debug('fetch default parameter from: %s' % default_parameter)
            parameter = dict()
            default_parameter = self.get_default_parameter(default_parameter)

            for k, v in default_parameter.items():
                if k in kwargs.keys():
                    parameter[k] = kwargs[k]
                else:
                    parameter[k] = v

        self.checkpoint_dir, self.parameter = self.__versioning(checkpoint_dir, parameter, checkpoint)
        self.__logger.debug('checkpoint: %s' % self.checkpoint_dir)

    def __call__(self, key):
        """ retrieve a parameter """
        if key not in self.parameter.keys():
            raise ValueError('unknown parameter %s' % key)
        return self.parameter[key]

    @staticmethod
    def get_default_parameter(default_parameter_toml_file: str):
        """ Get default parameter from toml file """
        assert default_parameter_toml_file.endswith('.toml')
        if not os.path.exists(default_parameter_toml_file):
            raise ValueError('no toml file: %s' % default_parameter_toml_file)
        parameter = toml.load(open(default_parameter_toml_file, "r"))
        parameter = dict([(k, v if v != '' else None) for k, v in parameter.items()])  # '' -> None
        return parameter

    @staticmethod
    def random_string(string_length=10, exceptions: list = None):
        """ Generate a random string of fixed length """
        if exceptions is None:
            exceptions = []
        while True:
            letters = string.ascii_lowercase
            random_letters = ''.join(random.choice(letters) for i in range(string_length))
            if random_letters not in exceptions:
                break
        return random_letters

    def __versioning(self, checkpoint_dir: str, parameter: dict = None, checkpoint: str = None):
        """ Checkpoint versioner: Either of `config` or `checkpoint` need to be specified (`config` has priority)

         Parameter
        ---------------------
        checkpoint_dir: directory where checkpoints will be saved, eg) `checkpoints/bert`
        parameter: parameter configuration to find same setting checkpoint
        checkpoint: existing checkpoint to be loaded

         Return
        --------------------
        path_to_checkpoint: path to new checkpoint dir
        parameter: parameter
        """
        if checkpoint is None and parameter is None:
            raise ValueError('either of `checkpoint` or `parameter` is needed.')

        if checkpoint is None:
            self.__logger.debug('issue new checkpoint id')
            # check if there are any checkpoints with same hyperparameters
            version_name = []
            for parameter_path in glob(os.path.join(checkpoint_dir, '*/hyperparameters.json')):
                _dir = parameter_path.replace('/hyperparameters.json', '')
                _dict = json.load(open(parameter_path))
                version_name.append(_dir.split('/')[-1])
                if parameter == _dict:
                    inp = input('found a checkpoint with same configuration\n'
                                'enter to delete the existing checkpoint %s\n'
                                'or exit by type anything but not empty' % _dir)
                    if inp == '':
                        os.remove(_dir)
                    else:
                        exit()

            new_checkpoint = self.random_string(exceptions=version_name)
            new_checkpoint_path = os.path.join(checkpoint_dir, new_checkpoint)
            os.makedirs(new_checkpoint_path, exist_ok=True)
            with open(os.path.join(new_checkpoint_path, 'hyperparameters.json'), 'w') as _f:
                json.dump(parameter, _f)
            return new_checkpoint_path, parameter

        else:
            self.__logger.debug('load existing checkpoint')
            checkpoints = glob(os.path.join(checkpoint_dir, checkpoint, 'hyperparameters.json'))
            if len(checkpoints) >= 2:
                raise ValueError('Checkpoints are duplicated: %s' % str(checkpoints))
            elif len(checkpoints) == 0:
                raise ValueError('No checkpoint: %s' % os.path.join(checkpoint_dir, checkpoint))
            else:
                parameter = json.load(open(checkpoints[0]))
                target_checkpoints_path = checkpoints[0].replace('/hyperparameters.json', '')
                return target_checkpoints_path, parameter


class Net(nn.Module):
    """ Network Architecture """

    def __init__(self):
        """ Network Architecture """
        super(Net, self).__init__()
        # components
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """ model output
         Parameter
        -------------
        x: tensor (batch, height, width, ch)

         Return
        -------------
        x: logit (batch, dim)
        y: prediction (batch, )
        p: probability (batch, dim)
        """
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        _, y = torch.max(x, dim=-1)
        p = torch.nn.functional.softmax(x, dim=1)
        return x, y, p


class ConvNet:
    """ CNN based image classifier """

    def __init__(self,
                 lr: float = 0.001,
                 momentum: float = 0.9,
                 progress_interval: int = 20000,
                 checkpoint_dir: str = None):
        """ CNN based image classifier
        * Allocate a GPU automatically; specify by CUDA_GPU_DEVICES
        * Load checkpoints if it exists
        """
        self.__logger = create_log()
        self.__logger.debug('initialize network: *** ConvNet for Image Classification ***')
        # setup parameter
        self.__param = ParameterManager(
            checkpoint_dir=checkpoint_dir,
            default_parameter='./parameters/ir_cnn_cifar10.toml',
            lr=lr,
            momentum=momentum)
        self.__checkpoint_model = os.path.join(self.__param.checkpoint_dir, 'model.pt')
        # build network
        self.__net = Net()
        if torch.cuda.device_count() >= 1:
            self.__logger.debug('running on GPU')
            self.if_use_gpu = True
            self.__net = self.__net.cuda()
        else:
            self.if_use_gpu = False
        # optimizer
        self.__optimizer = optim.SGD(self.__net.parameters(),
                                     lr=self.__param('lr'),
                                     momentum=self.__param('momentum'))
        # loss definition (CrossEntropyLoss includes softmax inside)
        self.__loss = nn.CrossEntropyLoss()
        # load pre-trained ckpt
        if os.path.exists(self.__checkpoint_model):
            self.__net.load_state_dict(torch.load(self.__checkpoint_model))
            self.__logger.debug('load ckpt from %s' % self.__checkpoint_model)
        # log
        self.__progress_interval = progress_interval
        self.__writer = SummaryWriter(log_dir=self.__param.checkpoint_dir)
        self.__sanity_check()

    @property
    def hyperparameters(self):
        return self.__param

    def __sanity_check(self):
        """ sanity check as logging model size """
        model_size = 0
        for k, v in self.__net.__dict__['_modules'].items():
            if hasattr(v, 'weight'):
                model_size += np.prod(v.weight.shape)
                self.__logger.debug(' - [weight size] %s: %s' % (k, str(list(v.weight.shape))))
        self.__logger.debug('model has %i parameters' % model_size)

    def train(self,
              data_train,
              data_valid,
              epoch: int):
        """ train model """
        self.__logger.debug('data loader instance')
        loader_train = torch.utils.data.DataLoader(
            data_train, batch_size=self.__param('batch_size'), shuffle=True, num_workers=2)
        loader_valid = torch.utils.data.DataLoader(
            data_valid, batch_size=self.__param('batch_size'), shuffle=False, num_workers=2)

        try:
            for epoch in range(epoch):  # loop over the epoch
                self.__epoch_train(loader_train, epoch_n=epoch)
                self.__epoch_valid(loader_valid, epoch_n=epoch)
        except KeyboardInterrupt:
            self.__logger.info('*** KeyboardInterrupt ***')
        self.__writer.close()
        torch.save(self.__net.state_dict(), self.__checkpoint_model)
        self.__logger.debug('complete training')

    def __epoch_train(self,
                      data_loader,
                      epoch_n: int):
        """ single epoch process for training """
        mean_loss = 0.0
        correct_count = 0.0
        data_size = 0.0
        inst_loss = 0.0

        for i, data in enumerate(data_loader, 1):
            # get the inputs (data is a list of [inputs, labels])
            inputs, labels = data
            if self.if_use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            # zero the parameter gradients
            self.__optimizer.zero_grad()
            # forward: output prediction and get loss
            logit, pred, prob = self.__net(inputs)

            tmp_loss = self.__loss(logit, labels)
            # backward: calculate gradient
            tmp_loss.backward()
            # optimize
            self.__optimizer.step()
            # accuracy
            correct_count += ((pred == labels).cpu().float().sum()).item()
            data_size += len(labels)

            # log
            loss = tmp_loss.cpu().item()
            self.__writer.add_scalar('train/loss', loss, i + epoch_n * len(data_loader))
            self.__writer.add_scalar('train/accuracy', correct_count / data_size, i + epoch_n * len(data_loader))
            inst_loss += loss
            mean_loss += loss

            if i % self.__progress_interval == 0:
                self.__logger.debug(' * epoch %i (%i/%i) instant loss: %.3f' %
                                    (epoch_n, i, len(data_loader), inst_loss / self.__progress_interval))
                inst_loss = 0.0

        self.__logger.debug('[epoch %i] (train) loss: %.3f, acc: %.3f' %
                            (epoch_n, mean_loss / len(data_loader), correct_count / data_size))

    def __epoch_valid(self,
                      data_loader,
                      epoch_n: int):
        """ single epoch process for validation """
        mean_loss = 0.0
        correct_count = 0.0
        data_size = 0.0
        for data in data_loader:
            inputs, labels = data
            if self.if_use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            logit, pred, _ = self.__net(inputs)
            correct_count += ((pred == labels).cpu().float().sum()).item()
            mean_loss += self.__loss(logit, labels).cpu().item()
            data_size += len(labels)
        self.__writer.add_scalar('valid/loss', mean_loss / len(data_loader), epoch_n)
        self.__writer.add_scalar('valid/accuracy', correct_count / data_size, epoch_n)
        self.__logger.debug('[epoch %i] (valid) loss: %.3f, acc: %.3f'
                            % (epoch_n, mean_loss / len(data_loader), correct_count / data_size))


if __name__ == '__main__':
    # CIFIAR 10 data loader
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # main
    nn_model = ConvNet(checkpoint_dir='./ckpt/ir_cnn_cifar10')
    nn_model.train(data_train=trainset, data_valid=testset, epoch=100)



