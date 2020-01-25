""" Utility """
import logging
from logging.config import dictConfig
import json
import os
import shutil
import random
import string
import toml
from glob import glob

__all__ = [
    "create_log",
    "ParameterManager"
]


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
                        shutil.rmtree(_dir)
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

