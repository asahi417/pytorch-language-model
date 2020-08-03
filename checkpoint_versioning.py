""" Checkpoint versioning based on hyperparameter file """
import os
import hashlib
import json
import shutil
import logging
from glob import glob
from logging.config import dictConfig

dictConfig({
    "version": 1,
    "formatters": {'f': {'format': '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'}},
    "handlers": {'h': {'class': 'logging.StreamHandler', 'formatter': 'f', 'level': logging.DEBUG}},
    "root": {'handlers': ['h'], 'level': logging.DEBUG}})
LOGGER = logging.getLogger()
CKPT_DIR = os.getenv("CKPT_DIR", './ckpt')

__all__ = 'Argument'


class Argument:
    """ Model training arguments manager """

    def __init__(self, prefix: str = None, checkpoint: str = None, **kwargs):
        """  Model training arguments manager

         Parameter
        -------------------
        prefix: prefix to filename
        checkpoint: existing checkpoint name if you want to load
        kwargs: model arguments
        """
        self.checkpoint_dir, self.parameter = self.__version(kwargs, checkpoint, prefix)
        LOGGER.info('checkpoint: %s' % self.checkpoint_dir)
        for k, v in self.parameter.items():
            LOGGER.info(' - [arg] %s: %s' % (k, str(v)))
        self.__dict__.update(self.parameter)

    def remove_ckpt(self):
        shutil.rmtree(self.checkpoint_dir)

    @staticmethod
    def md5(file_name):
        """ get MD5 checksum """
        hash_md5 = hashlib.md5()
        with open(file_name, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def __version(self, parameter: dict = None, checkpoint: str = None, prefix: str = None):
        """ Checkpoint version

         Parameter
        ---------------------
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
            LOGGER.info('issue new checkpoint id')
            # check if there are any checkpoints with same hyperparameters
            version_name = []
            for parameter_path in glob(os.path.join(CKPT_DIR, '*/parameter.json')):
                _dir = parameter_path.replace('/parameter.json', '')
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

            with open(os.path.join(CKPT_DIR, 'tmp.json'), 'w') as _f:
                json.dump(parameter, _f)
            new_checkpoint = self.md5(os.path.join(CKPT_DIR, 'tmp.json'))
            new_checkpoint = '_'.join([prefix, new_checkpoint]) if prefix else new_checkpoint
            new_checkpoint_dir = os.path.join(CKPT_DIR, new_checkpoint)
            os.makedirs(new_checkpoint_dir, exist_ok=True)
            shutil.move(os.path.join(CKPT_DIR, 'tmp.json'), os.path.join(new_checkpoint_dir, 'parameter.json'))
            return new_checkpoint_dir, parameter

        else:
            LOGGER.info('load existing checkpoint')
            checkpoints = glob(os.path.join(CKPT_DIR, checkpoint, 'parameter.json'))
            if len(checkpoints) >= 2:
                raise ValueError('Checkpoints are duplicated: %s' % str(checkpoints))
            elif len(checkpoints) == 0:
                raise ValueError('No checkpoint: %s' % os.path.join(CKPT_DIR, checkpoint))
            else:
                parameter = json.load(open(checkpoints[0]))
                target_checkpoints_path = checkpoints[0].replace('/parameter.json', '')
                return target_checkpoints_path, parameter
