""" Fetch dataset for NLP task """
import os
import logging
from logging.config import dictConfig

import torchtext

dictConfig({
    "version": 1,
    "formatters": {'f': {'format': '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'}},
    "handlers": {'h': {'class': 'logging.StreamHandler', 'formatter': 'f', 'level': logging.DEBUG}},
    "root": {'handlers': ['h'], 'level': logging.DEBUG}})
LOGGER = logging.getLogger()
CACHE_DIR = os.getenv("CACHE_DIR", './cache')


def get_dataset_ner(data_name: str = 'wnut_17', label_to_id: dict = None, allow_update: bool=True):
    """ download dataset file and return dictionary including training/validation split """

    def split(_data, label, _export_path, _label_to_id):
        assert len(_data) == len(label)
        os.makedirs(_export_path, exist_ok=True)
        id_to_label = {v: k for k, v in _label_to_id.items()}
        train_n = int(len(_data) * 0.7)
        valid_n = int(len(_data) * 0.2)

        with open(os.path.join(_export_path, 'train.txt'), 'w') as f:
            for x, y in zip(_data[:train_n], label[:train_n]):
                for _x, _y in zip(x, y):
                    f.write("{} {}\n".format(_x, id_to_label[_y]))
                f.write('\n')

        with open(os.path.join(_export_path, 'valid.txt'), 'w') as f:
            for x, y in zip(_data[train_n:train_n + valid_n], label[train_n:train_n + valid_n]):
                for _x, _y in zip(x, y):
                    f.write("{} {}\n".format(_x, id_to_label[_y]))
                f.write('\n')

        with open(os.path.join(_export_path, 'test.txt'), 'w') as f:
            for x, y in zip(_data[train_n + valid_n:], label[train_n + valid_n:]):
                for _x, _y in zip(x, y):
                    f.write("{} {}\n".format(_x, id_to_label[_y]))
                f.write('\n')

    label_to_id = dict() if label_to_id is None else label_to_id
    data_path = os.path.join(CACHE_DIR, data_name)

    def decode_file(file_name, _label_to_id: dict):
        inputs, labels = [], []
        with open(os.path.join(data_path, file_name), 'r') as f:
            sentence, entity = [], []
            for n, line in enumerate(f):
                line = line.strip()
                if len(line) == 0 or line.startswith("-DOCSTART-"):
                    if len(sentence) != 0:
                        assert len(sentence) == len(entity)
                        inputs.append(sentence)
                        labels.append(entity)
                        sentence, entity = [], []
                else:
                    ls = line.split()
                    if len(ls) < 2:
                        continue
                    sentence.append(ls[0])
                    # Examples could have no label for mode = "test"
                    tag = 'O' if len(ls) < 2 else ls[-1]
                    if tag not in _label_to_id.keys():
                        assert allow_update
                        _label_to_id[tag] = len(_label_to_id)
                    entity.append(_label_to_id[tag])
        return _label_to_id, {"data": inputs, "label": labels}

    if data_name == 'conll_2003':
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
            os.system('git clone https://github.com/mohammadKhalifa/xlm-roberta-ner')
            os.system('mv ./xlm-roberta-ner/data/coNLL-2003/* {}/'.format(data_path))
            os.system('rm -rf ./xlm-roberta-ner')
        files = ['train.txt', 'valid.txt', 'test.txt']
    elif data_name == 'wnut_17':
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
            os.system("curl -L 'https://github.com/leondz/emerging_entities_17/raw/master/wnut17train.conll'  | tr '\t' ' ' > {}/train.txt.tmp".format(data_path))
            os.system("curl -L 'https://github.com/leondz/emerging_entities_17/raw/master/emerging.dev.conll' | tr '\t' ' ' > {}/dev.txt.tmp".format(data_path))
            os.system("curl -L 'https://raw.githubusercontent.com/leondz/emerging_entities_17/master/emerging.test.annotated' | tr '\t' ' ' > {}/test.txt.tmp".format(data_path))
        files = ['train.txt.tmp', 'dev.txt.tmp', 'test.txt.tmp']
    elif data_name in ['wiki-ja-500', 'wiki-news-ja-1000']:
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
            os.system('git clone https://github.com/Hironsan/IOB2Corpus')
            if data_name == 'wiki-ja-500':
                os.system('mv ./IOB2Corpus/hironsan.txt {}/tmp.txt'.format(data_path))
            else:
                os.system('mv ./IOB2Corpus/ja.wikipedia.conll {}/tmp.txt'.format(data_path))
            os.system('rm -rf ./IOB2Corpus')
            label_to_id, data = decode_file('tmp.txt', dict())
            split(data['data'], data['label'], data_path, label_to_id)
            os.system('rm -rf {}/tmp.txt'.format(data_path))
        files = ['train.txt', 'valid.txt', 'test.txt']
    else:
        raise ValueError('unknown dataset: %s' % data_name)

    data_split = dict()
    for name, filepath in zip(['train', 'valid', 'test'], files):
        label_to_id, data_dict = decode_file(filepath, _label_to_id=label_to_id)
        data_split[name] = data_dict
        LOGGER.info('dataset {}/{}: {} entries'.format(data_name, filepath, len(data_dict['data'])))
    if allow_update:
        return data_split, label_to_id
    else:
        return data_split


def get_dataset_sentiment(data_name: str = 'sst', label_to_id: dict = None, allow_update: bool=True):
    """ download dataset file and return dictionary including training/validation split """
    label_to_id = dict() if label_to_id is None else label_to_id

    def decode_data(iterator, _label_to_id: dict):
        list_text = []
        list_label = []
        for i in iterator:
            if data_name == 'sst' and i.label == 'neutral':
                continue
            list_text.append(' '.join(i.text))
            list_label.append(i.label)

        for unique_label in list(set(list_label)):
            if unique_label not in _label_to_id.keys():
                assert allow_update
                _label_to_id[unique_label] = len(_label_to_id)
        list_label = [int(_label_to_id[l]) for l in list_label]
        assert len(list_label) == len(list_text)
        return _label_to_id, {"data": list_text, "label": list_label}

    data_field, label_field = torchtext.data.Field(sequential=True), torchtext.data.Field(sequential=False)
    if data_name == 'imdb':
        iterator_split = torchtext.datasets.IMDB.splits(data_field, root=CACHE_DIR, label_field=label_field)
    elif data_name == 'sst':
        iterator_split = torchtext.datasets.SST.splits(data_field, root=CACHE_DIR, label_field=label_field)
    else:
        raise ValueError('unknown dataset: %s' % data_name)

    data_split, data = dict(), None
    for name, it in zip(['train', 'valid', 'test'], iterator_split):
        label_to_id, data = decode_data(it, _label_to_id=label_to_id)
        data_split[name] = data
        LOGGER.info('dataset %s/%s: %i' % (data_name, name, len(data['data'])))
    if allow_update:
        return data_split, label_to_id
    else:
        return data_split
