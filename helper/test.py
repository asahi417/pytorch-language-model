""" self-contained NER finetuning on hugginface.transformers (conll_2003/wnut_17) """
import os
import logging
import transformers
import torch
from torch import nn
from logging.config import dictConfig
from itertools import chain

dictConfig({
    "version": 1,
    "formatters": {'f': {'format': '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'}},
    "handlers": {'h': {'class': 'logging.StreamHandler', 'formatter': 'f', 'level': logging.DEBUG}},
    "root": {'handlers': ['h'], 'level': logging.DEBUG}})
LOGGER = logging.getLogger()
NUM_WORKER = int(os.getenv("NUM_WORKER", '4'))
PROGRESS_INTERVAL = int(os.getenv("PROGRESS_INTERVAL", '100'))
CACHE_DIR = os.getenv("CACHE_DIR", './cache')
CKPT_DIR = os.getenv("CKPT_DIR", './ckpt')
PAD_TOKEN_LABEL_ID = nn.CrossEntropyLoss().ignore_index


def get_dataset(data_name: str = 'wnut_17', label_to_id: dict = None, allow_update: bool=True):
    """ download dataset file and return dictionary including training/validation split """
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
            os.system('mv ./xlm-roberta-ner/data/coNLL-2003/* %s/' % data_path)
            os.system('rm -rf ./xlm-roberta-ner')
        files = ['train.txt', 'valid.txt', 'test.txt']
    elif data_name == 'wnut_17':
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
            os.system("curl -L 'https://github.com/leondz/emerging_entities_17/raw/master/wnut17train.conll'  | tr '\t' ' ' > %s/train.txt.tmp" % data_path)
            os.system("curl -L 'https://github.com/leondz/emerging_entities_17/raw/master/emerging.dev.conll' | tr '\t' ' ' > %s/dev.txt.tmp" % data_path)
            os.system("curl -L 'https://raw.githubusercontent.com/leondz/emerging_entities_17/master/emerging.test.annotated' | tr '\t' ' ' > %s/test.txt.tmp" % data_path)
        files = ['train.txt.tmp', 'dev.txt.tmp', 'test.txt.tmp']
    else:
        raise ValueError('unknown dataset: %s' % data_name)

    data_split = dict()
    for name, filepath in zip(['train', 'valid', 'test'], files):
        label_to_id, data_dict = decode_file(filepath, _label_to_id=label_to_id)
        data_split[name] = data_dict
        LOGGER.info('dataset %s/%s: %i entries' % (data_name, filepath, len(data_dict['data'])))
    if allow_update:
        return data_split, label_to_id
    else:
        return data_split


class Transforms:
    """ Text encoder with transformers tokenizer """

    def __init__(self, transformer_tokenizer: str, max_seq_length: int = None, pad_to_max_length: bool = True):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(transformer_tokenizer, cache_dir=CACHE_DIR)
        if max_seq_length and max_seq_length > self.tokenizer.max_len:
            raise ValueError('`max_seq_length should be less than %i' % self.tokenizer.max_len)
        self.max_seq_length = max_seq_length if max_seq_length else self.tokenizer.max_len
        self.pad_to_max_length = pad_to_max_length

    def __call__(self, text: str):
        return self.tokenizer.encode_plus(
            text, max_length=self.max_seq_length, pad_to_max_length=self.pad_to_max_length)

    @property
    def all_special_ids(self):
        return self.tokenizer.all_special_ids

    def tokenize(self, *args, **kwargs):
        return self.tokenizer.tokenize(*args, **kwargs)


class Dataset(torch.utils.data.Dataset):
    """ torch.utils.data.Dataset with transformer tokenizer """

    def __init__(self, data: list, transform_function, label: list = None):
        self.data = data  # list of half-space split tokens
        self.transform_function = transform_function
        self.label = self.fix_label(label)

    def fix_label(self, label):
        assert len(label) == len(self.data)
        fixed_labels = []
        for y, x in zip(label, self.data):
            assert len(y) == len(x)
            encode = self.transform_function(' '.join(x))
            fixed_label = list(chain(*[
                [label] + [PAD_TOKEN_LABEL_ID] * (len(self.transform_function.tokenize(word)) - 1)
                for label, word in zip(y, x)]))
            if encode['input_ids'][0] in self.transform_function.all_special_ids:
                fixed_label = [PAD_TOKEN_LABEL_ID] + fixed_label
            fixed_label += [PAD_TOKEN_LABEL_ID] * (len(encode['input_ids']) - len(fixed_label))
            fixed_label = fixed_label[:self.transform_function.max_seq_length]
            fixed_labels.append(fixed_label)
        return fixed_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # encode = self.transform_function(' '.join(self.data[idx]))
        # if self.label is not None:
            # assert len(encode) == len(self.label[idx])
            # encode['labels'] = self.label[idx]
        # float_list = ['attention_mask']
        # encode = {k: torch.tensor(v, dtype=torch.float32) if k in float_list else torch.tensor(v, dtype=torch.long)
        #           for k, v in encode.items()}
        # print(encode.keys())
        # print(encode['input_ids'].shape, encode['labels'].shape, encode['attention_mask'].shape)
        return torch.tensor(self.label[idx], dtype=torch.float32)


if __name__ == '__main__':
    _data, _dict = get_dataset('conll_2003')
    _trans = Transforms('xlm-roberta-base', 128)
    _iter = torch.utils.data.DataLoader(
        Dataset(**_data['train'], transform_function=_trans),
        num_workers=4,
        batch_size=100,
        shuffle=True,
        drop_last=True)

    for n, i in enumerate(_iter):
        print(n)


