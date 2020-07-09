import transformers
import torch
import re
from itertools import chain
from torch import nn


PAD_TOKEN_LABEL_ID = nn.CrossEntropyLoss().ignore_index


class Transforms:
    """ Text encoder with transformers tokenizer """

    def __init__(self, transformer_tokenizer: str='xlm-roberta-base', max_seq_length: int = None, pad_to_max_length: bool = True):
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
        """ fix label for token label match """
        if label is None:
            return None
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
        encode = self.transform_function(' '.join(self.data[idx]))
        if self.label is not None:
            encode['labels'] = self.label[idx]
        float_list = ['attention_mask']
        encode = {k: torch.tensor(v, dtype=torch.float32) if k in float_list else torch.tensor(v, dtype=torch.long)
                  for k, v in encode.items()}
        return encode


if __name__ == '__main__':
    # tr = Transforms()
    max_seq_length = 50
    tokenizer = transformers.AutoTokenizer.from_pretrained('xlm-roberta-base', cache_dir='./cache')
    print('special token:', tokenizer.all_special_tokens, tokenizer.all_special_ids)

    sentence = "ITは今週金曜日の19:00から重要なセキュリティ更新をネットワークスイッチにインストールします。アップデート中は、wifiがダウンします。"
    token_list = [
        ['IT', 'は', '今週金曜日の', '19:00', 'から', '重要な', 'セキュリティ更新', 'を', 'ネットワークスイッチ', 'に',
         'インストール', 'します', '。', 'アップデート', '中', 'は', '、', 'wifi', 'が', 'ダウン', 'します', '。'],
        # ['I', 'T', 'は', '今', '週', '金', '曜', '日', 'の', '1', '9', ':', '0', '0', 'か', 'ら', '重', '要', 'な', 'セ', 'キ',
        #  'ュ', 'リ', 'テ', 'ィ', '更', '新', 'を', 'ネ', 'ッ', 'ト', 'ワ', 'ー', 'ク', 'ス', 'イ', 'ッ', 'チ', 'に', 'イ', 'ン',
        #  'ス', 'ト', 'ー', 'ル', 'し', 'ま', 'す', '。', 'ア', 'ッ', 'プ', 'デ', 'ー', 'ト', '中', 'は', '、', 'w', 'i', 'f', 'i', 'が', 'ダ', 'ウ', 'ン', 'し', 'ま', 'す', '。']
    ]
    print('from string (encode_plus)')
    print(' * sentence:', sentence)
    print(' * tokenize:', tokenizer.tokenize(sentence))
    print(' * encoded :')
    __encode = tokenizer.encode_plus(sentence, max_length=max_seq_length, pad_to_max_length=True, truncation=True)
    for p in zip(__encode['input_ids'], __encode['attention_mask']):
        print(p)

    # find tokenizer-depend prefix

    sentence_go_around = ''.join(tokenizer.tokenize('get tokenizer specific prefix'))
    prefix = sentence_go_around[:list(re.finditer('get', sentence_go_around))[0].span()[0]]

    def new_pipeline(tokens, labels):

        assert len(tokens) == len(labels)

        # get special tokens at start/end of sentence based on first token
        encode_first = tokenizer.encode_plus(tokens[0])
        sp_token_start, token_ids, sp_token_end = [], [], []
        for i in encode_first['input_ids']:
            if i in tokenizer.all_special_ids:
                if len(token_ids) == 0:
                    sp_token_start += [i]
                else:
                    sp_token_end += [i]
            else:
                token_ids += [i]

        encode = {k: v[:-len(sp_token_end)] for k, v in encode_first.items()}
        encode['labels'] = [PAD_TOKEN_LABEL_ID] * len(sp_token_start)
        encode['labels'] += [labels[0]] + [PAD_TOKEN_LABEL_ID] * (len(token_ids) - 1)

        # add inter-mid token info
        for t, l in zip(tokens[1:], labels[1:]):
            # input_ids without prefix/special tokens
            tmp_tokens = list(filter(lambda x: len(x) > 0, [t.replace(prefix, '') for t in tokenizer.tokenize(t)]))
            encode['input_ids'] += [tokenizer.convert_tokens_to_ids(t) for t in tmp_tokens]
            # other attribution without prefix/special tokens
            tmp_encode = tokenizer.encode_plus(t)
            input_ids_with_prefix = tmp_encode.pop('input_ids')[len(sp_token_start):-len(sp_token_end)]
            prefix_length = len(input_ids_with_prefix) - len(tmp_tokens)
            for k, v in tmp_encode.items():
                encode[k] += v[len(sp_token_start) + prefix_length:-len(sp_token_end)]
            # add fixed label
            encode['labels'] += [l] + [PAD_TOKEN_LABEL_ID] * (len(tmp_tokens) - 1)

        # add special token at the end and padding/truncate accordingly
        encode['labels'] = encode['labels'][:min(len(encode['labels']), max_seq_length - len(sp_token_end))]
        encode['labels'] += [PAD_TOKEN_LABEL_ID] * (max_seq_length - len(encode['labels']))

        encode['input_ids'] = encode['input_ids'][:min(len(encode['input_ids']), max_seq_length - len(sp_token_end))]
        encode['input_ids'] += sp_token_end
        encode['input_ids'] += [tokenizer.pad_token_id] * (max_seq_length - len(encode['input_ids']))

        attributions = list(filter(lambda x: x not in ['input_ids', 'labels'], encode.keys()))
        for k in attributions:
            encode[k] = encode[k][:min(len(encode[k]), max_seq_length - len(sp_token_end))]
            encode[k] += [0] * (max_seq_length - len(encode[k]))
        return encode


    print('\nfrom token')
    for t_list in token_list:
        print(' * sentence:', t_list)
        print(' * encoded :')
        __encode = new_pipeline(t_list, list(range(len(t_list))))
        for p in zip(__encode['input_ids'], __encode['attention_mask'], __encode['labels']):
            print(p)
