import transformers
import torch
from torch import nn
from itertools import chain

model = 'xlm-roberta-base'
tokenizer = transformers.AutoTokenizer.from_pretrained(model, cache_dir='./cache')
model_cls = transformers.AutoModelForTokenClassification.from_pretrained(model, config='./cache')
test_sentence = "Use the FileLock if all instances of your application are running on the same host and a SoftFileLock."
label = list(range(len(test_sentence.split())))
# Use cross entropy ignore_index as padding label id so that only real label ids contribute to the loss later.
pad_token_label_id = nn.CrossEntropyLoss().ignore_index

print('test sentence:', len(test_sentence.split()), test_sentence)
print('test label   :', len(label), label)
print('special token:', tokenizer.all_special_tokens, tokenizer.all_special_ids)

# check if nn.CrossEntropyLoss().ignore_index is working
input_ids = torch.tensor(tokenizer.encode(test_sentence, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
labels = torch.tensor([pad_token_label_id] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
outputs = model_cls(input_ids, labels=labels)
loss = outputs[0]
print('loss with pad_token_label_id:', loss.item())
assert loss.item() == 0
input_ids = torch.tensor(tokenizer.encode(test_sentence, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
outputs = model_cls(input_ids, labels=labels)
loss, score = outputs[:2]
print('loss without pad_token_label_id:', loss.item())
assert loss.float() != 0
print(score)
print(torch.max(score, 2)[1].cpu().int().tolist())


# to fix tokenization effect on label sequence
assert len(label) == len(test_sentence.split())
encoding = tokenizer.encode_plus(test_sentence, max_length=tokenizer.max_len, pad_to_max_length=True)
# Use the real label id for the first token of the word, and padding ids for the remaining tokens
fixed_label = list(chain(*[[label] + [pad_token_label_id] * (len(tokenizer.tokenize(word))-1)
                           for label, word in zip(label, test_sentence.split())]))
print(fixed_label)
if encoding['input_ids'][0] in tokenizer.all_special_ids:
    fixed_label = [pad_token_label_id] + fixed_label
fixed_label += [pad_token_label_id] * (len(encoding['input_ids']) - len(fixed_label))

print('encoding by `encode_plus`:', len(encoding['input_ids']), encoding['input_ids'])
print('fixed label              :', len(fixed_label), fixed_label)

