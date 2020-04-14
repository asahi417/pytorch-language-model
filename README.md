# Pytorch Language Model Experiments
Some experimental stuffs related to language models:
- Language model in pytorch and benchmarks (not just use pre-trained checkpoints, but implement and train from scratch over language modeling task).
- Finetuning pre-trained language models over some downstream tasks

## Contents
All the script work with python>=3.6. 

```
git clone https://github.com/asahi417/pytorch-tutorial
cd pytorch-tutorial
pip install -r requirement.txt
```

## Language Modeling  

- PennTreebank

| model | tokenizer      |  perplexity (valid) | perplexity (test)  | parameter |
| ----- |:---------------|:-------------------:|:------------------:|:---------:|
| LSTM  | SentencePieace | 60.84               | 55.66              | [toml file](./parameters/PennTreebank/SentencePieceBPETokenizer/lstm.toml) | 

- WikiText103

| model | tokenizer      |  perplexity (valid) | perplexity (test)  | parameter |
| ----- |:---------------|:-------------------:|:------------------:|:---------:|
| LSTM  | SentencePieace | 39.40               | 39.53              | [toml file](./parameters/WikiText103/SentencePieceBPETokenizer/lstm.toml) | 

- enwiki8

| model | tokenizer      |  bpc (valid) | bpc (test)  | parameter |
| ----- |:---------------|:------------:|:-----------:|:---------:|
| LSTM  | Whitespace     | 1.71         | 1.702       | [toml file](./parameters/enwiki8/WhitespaceTokenizer/lstm.toml) | 


## Forthcoming...
- [x] Remove batch in validation
- [x] Regard `<eos>` as a special token in SentencePiece tokenizer
- [x] Train from other checkpoint
- [x] Avoid explosion by clamp exp 
- [ ] mixture precision (fp16 training) 
- [ ] [Adaptive Attention Span](https://arxiv.org/pdf/1905.07799.pdf) as it efficiently runs on single GPU.
- [ ] Async data batcher (well, the RAM is always full so it might not be useful anymore...)

  
