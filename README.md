# Pytorch Language Modeling
Pytorch language modeling.

## Contents
All the script work with python>=3.6. 

```
git clone https://github.com/asahi417/pytorch-tutorial
cd pytorch-tutorial
pip install -r requirement.txt
```

## Short result  

- PennTreebank

| model | tokenizer      |  perplexity (valid) | perplexity (test)  | parameter |
| ----- |:---------------|:-------------------:|:------------------:|:---------:|
| LSTM  | SentencePieace | 60.84               | 55.66              | [toml file](./parameters/PennTreebank/SentencePieceBPETokenizer/lstm.toml) | 

- WikiText103

| model | tokenizer      |  perplexity (valid) | perplexity (test)  | parameter |
| ----- |:---------------|:-------------------:|:------------------:|:---------:|
| LSTM  | SentencePieace | 43.31               | 43.15              | [toml file](./parameters/WikiText103/SentencePieceBPETokenizer/lstm.toml) | 

- enwiki8

| model | tokenizer      |  bpc (valid) | bpc (test)  | parameter |
| ----- |:---------------|:------------:|:-----------:|:---------:|
| LSTM  | Whitespace     | 1.71         | 1.702       | [toml file](./parameters/enwiki8/WhitespaceTokenizer/lstm.toml) | 


## Forthcoming...
- [x] Remove batch in validation
- [x] Regard `<eos>` as a special token in SentencePiece tokenizer
- [x] Train from other checkpoint
- [ ] Configuration for Transformer XL (enwiki/Penn/Wikitext103)
- [ ] [Adaptive Attention Span](https://arxiv.org/pdf/1905.07799.pdf)
- [ ] mixture precision (fp16 training) 
- [ ] Async data batcher
- [ ] Adaptive Softmax of transformer XL in WikiText103 (SentencePiece can reduce the vocab so may not be needed anymore)

## Checkpoint
- [ ] WikiText further train `wkahhxbtix3`
- [ ] enwiki8 further train `exruxoigkx`
  
