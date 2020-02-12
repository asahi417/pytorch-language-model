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
| LSTM  | SentencePieace | 71.27               |  64.96             | [toml file](./parameters/PennTreebank/SentencePieceBPETokenizer/lstm.toml)  | 

- enwiki8

| model | tokenizer      |  bpc (valid) | bpc (test)  | parameter |
| ----- |:---------------|:------------:|:-----------:|:---------:|
| LSTM  | Whitespace     | 1.72         | 1.71        | [toml file](./parameters/enwiki8/WhitespaceTokenizer/lstm.toml)  | 

- WikiText103

| model | tokenizer      |  perplexity (valid) | perplexity (test)  | parameter |
| ----- |:---------------|:-------------------:|:------------------:|:---------:|
| LSTM  | SentencePieace | 70.62               | 69.41              | [toml file](./parameters/WikiText103/SentencePieceBPETokenizer/lstm.toml)  | 

## Forthcoming...
- [ ] mixture precision (fp16 training) 
- [ ] Async data batcher
- [ ] Remove batch in validation
- [x] Regard `<eos>` as a special token in SentencePiece tokenizer
  
