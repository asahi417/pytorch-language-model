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
| LSTM  | SentencePieace |                |               | [toml file](./parameters/PennTreebank/SentencePieceBPETokenizer/lstm.toml)  | 

- WikiText103

| model | tokenizer      |  perplexity (valid) | perplexity (test)  | parameter |
| ----- |:---------------|:-------------------:|:------------------:|:---------:|
| LSTM  | SentencePieace | 46.94               | 46.52              | [toml file](./parameters/WikiText103/SentencePieceBPETokenizer/lstm.toml)  | 

- enwiki8

| model | tokenizer      |  bpc (valid) | bpc (test)  | parameter |
| ----- |:---------------|:------------:|:-----------:|:---------:|
| LSTM  | Whitespace     | 1.78         | 1.76        | [toml file](./parameters/enwiki8/WhitespaceTokenizer/lstm.toml)  | 


## Forthcoming...
- [x] Remove batch in validation
- [x] Regard `<eos>` as a special token in SentencePiece tokenizer
- [ ] Train from other checkpoint
- [ ] Configuration for Transformer XL (enwiki/Penn/Wikitext103)
- [ ] Adaptive Softmax of transformer XL in WikiText103
- [ ] [Adaptive Attention Span](https://arxiv.org/pdf/1905.07799.pdf)
- [ ] mixture precision (fp16 training) 
- [ ] Async data batcher
  
