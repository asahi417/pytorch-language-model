# Pytorch scripts
Pytorch self-contained sample scripts:

## Contents
All the script work with python>=3.6. 

```
git clone https://github.com/asahi417/pytorch-tutorial
cd pytorch-tutorial
pip install -r requirement.txt
```

### [GPT2 Language Model on SentencePieceTokenizer](./lm_gpt_ptb.py)
Build PTB corpus by sentence piece  

```
python corpus_tokenizer.py 
```

Train LSTM language model on PTB

```
python lm_gpt_ptb.py
```

Hyperparameter can be changed by editing [toml file](./parameters/lm_lstm_ptb.toml).

***Mixed precision training***

WIP


### [LSTM Language Model on SentencePieceTokenizer](./lm_lstm_ptb.py)  

Build PTB corpus by sentence piece  

```
python corpus_tokenizer.py 
```

Train LSTM language model on PTB

```
python lm_lstm_ptb.py
```

Hyperparameter can be changed by editing [toml file](./parameters/lm_lstm_ptb.toml),
and here is a brief result with different learning rate settings.

| Learning Rate | perplexity (valid) | epoch |
| ------------- |:------------------:|:-----:|
| 0.001         |            88.49   | 142   |
| 0.0005        |            96.18   | 229   |
| 0.0001        |            90.89   | 131   |

Test perplexity of the best model (learning rate as 0.001) is 86.73

### Others
- [image recognition](helper/ir_cnn_cifar10.py)
- [image recognition with pre-trained checkpoint](helper/ir_resnet_hymenoptera.py)


