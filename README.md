# Pytorch scripts
Pytorch self-contained sample scripts:
- logging instance loss
- checkpoint manager (write/load checkpoint)
- tensorboard visualization

## Contents
All the script work with python>=3.6. 

```
git clone https://github.com/asahi417/pytorch-tutorial
cd pytorch-tutorial
pip install -r requirement.txt
```

### [LSTM Language Model](./lm_lstm_ptb.py)  
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

| Learning Rate | perplexity (train) | perplexity (valid) |
| ------------- |:------------------:|:------------------:|
| 0.005         |                    |                    |
| 0.005         |                    |                    |
| 0.005         |                    |                    |
| 0.005         |                    |                    |



### Others
- [image recognition](./ir_cnn_cifar10.py)
- [image recognition with pre-trained checkpoint](./ir_resnet_hymenoptera.py)


