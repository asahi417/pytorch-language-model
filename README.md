# Pytorch Language Modeling
Pytorch language modeling.

## Contents
All the script work with python>=3.6. 

```
git clone https://github.com/asahi417/pytorch-tutorial
cd pytorch-tutorial
pip install -r requirement.txt
```

### LSTM Language Model on SentencePieceTokenizer  

Train LSTM language model on PTB  
```
python main.py 
```
Hyperparameter can be changed by editing [toml file](parameters/PennTreebank/SentencePieceBPETokenizer/lstm.toml),
and here is a brief result with different learning rate settings.

| Learning Rate | perplexity (valid) | epoch |
| ------------- |:------------------:|:-----:|
| 0.001         |            88.49   | 142   |
| 0.0005        |            96.18   | 229   |
| 0.0001        |            90.89   | 131   |

Test perplexity of the best model (learning rate as 0.001) is 86.73

