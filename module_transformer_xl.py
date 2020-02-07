""" pytorch TransformerXL implementation

https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/train.py#L436
https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py#L664
"""

import math
import torch
import torch.nn as nn


class TransformerXL(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, x, cached_key_value: list=None):
