""" WIP: mixed precision training

Operations that can use FP16 storage
- matrix multiplications (linear, matmul, bmm, conv)
- most pointwise operations (relu, tanh, add, sub, mul)
Operations that need FP32 mantissa
- reduction operations (batch norm, layer norm, sum, softmax)

Operations that need FP32 range
- pointwise operations (exp, log, pow)
- loss functions (cross entropy, l2 loss, weight decay)

https://nvlabs.github.io/iccv2019-mixed-precision-tutorial/files/dusan_stosic_intro_to_mixed_precision_training.pdf
"""
import torch
import torch.nn as nn


class ToFloat16(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.half()


def revert_float32(network_module):
    if isinstance(network_module, torch.nn.LayerNorm):
        network_module = network_module.float()
    for child in network_module.children():
        revert_float32(child)
    return network_module


def network_to_half(network):
    return nn.Sequential(ToFloat16(), revert_float32(network.half()))

