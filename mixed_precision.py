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


class tofp16(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.half()


def copy_in_params(net, params):
    net_params = list(net.parameters())
    for i in range(len(params)):
        net_params[i].data.copy_(params[i].data)


def set_grad(params, params_with_grad):

    for param, param_w_grad in zip(params, params_with_grad):
        if param.grad is None:
            param.grad = torch.nn.Parameter(param.data.new().resize_(*param.data.size()))
        param.grad.data.copy_(param_w_grad.grad.data)


def BN_convert_float(module):
    '''
    BatchNorm layers to have parameters in single precision.
    Find all layers and convert them back to float. This can't
    be done with built in .apply as that function will apply
    fn to all modules, parameters, and buffers. Thus we wouldn't
    be able to guard the float conversion based on the module type.
    '''
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        BN_convert_float(child)
    return module


def network_to_half(network):
    return nn.Sequential(tofp16(), BN_convert_float(network.half()))
