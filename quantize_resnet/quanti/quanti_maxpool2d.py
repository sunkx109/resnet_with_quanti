"""
@file quanti_maxpool2d.py
@date 2021-07-23
"""
import torch
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn.functional import max_pool2d
from .quanti_base import QuantiBase, QuantiMode
from torch.nn.modules.utils import _pair
import numpy as np


class QuantiMaxPool2d(QuantiBase):
    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
    ):

        super(QuantiMaxPool2d, self).__init__()
        kernel_size = _pair(kernel_size)
        self._kwargs = {
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
        }
        self._qoutput_kwargs = {"qmin": -127, "qmax": 127}

    def forward(self,input):
        out = max_pool2d(input, **self._kwargs)
        return out

    def __repr__(self):
        return "QuantiMaxPool2d"
