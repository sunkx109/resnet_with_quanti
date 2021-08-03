"""
@file quanti_relu.py
@date 2020-07-08
@author Yushu Gao (yushu.gao@horizon.ai)
"""

import torch
from torch import nn
from .quanti_base import QuantiBase


class QuantiReLU(QuantiBase):
    def __init__(self, max_value=6):
        super(QuantiReLU, self).__init__()
        self.max = max_value

    def forward(self, input):
        return torch.clamp(input, 0, self.max)
