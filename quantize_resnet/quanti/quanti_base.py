"""
@file quanti_base.py
@date 2021-07-23
@author Kaixin Sun (kaixin.sun@horizon.ai)
@reference Yushu Gao (yushu.gao@horizon.ai)
"""

from torch import nn

class QuantiMode:
    kFloatTraining = "FloatTraining"
    kTrainingWithBN = "TrainingWithBN"
    kTrainingWithoutBN = "TrainingWithoutBN"
    kIntInference = "IntInference"


class QuantiBase(nn.Module):
    def __init__(self):
        super(QuantiBase, self).__init__()
        #self.mode = mode

    def set_input_scale(self, scale):
        raise NotImplementedError

    def forward(self, input):
        raise NotImplementedError