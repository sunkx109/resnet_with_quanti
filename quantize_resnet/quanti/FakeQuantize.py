import torch
import math
import numpy as np
from torch.autograd import Variable
from torch.autograd import Function



def quantize(input, scale, qmin, qmax):
    q_input = input / scale
    q_input.clamp_(qmin,qmax).round_()
    return q_input

def dequantize(input,scale):
    return input * scale

class FakeQuantize(Function):    
    @staticmethod
    def forward(ctx,input_data,scale,qmin,qmax):
        out = quantize(input_data,scale,qmin,qmax)
        out = dequantize(out,scale)
        return out
    
    @staticmethod
    def backward(ctx,grad_output):
        return grad_output,None,None,None
