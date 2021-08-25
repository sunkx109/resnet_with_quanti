"""
@file quanti_conv2d.py
@date 2021-07-23
"""
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn import init, Conv2d
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.modules.utils import _pair
from .quanti_relu import QuantiReLU
from .quanti_base import QuantiBase, QuantiMode
from .scale_quanti import ScaleQuanti
import numpy as np

class QuantiFusedConv2d(QuantiBase):
	def __init__(
		self,
		in_channels,
		out_channels,
		kernel_size,
		stride=1,
		padding=0,
		dilation=1,
		groups=1,
		bias=True,
		bn_kwargs=None,
		activation=None,
		disable_qoutput=False,
		mode = QuantiMode.kTrainingWithoutBN,

	):
		super(QuantiFusedConv2d,self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = _pair(kernel_size)
		self.stride = _pair(stride)
		self.padding = _pair(padding)
		self.dilation = _pair(dilation)
		self.groups = groups
		self.use_bias = bias

		self.bn_kwargs = bn_kwargs
		self.activation = activation
		self.disable_qoutput = disable_qoutput

		assert activation is None or activation == "relu"
		assert in_channels % groups == 0 and out_channels % groups == 0

		self.num_features = out_channels
		self.mode = mode
		self._set_use_bias()
		if self.mode in [
			QuantiMode.kFloatTraining,
			QuantiMode.kTrainingWithBN,
			QuantiMode.kTrainingWithoutBN,
		]:
			self._init_training()
		else:
			self._init_inference()
	def _set_use_bias(self):
		if (
			self.bn_kwargs is not None
			and self.mode == QuantiMode.kTrainingWithoutBN
		) or self.mode == QuantiMode.kIntInference:
			self.use_bias = True

	def _init_training(self):
		# conv
		self._conv_kwargs = {
            "stride": self.stride,
            "padding": self.padding,
            "dilation": self.dilation,
            "groups": self.groups,
        }
		# weight
		self.weight = Parameter(
			torch.empty(
			self.out_channels,
            self.in_channels // self.groups,
            self.kernel_size[0],
            self.kernel_size[1],
			)
		)
		if self.use_bias:
			self.bias = Parameter(torch.zeros(self.out_channels))
		else:
			self.bias = None


        #quanti function
		self.qweight_op = self._get_qweight()
		self.bn_op = self._get_bn()

		self.activation_op = self._get_activation()
		self.qoutput_op = None if self.disable_qoutput else self._get_qoutput()
		self.reset_parameters()

    

	def _get_qweight(self): 
		#return 一个int8的量化对象
		return ScaleQuanti("int8")

	def _get_bn(self):
		if self.bn_kwargs is not None:
			if self.mode in [
                QuantiMode.kFloatTraining,
                QuantiMode.kTrainingWithBN,
           	]:
				return nn.BatchNorm2d(self.num_features, **self.bn_kwargs)
			return None

	def _get_qoutput(self):
		return ScaleQuanti("int8")

	def _get_activation(self):
		if self.activation=="relu":
			return QuantiReLU()
		else:
			return None

	def reset_parameters(self):
		init.kaiming_uniform_(self.weight, a=math.sqrt(5))
		if self.bias is not None:
			fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
			bound = 1 / math.sqrt(fan_in)
			init.uniform_(self.bias, -bound, bound)
    
	def _train_forward(self,input):
		#get quanti weight
		qweight = self.qweight_op(self.weight)
		qweight = qweight.cuda()
		
		out = F.conv2d(input, qweight, None, **self._conv_kwargs)
		if self.bn_op is not None:
			out = self.bn_op(out)
		if self.activation_op is not None:
			out = self.activation_op(out)
		if self.qoutput_op is not None:
			out = self.qoutput_op(out)

		return out

	def _init_inference(self):
		if not hasattr(self.qweight_op,"scale") or not hasattr(self.qoutput_op,"scale"):
			raise ValueError('qweight_op.scale or qoutput_op.scale is not existed, should be provided.')
		self.M = self.qweight_op.scale / self.qoutput_op.scale

	def  _init_inference_forward(self,input):
		with torch.no_grad():
			#前向推理过程，因为采用的是对称量化的方式，所以没有Z
			qweight = self.qweight_op(self.weight)
			out =  F.conv2d(input, qweight, self.bias, **self._conv_kwargs)
			#公式(3)
			out = self.M * out
			out.round_()
			out.clamp_(qweight.qmin,qweight.qmax)
			return out

	def forward(self,input):
		if self.mode == QuantiMode.kIntInference:
			return self._init_inference_forward(input)
		else:
			return self._train_forward(input)

	def __repr__(self):
		return "QuantiFusedConv2d"













