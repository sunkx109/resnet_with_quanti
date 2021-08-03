"""
@file quanti_input.py
@date 2021-07-23
@author Kaixin Sun (kaixin.sun@horizon.ai)
@reference Yushu Gao (yushu.gao@horizon.ai)
"""
import torch
from .scale_quanti import ScaleQuanti
from .quanti_base import QuantiBase, QuantiMode


class QuantiInput(QuantiBase):
	def __init__(self):
		super(QuantiInput,self).__init__()

		self.quanti_op = ScaleQuanti("int8")

	def forward(self,data):
		return self.quanti_op(data)
