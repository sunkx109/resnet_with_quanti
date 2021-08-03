"""
@file scale_quanti.py
@date 2021-07-23
@author Kaixin Sun (kaixin.sun@horizon.ai)
@reference Yushu Gao (yushu.gao@horizon.ai)
"""
import math
import torch
from .quanti_base import QuantiBase, QuantiMode 
from .FakeQuantize import FakeQuantize

#量化方法(可更改)
'''
def scale_quanti(input, scale, qmin, qmax):
    return torch.ops.horizon.scale_quanti(input, scale, qmin, qmax) 
'''
class ScaleQuanti(QuantiBase):
    
    def __init__(self,qmode,signed=True):
        """
        参数说明：
        * qmode  -- 表示量化的方式("int4","int8","uint4")
        * signed -- 表示是否采用采用对称量化的方式
        """
        super(ScaleQuanti,self).__init__()
        numerical_limits = {"int4":(-7,7),"int8":(-127,127),"uint4":(0,15)}
        #根据量化方式确定量化的上下限
        self.qmin,self.qmax = numerical_limits.get(qmode)
        self.rmin = None
        self.rmax = None
        self.signed = signed

    def _update_running_scale(self,data):

        with torch.no_grad():
            if self.rmax is None or self.rmax < torch.max(data):
                self.rmax = torch.max(data)
            if self.rmin is None or self.rmin > torch.min(data):
                self.rmin = torch.min(data)

            if self.signed:
                self.rmax = torch.max(torch.abs(self.rmax),torch.abs(self.rmin))
                self.rmin = -self.rmax
            
            self.scale = 1.0 * (self.rmax - self.rmin) / (self.qmax-self.qmin)
    
    def forward(self,data):
        self._update_running_scale(data)
        out = FakeQuantize.apply(data,self.scale,self.qmin,self.qmax)

        return out


    def __repr__(self):
        return "ScaleQuanti"











