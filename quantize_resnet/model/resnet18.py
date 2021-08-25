"""
@file resnet18.py
@date 2021-07-23
"""

import sys 
import torch
sys.path.append("..") 
from quanti import *
bn_kwargs = {}


class BasicBlock(torch.nn.Module):
    expansion=1
    def __init__(self,in_channels, out_channels, Basicmode=QuantiMode.kTrainingWithoutBN,stride=1):
        super(BasicBlock,self).__init__()
        self.conv_cell1 =  QuantiFusedConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            bn_kwargs=bn_kwargs,
            activation="relu",
            mode = Basicmode,
        )
        self.use_shortcut = (
            "add"
            if (stride != 1 or in_channels != self.expansion * out_channels)
            else None
        )
        if self.use_shortcut is not None:
            self.shortcut = QuantiFusedConv2d(
                in_channels,
                self.expansion * out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                bn_kwargs=bn_kwargs,
                mode = Basicmode,
            )
        self.conv_cell2 = QuantiFusedConv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            activation="relu",
            mode = Basicmode ,
        )

    def forward(self,x):
        out = self.conv_cell1(x)
        if self.use_shortcut is not None:
            edata = self.shortcut(x)
            out = self.conv_cell2(out)+edata
        else:
            out = self.conv_cell2(out)
        return out

class ResNet18(torch.nn.Module):
    def __init__(self,num_class=10,Basicmode=QuantiMode.kTrainingWithoutBN):
        super(ResNet18,self).__init__()
        num_block = [2, 2, 2, 2]
        self.in_channels = 64

        self.qinput = QuantiInput()
        self.conv1 = QuantiFusedConv2d(
            3,
            64,
            3,
            padding=1,
            bias=False,
            bn_kwargs=bn_kwargs,
            activation="relu",
            mode = Basicmode,
        )
        self.layer1 = self._make_layer(64, num_block[0], stride=1,mode=Basicmode)
        self.layer2 = self._make_layer(128, num_block[1], stride=2,mode=Basicmode)
        self.layer3 = self._make_layer(256, num_block[2], stride=2,mode=Basicmode)
        self.layer4 = self._make_layer(512, num_block[3], stride=2,mode=Basicmode)
        self.max_pool = QuantiMaxPool2d(4, stride=1, padding=0)
        self.linear = QuantiFusedConv2d(512*BasicBlock.expansion,10,kernel_size=1,disable_qoutput=False,mode=Basicmode)

    def _make_layer(self, out_channels, num_blocks, stride,mode):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, mode,stride))
            self.in_channels = out_channels * BasicBlock.expansion
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.qinput(x)
        out = self.conv1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.max_pool(out)
        out = self.linear(out)
        out = out.view(-1, 10)
        return out



    	





