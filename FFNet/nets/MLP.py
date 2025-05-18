#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/7 17:55
# @Author  : Sun Chunli
# @Software: PyCharm
import  torch
from torch import nn
from config import config
num_classes = config.NUM_CLASSES
from nets import top_k
class MLP(nn.Module):
    def __init__(self, input_size, common_size,sing_train = False):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(inplace=True),
            nn.Linear(input_size, input_size),
            nn.ReLU(inplace=True),
            nn.Linear(input_size, common_size),
        )
        self.classifier = nn.Linear(common_size,num_classes)
        self.lm = sing_train
        self.top_k = top_k.Top_k(256)

    def forward(self, x):
        out = self.linear(x)        
        if self.lm:
            out = self.classifier(out)
            assert out.size(1) == num_classes
        return out

class Mul_layers_MLP(nn.Module):
    def __init__(self, input_size, common_size,sing_train = False):
        super(Mul_layers_MLP, self).__init__()


        self.linear = nn.Sequential(
            MLP(input_size,512),
            MLP(512,1024),
            MLP(1024,2048),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048, common_size),
            nn.BatchNorm1d(common_size),
            nn.Linear(common_size,num_classes),
        )
        
        self.classifier_top_k = nn.Sequential(
            nn.Linear(256, common_size),
            nn.BatchNorm1d(common_size),
            nn.Linear(common_size,num_classes),
        )
        self.lm = sing_train
        self.top_k = top_k.Top_k(256)
       
        
    def forward(self, x):
        #print(x.size())
        out = self.linear(x)
        out = self.top_k(out)
        
        if self.lm:
            out = self.classifier_top_k(out)
            assert out.size(1) == num_classes
        return out