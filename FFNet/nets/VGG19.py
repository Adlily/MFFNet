#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/28 10:09
# @Author  : Sun Chunli
# @Software: PyCharm
import os
import torch.nn as nn
from torchvision import models
import torch
from config import config
num_classes = config.NUM_CLASSES
width = config.WIDTH
weights_pre = config.WEIGHTS_MODEL
import numpy as np
class Vgg19(nn.Module):
    """
    pretrained VGG 19
    """

    def __init__(self,den_size = 200,pre_flag=True):
       
        self.pre_flag = pre_flag
        self.den_size = den_size

        super(Vgg19,self).__init__()

        weights_res = os.path.join(weights_pre,'vgg19_bn-c79401a0.pth')

        if os.path.exists(weights_res):
            self.xy_model = models.vgg19_bn(pretrained=False)
            self.xy_model.load_state_dict(torch.load(weights_res))
        else:
            self.xy_model = models.vgg19_bn(pretrained=True)
        #fc_features = self.xy_model.fc.in_features
        self.xy_model.classifier[6] = nn.Linear(4096,self.den_size)
        #self.features = nn.Sequential(*list(xy_model.childre())[: -1])  # after gap
        #print(list(self.features.children())[-1])

        #xy_model.load_weights(weight_res)
        #设置预训练的参数为不保存
        ct=0
        for name,child in self.xy_model.named_children():
             ct+=1
             if ct<5:
                #print(params)
                for name2,params in child.named_parameters():
                   params.requires_grad = False
        # 但是参数全部固定了，也没法进行学习，所以我们不固定最后一层，即全连接层fc
        #for params in xy_model.conv5.paramters():
        #    params.requires_grad = True
        for param in self.xy_model.classifier[6].parameters():
             param.requires_grad = True
        # vgg16 参数冻结, 这里更改为只冻结vgg16_bn的前几层！！！对后面的层进行微调
        #print()
        #fc_features = self.xy_model.fc.in_features
        #print(fc_features)


        self.linear = nn.Linear(self.den_size,200)
        self.fusion1 = nn.Sequential(
             nn.Linear(400,200),
             nn.ReLU(),
             nn.Dropout(0.4),
             nn.Linear(200,num_classes),
             #nn.Softmax()
         )

    def forward(self,input,input2,y):

        y1 = self.xy_model(input)
        y2 = self.xy_model(input2)

        y =torch.cat((y1,y2),1)
        #print(np.shape(y))
        ouput_ = self.fusion1(y)

        return  ouput_

