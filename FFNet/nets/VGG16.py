#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/2/27 11:23
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


class Vgg16(nn.Module):
    def __init__(self):	   #num_classes，此处为 二分类值为2
        super(Vgg16, self).__init__()
        
        weights_res = os.path.join(weights_pre,'vgg16-397923af.pth')
        #net = models.vgg16(pretrained=True)   #从预训练模型加载VGG16网络参数
        
        if os.path.exists(weights_res):
            net = models.vgg16(pretrained=False)
            net.load_state_dict(torch.load(weights_res))
        else:
            net = models.vgg16(pretrained=True)   #从预训练模型加载VGG16网络参数
        
        net.classifier = nn.Sequential()	#将分类层置空，下面将改变我们的分类层
        self.features = net		#保留VGG16的特征层

        #只训练后面的几个卷积层
        ct=0
        for name,child in self.features.named_children():
             ct+=1
             if ct<6:
                #print(params)
                for name2,params in child.named_parameters():
                   params.requires_grad = False
        # 但是参数全部固定了，也没法进行学习，所以我们不固定最后一层，即全连接层fc
        #for params in xy_model.conv5.paramters():
        #    params.requires_grad = True


        self.classifier = nn.Sequential(    #定义自己的分类层
                nn.Linear(512 * 7 * 7, 200),  #512 * 7 * 7不能改变 ，由VGG16网络决定的，第二个参数为神经元个数可以微调
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(200, 200),
                nn.ReLU(True),
                nn.Dropout(),
        )
        
        self.fusion1 = nn.Sequential(    #定义自己的分类层
                nn.Linear(400, 200),  #512 * 7 * 7不能改变 ，由VGG16网络决定的，第二个参数为神经元个数可以微调
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(200, 200),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(200, num_classes),
        )

    def forward(self, input,input1,y):
        x = self.features(input)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        x1 = self.features(input1)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.classifier(x1)
        
        y =torch.cat((x,x1),1)
        #print(np.shape(y))
        ouput_ = self.fusion1(y)

        return  ouput_
        
if __name__=='__main__':
    res = Vgg16()
    for name,child in res.named_children():
        print(name)
