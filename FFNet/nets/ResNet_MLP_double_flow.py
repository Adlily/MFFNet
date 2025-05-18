#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/21 21:05
# @Author  : Sun Chunli
# @Software: PyCharm
import numpy as np
import torch.nn as nn
from torchvision import models
import torch
from config import config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes = config.NUM_CLASSES
width = config.WIDTH
fea_num = config.FEATURES_NUM
DCA_FLAG = config.DCA_FLAG
set_trainable = config.SET_TRAINABLE
#weights_dir = config.PREWEIGHTS_MODEL
weights_pre = config.WEIGHTS_MODEL

import os
#import TSNE
from nets import resnet
from nets import MLP
cnn_name = config.MODEL

class SELayer(nn.Module):
    def __init__(self, channel, reduction = 16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc       = nn.Sequential(
                        nn.Linear(channel, channel // reduction),
                        nn.ReLU(inplace = True),
                        nn.Linear(channel // reduction, channel),
                        nn.Sigmoid()
                )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y



class Resnet_MLP(nn.Module):
    """
    """

    def __init__(self,top_k,threshold=0.3):
        super(Resnet_MLP, self).__init__()

        self.hidden =200
        reduction = 2
        self.top_k = top_k
        self.channel1 = 512
        self.channel2 = 1024
        self.channel3 = 2048

        # 第 4 路输入，应用于 color_image
        # vgg16_bn 实例化
        # inputxy = Input(shape=(width, width, 3), dtype="float32", name="xy_4_input")
        # 第 4 路输入，应用于 color_image
        # vgg16_bn 实例化
        # inputxy = Input(shape=(width, width, 3), dtype="float32", name="xy_4_input")
        self.img_conv  = nn.Sequential(
             nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False),
             nn.BatchNorm2d(64),
             nn.ReLU(inplace=True),
             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.img_model_layer = resnet.se_resnet50(pretrained=True)
        #elf.img_model_layer = VGG19_new.Vgg()
        #self.img_model_layer = BoTnet.ResNet50(num_classes=10, resolution=(256,256), heads=4,pretrain = True)


        #加载处理LM features的网络
        self.lm_model1 = MLP.MLP(fea_num,self.channel1)
        self.lm_model2 = MLP.MLP(self.channel1,self.channel2)
        self.lm_model3 = MLP.MLP(self.channel2,self.channel3)


        self.fc= nn.Sequential(
                        nn.Linear(self.channel1, self.channel1 // reduction),
                        nn.ReLU(inplace = True),
                        nn.Linear(self.channel1 // reduction, self.channel1),
                        nn.Sigmoid()
                )
        channel = 1024
        self.fc2= nn.Sequential(
                        nn.Linear(self.channel2, self.channel2 // reduction),
                        nn.ReLU(inplace = True),
                        nn.Linear(self.channel2 // reduction, self.channel2),
                        nn.Sigmoid()
                )
        channel = 2048
        self.fc3= nn.Sequential(
                        nn.Linear(self.channel3, self.channel3 // reduction),
                        nn.ReLU(inplace = True),
                        nn.Linear(self.channel3 // reduction, self.channel3),
                        nn.Sigmoid()
                )

        self.bottleneck = nn.BatchNorm1d(self.channel1)
        self.bottleneck.bias.requires_grad_(False)  # no shift


        self.bottleneck2 = nn.BatchNorm1d(self.channel2)
        self.bottleneck2.bias.requires_grad_(False)  # no shift

        self.bottleneck3 = nn.BatchNorm1d(self.channel3)
        self.bottleneck3.bias.requires_grad_(False)  # no shift

        self.bottleneck4 = nn.BatchNorm1d(self.top_k)
        self.bottleneck4.bias.requires_grad_(False)  # no shift

        nn.init.normal_(self.bottleneck.weight.data, 1.0, 0.01)
        nn.init.zeros_(self.bottleneck.bias.data)

        self.avg_pool_2 = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(self.channel3,self.hidden),
            nn.BatchNorm1d(self.hidden),
            nn.Dropout(0.4),
            nn.Linear(self.hidden, num_classes),
        )

        self.classifier_lm = nn.Sequential(
            nn.Linear(self.channel3,self.hidden),
            nn.BatchNorm1d(self.hidden),
            nn.Dropout(0.4),
            nn.Linear(self.hidden, num_classes),
        )
        self.sigmoid = nn.Sigmoid()



    def selecte_k(self,feature):
        """

        :param feature: vector(N,1)
        :param k: int， top L to select value
        :return: feature[:k]
        """

        value = self.sigmoid(feature)

        sort_index = value.sort(descending = True)[1]
        #print(sort_index)
        self.sort_value = torch.zeros((feature.size(0),self.top_k))
        for i in range(feature.size(0)):
            cur_value = torch.zeros((1,self.top_k))
            for j in range(self.top_k):
                #print(feature[i][sort_index[i][j]])
                #print(cur_value[0, j])
                cur_value[0,j] = feature[i][sort_index[i][j]]
            self.sort_value[i] = cur_value
        #print(self.top_k)
        #print(self.sort_value)

        return self.sort_value
    def forward(self, img_input,lm_input,tsne=False):
        """
        :param input2: 256*256*3
        :return:
        """
        #features of different layers based on img
        img_y1= self.img_model_layer(img_input,2)
        b,c,_,_= img_y1.size()
        img_p_y1 = self.avg_pool_2(img_y1).view(b,c) #b*c
        #print(img_p_y1.size())
        lm_y1 = self.lm_model1(lm_input) #b*c

        #归一化
        bn_img_p_y1 = self.bottleneck(img_p_y1)
        bn_lm_y1 = self.bottleneck(lm_y1)
        add_y1 = bn_img_p_y1.add(bn_lm_y1)
        #print(add_y1.size())
        att_y1_ = self.fc(add_y1)
        att_y1 = att_y1_.view(b,c,1,1)
        input_2 = img_y1 * att_y1
        lm_y1 = lm_y1 * att_y1_
        #print(att_y1_.size(),lm_y1.size())

        #second attention
        #print(input_2.size())
        img_y2 = self.img_model_layer(input_2, 3)
        b,c,_,_= img_y2.size()
        img_p_y2= self.avg_pool_2(img_y2).view(b,c) #b*c

        lm_y2 = self.lm_model2(lm_y1) #b*c
        #print(lm_y2.size())
        #归一化
        bn_img_p_y2 = self.bottleneck2(img_p_y2)
        bn_lm_y2 = self.bottleneck2(lm_y2)
        add_y2 = bn_img_p_y2.add(bn_lm_y2)

        att_y2_ = self.fc2(add_y2)
        att_y2 = att_y2_.view(b,c,1,1)
        input_3 = img_y2 * att_y2
        lm_y2 = lm_y2 * att_y2_

        #second attention
        img_y3 = self.img_model_layer(input_3, 4)
        b,c,_,_= img_y3.size()
        img_p_y3 = self.avg_pool_2(img_y3).view(b,c) #b*c
        #print(img_p_y3.size())

        lm_y3 = self.lm_model3(lm_y2) #b*c
        #归一化
        bn_img_p_y3 = self.bottleneck3(img_p_y3)
        bn_lm_y3 = self.bottleneck3(lm_y3)
        add_y3 = bn_img_p_y3.add(bn_lm_y3)

        att_y3_ = self.fc3(add_y3)
        att_y3 = att_y3_.view(b,c,1,1)
        input_4 = img_y3 * att_y3
        lm_y3 = lm_y3 * att_y3_


        #分类层
        img_pool = self.avg_pool_2(input_4).view(input_4.size(0),-1)

        #img_pool1 = self.selecte_k(img_pool)
        #img_pool1 = img_pool1.to(device)

        #lm_y31 = self.selecte_k(lm_y3 )
        #lm_y31 = lm_y31.to(device)

        #img_pool_ = torch.cat((self.bottleneck4(img_pool1),self.bottleneck4(lm_y31)),1)
        img_pool = self.bottleneck3(img_pool).add(self.bottleneck3(lm_y3))
        #average
        img_pool_ = self.bottleneck3(img_pool).add(self.bottleneck3(lm_y3))
        
        
        #img_pool_ = self.selecte_k(img_pool_)
        #img_pool_ = img_pool_.to(device)
        if tsne:
            #return self.classifier(img_pool_)
            return img_pool_

        out_y = self.classifier(img_pool_)

        out_lm = self.classifier_lm(lm_y3)

        out_img = self.classifier_lm(img_p_y3)

        return out_y, out_img, out_lm



