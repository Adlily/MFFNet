#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/4 20:43
# @Author  : Sun Chunli
# @Software: PyCharm

# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/21 21:05
# @Author  : Sun Chunli
# @Software: PyCharm
import numpy as np
import torch.nn as nn
from torchvision import models
import torch
import math
from config import config
from .IWPA import IWPA
from .GroupSchema import GroupSchema
from .self_attention import EfficientAttention
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = config.NUM_CLASSES
width = config.WIDTH
fea_num = config.FEATURES_NUM
DCA_FLAG = config.DCA_FLAG
set_trainable = config.SET_TRAINABLE
# weights_dir = config.PREWEIGHTS_MODEL
weights_pre = config.WEIGHTS_MODEL

import os
# import TSNE
from nets import resnet,VGG19_new,VGG16_new
from torchvision.models import densenet121,mobilenet_v3_small
from nets import MLP

cnn_name = config.MODEL


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class inter_features(nn.Module):
    """
    transformer fusion
    """

    def __init__(self,hidden_dims,out_dims):
        super(inter_features,self).__init__()

        self.hidden = hidden_dims
        self.out = out_dims
        #print('hidden',self.hidden,self.out)
        # for features1
        self.q_f1_fc = nn.Linear(self.hidden, self.out)
        self.k_f1_fc = nn.Linear(self.hidden,self.out)
        self.v_f1_fc = nn.Linear(self.hidden,self.out)
        self.f1_fc = nn.Linear(self.out, self.out)
        #for fetures 2
        self.q_f2_fc = nn.Linear(self.hidden, self.out)
        self.k_f2_fc = nn.Linear(self.hidden,self.out)
        self.v_f2_fc = nn.Linear(self.hidden,self.out)
        self.f2_fc = nn.Linear(self.out,self.out)

        self.bn = nn.BatchNorm1d(self.hidden)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self,fea1,fea2):

        #for image reshape
        b,c = fea1.size(0), fea1.size(1)
        im_f1 = self.avg_pool(fea1)
        #print(im_f1.size())
        img_f1 = im_f1.view(b,c)
        #bn
        img_input = self.bn(img_f1)
        lm_input = self.bn(fea2)

        #get the key, query, and value of image feature
        img_q = self.q_f1_fc(img_input)
        img_k = self.k_f1_fc(img_input)
        img_v = self.v_f1_fc(img_input)

        # get the key, query, and value of LM feature
        lm_q = self.q_f2_fc(lm_input)
        lm_k = self.k_f2_fc(lm_input)
        lm_v = self.v_f2_fc(lm_input)

        #交互
        img_to_lm  = (lm_q*img_k)/pow(self.out,0.5)
        img_weight = self.softmax(img_to_lm)
        img_attention = img_weight*img_v

        lm_to_img  = (img_q*lm_k)/pow(self.out,0.5)
        #print(lm_to_img.size())
        lm_weight = self.softmax(lm_to_img)
        #print(lm_weight.size())
        lm_attention = lm_weight*lm_v
        #print(lm_attention.size())

        lm_ouput = self.f2_fc(fea2.add(lm_attention))

        img_output_ = self.f1_fc(img_input.add(img_attention))
        #因为fea1:B,C,H,W
        img_output_ = img_output_.reshape(b,c,1,1)
        img_output = fea1 *img_output_

        return lm_ouput,img_output
        
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class MHA_FFN(nn.Module):
    def __init__(self,C):
        super(MHA_FFN,self).__init__()

        self.cross_attne = nn.MultiheadAttention(C//2,4)
        self.linear  = nn.Linear(C//2,C//2)
        self.linear2 = nn.Linear(C//2,C)
        self.linear3 = nn.Linear(C,C)
        self.linear4 = nn.Linear(C,C//2)
        self.bn1 = nn.BatchNorm1d(C)
        self.bn2 = nn.BatchNorm1d(C//2)
        self.conv1 = nn.Conv1d(C,C,1)
        self.conv2 = nn.Conv2d(C,C,1)
        self.avg_pool_2 = nn.AdaptiveAvgPool2d((1,1))

        self.sigmoid = nn.Sigmoid()
        self.ReLU = nn.ReLU()

    def forward(self,x1,x2):
        """
        :param x1: fc 或者avgpool之后的 [BS,C]
        :param x2:fc 或者avgpool之后的 [BS,C]
        entropy as the metric
        :return:
        """
        #print(x1.size(),x2.size())
        flag_orix1 = False
        ori_x1 = x1
        ori_x2 = x2
        if len(x1.size())>=4:
            #print('img zzzzzz',x1.size())
            x1 = self.avg_pool_2(x1).view(x1.size(0),-1)
            #print(x1.size())
        if len(x2.size())>=4:
            #print('img ',x2.size())
            flag_orix1 = True
            x2 = self.avg_pool_2(x2).view(x2.size(0),-1)

        B,C = x2.size()
        Q = self.linear4(x2)
        K,V = self.linear4(x1),self.linear4(x1),
        mul_ouput,_ = self.cross_attne(Q,K,V)
        #residual+ffn
        bn1_output = self.bn2(Q+mul_ouput)
        #FFN
        last_output = self.bn1(self.linear2(bn1_output) + self.ReLU(self.linear3(self.ReLU(self.linear2(bn1_output)))))
        
        last_output #此时返回的是x1
        #此时作为mask做添加到原来的ori_x1
        if len(ori_x2.size())==4:
            last_output = last_output.unsqueeze(-1).unsqueeze(-1)
            #print(ori_x2.size())
            #print((ori_x2*last_output).size())
            xq_mt = self.conv2(ori_x2)  + ori_x2*last_output
        else:
            xq_mt = self.linear3(ori_x2)  + ori_x2*last_output
        return xq_mt


class Resnet_MLP(nn.Module):
    """
    """

    def __init__(self, top_k,threshold=0.8,location= 'P1'):
        super(Resnet_MLP, self).__init__()

        self.hidden = 200
        reduction = 2
        self.top_k = top_k
        #512,resnet101和resnet50是512，1024,2048;resnet18，resnet34:128,256,512;vgg19:256,515,512
        #denset:64,64,64;mobilenet_v3:48,48,96
        self.channel1 = 512 #512
        self.channel2 = 1024 # 1024
        self.channel3 = 2048 #2048
        self.threshold = threshold
        self.location = location

        # 第 4 路输入，应用于 color_image
        # vgg16_bn 实例化
        # inputxy = Input(shape=(width, width, 3), dtype="float32", name="xy_4_input")
        # 第 4 路输入，应用于 color_image
        # vgg16_bn 实例化
        # inputxy = Input(shape=(width, width, 3), dtype="float32", name="xy_4_input")
        self.img_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.img_model_layer = resnet.se_resnet50(pretrained=True)
        #self.img_model_layer = resnet.se_resnet101(pretrained=True)
        #self.img_model_layer = resnet.se_resnet34(pretrained=True)
        #self.img_model_layer = VGG19_new.Vgg()
        #self.img_model_layer = VGG16_new.Vgg()
        # self.img_model_layer = BoTnet.ResNet50(num_classes=10, resolution=(256,256), heads=4,pretrain = True)
        #self.img_model_layer = densenet121(num_classes=num_classes)
        #self.img_model_layer = mobilenet_v3_small(pretrain=True)

        # 加载处理LM features的网络
        self.lm_model1 = MLP.MLP(fea_num, self.channel1)
        self.lm_model2 = MLP.MLP(self.channel1, self.channel2)
        self.lm_model3 = MLP.MLP(self.channel2, self.channel3)

        self.fc = nn.Sequential(
            nn.Linear(self.channel1, self.channel1//reduction),
            #nn.BatchNorm1d(self.channel1),
            nn.ReLU(inplace=True),
            nn.Linear(self.channel1//reduction,self.channel1),
            nn.Sigmoid(),
        )
        channel = 1024
        self.fc2 = nn.Sequential(
            nn.Linear(self.channel2, self.channel2//reduction),
            #nn.BatchNorm1d(self.channel1),
            nn.ReLU(inplace=True),
            nn.Linear(self.channel2//reduction,self.channel2),
            nn.Sigmoid(),
        )
        channel = 2048
        self.fc3 = nn.Sequential(
            nn.Linear(self.channel3, self.channel3//reduction),
            #nn.BatchNorm1d(self.channel1),
            nn.ReLU(inplace=True),
            nn.Linear(self.channel3//reduction,self.channel3),
            nn.Sigmoid(),
        )


        self.fc11 = nn.Sequential(
            nn.Linear(2*self.channel1, self.channel1//reduction),
            #nn.BatchNorm1d(self.channel1),
            nn.ReLU(inplace=True),
            nn.Linear(self.channel1//reduction,self.channel1),
            #nn.Sigmoid(),
        )
        channel = 1024
        self.fc21 = nn.Sequential(
            nn.Linear(2*self.channel2, self.channel2//reduction),
            #nn.BatchNorm1d(self.channel1),
            nn.ReLU(inplace=True),
            nn.Linear(self.channel2//reduction,self.channel2),
            #nn.Sigmoid(),
        )
        channel = 2048
        self.fc31 = nn.Sequential(
            nn.Linear(2*self.channel3, self.channel3//reduction),
            #nn.BatchNorm1d(self.channel1),
            nn.ReLU(inplace=True),
            nn.Linear(self.channel3//reduction,self.channel3),
            #nn.Sigmoid(),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(2*self.channel1, self.channel1,1),
            nn.BatchNorm2d(self.channel1),
            nn.ReLU(inplace=True),
        )
        channel = 1024
        self.conv2 = nn.Sequential(
            nn.Conv2d(2*self.channel2, self.channel2,1),
            nn.BatchNorm2d(self.channel2),
            nn.ReLU(inplace=True),
        )
        channel = 2048
        self.conv3 = nn.Sequential(
            nn.Conv2d(2*self.channel3, self.channel3,1),
            nn.BatchNorm2d(self.channel3),
            nn.ReLU(inplace=True),
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
            nn.Linear(self.channel3, self.hidden),
            nn.BatchNorm1d(self.hidden),
            nn.Dropout(0.4),
            nn.Linear(self.hidden, num_classes),
        )

        self.classifier_lm = nn.Sequential(
            nn.Linear(self.channel3, self.hidden),
            nn.BatchNorm1d(self.hidden),
            nn.Dropout(0.4),
            nn.Linear(self.hidden, num_classes),
        )
        
        self.img_iwpa1 = IWPA(self.channel1)
        self.img_iwpa2 = IWPA(self.channel2)
        self.img_iwpa3 = IWPA(self.channel3)
        
        self.view_score = GroupSchema(self.channel1)
        init_weights(self.view_score)
        self.view_score2 = GroupSchema(self.channel2)
        init_weights(self.view_score2)
        self.view_score3 = GroupSchema(self.channel3)
        init_weights(self.view_score3)
        
        self.sigmoid = nn.Sigmoid()
        #print('channnel1',self.channel1,self.channel2,self.channel3)
        self.inter1 =  inter_features(self.channel1,self.channel1)
        self.inter2 = inter_features(self.channel2, self.channel2)
        self.inter3 = inter_features(self.channel3, self.channel3)

        self.cross_attne1 = MHA_FFN(self.channel1)
        self.cross_attne2 = MHA_FFN(self.channel2)
        self.cross_attne3 = MHA_FFN(self.channel3)

        self.locaa1 = EfficientAttention(self.channel1,self.channel1,self.channel1)
        self.locaa2 = EfficientAttention(self.channel2,self.channel2,self.channel2)
        self.locaa3 = EfficientAttention(self.channel3,self.channel3,self.channel3)

    def control_gate_score(self,x1):
        """
        :param x1: fc 或者avgpool之后的 [BS,C]
        :param x2:fc 或者avgpool之后的 [BS,C]
        entropy as the metric
        :return:
        """
        ent = x1.sigmoid()
        #数值 线性变化呢
        #print('ent size is',ent.size())
        mask = ent/math.sqrt(x1.size(1))
        #mask = ent.ge(threshold).float() #熵小于threshold
        #print('mask size is',mask.size())
        #x2_gate = x2*mask
        #x2_softmax  = x2_gate.softmax(1)
        #print('x2_softmax is',x2_gate.size())
        return mask.softmax(1)
    
    def control_gate(self,x1,x2,threshold):
        """
        :param x1: fc 或者avgpool之后的 [BS,C]
        :param x2:fc 或者avgpool之后的 [BS,C]
        entropy as the metric
        :return:
        """
        flag_orix1 = False
        ori_x1 = x1
        ori_x2 = x2
        if len(x1.size())>=4:
            #print('img zzzzzz',x1.size())
            x1 = self.avg_pool_2(x1).view(x1.size(0),-1)
            #print(x1.size())
        if len(x2.size())>=4:
            #print('img ',x2.size())
            flag_orix1 = True
            x2 = self.avg_pool_2(x2).view(x2.size(0),-1)
        
        # ent = -(x1.softmax(1) * x1.log_softmax(1)).sum(1)#求得熵 H(Pi)
        # ent = x1.sigmoid()
        # #数值 线性变化呢
        # #print('ent size is',ent.size())
        # mask = ent.ge(threshold).float() #熵小于threshold
        # #print('mask size is',mask.size())
        # x2_gate = (1+mask)
        # #直接利用X2作为Query,没有threshold的选择
        # #原始的X1先自己计算weights
        # #print(x1.size(),x2.size())
        # x1_ = x1.unsqueeze(1)
        # x2_ = x2.unsqueeze(1).permute(0,2,1)
        # #print(x1_.size(),x2_.size())

        # weight = x2_*x1_/pow(2048,0.5) #这里是Q2作为Query,Q1为Key,如果两者相似性较强，则是我们寻找的共同的特征
        # #weight = torch.mean(weight,1) #这里是B,C,C的大小
        # #在这里加入gate门控，控制不相关的区域，仅加强相关的区域
        # #weigth = torch.sigmoid(weigth) #一部分是channel层的带gate的，一部分是点乘的方法；
        # #这里的threshold为每一个channel的最大值或均值；
        # #self.threshold = torch.mean(x2,1) #or torch.max(x2,2)[1]
        # #print(weight.size())
        # x22 = x2.unsqueeze(1)
        # mask_weight = weight.ge(self.threshold).float()
        
        # #print(x2_.size(),x22.size(),mask_weight.size())
        # wegith_x = torch.matmul(x22,mask_weight)
        # #print(wegith_x.size())

        # x2_softmax  = x2_gate.softmax(1)
        # #print('x2_softmax is',x2_gate.size())
        # if flag_orix1:
        #     #print('...')
        #     x2_softmax = x2_softmax.view(ori_x2.size(0),ori_x2.size(1),1,1)
        #     #print(x2_softmax.size())
        # #print((ori_x2*x2_softmax).size())
        # #print((ori_x2*x2_softmax).add(ori_x2).size())
        # #return ori_x2*x2_softmax .add(ori_x2)
        # weig_s1 = wegith_x.squeeze(1) #N,C,1
        # #weig_s1 = weig_s.permute(0,3,1,2)
        # #print(weig_s1.size(),ori_x2.size())
        # we_x = weig_s1*ori_x2
        #print(we_x.size())
        #return we_x+ori_x2
        #使用cross-attention MHA
        return self.cross_attne3(x1,x2)
        
  
    def selecte_k(self, feature):
        """

        :param feature: vector(N,1)
        :param k: int， top L to select value
        :return: feature[:k]
        """

        value = self.sigmoid(feature)

        sort_index = value.sort(descending=True)[1]
        # print(sort_index)
        self.sort_value = torch.zeros((feature.size(0), self.top_k))
        for i in range(feature.size(0)):
            cur_value = torch.zeros((1, self.top_k))
            for j in range(self.top_k):
                # print(feature[i][sort_index[i][j]])
                # print(cur_value[0, j])
                cur_value[0, j] = feature[i][sort_index[i][j]]
            self.sort_value[i] = cur_value
        # print(self.top_k)
        # print(self.sort_value)

        return self.sort_value

    def forward(self, img_input, lm_input, tsne=False):
        """
        :param input2: 256*256*3
        :return:
        """
        #features of different layers based on img
        # img_y1= self.img_model_layer(img_input,2)
        # b,c,_,_= img_y1.size()
        # img_p_y1 = self.avg_pool_2(img_y1).view(b,c) #b*c
        # #print(img_p_y1.size())
        # lm_y1 = self.lm_model1(lm_input) #b*c
        
        # #这里是交互模块
        # #进行交互运行
        # bn_img_p_y1 = self.bottleneck(img_p_y1)
        # bn_lm_y1 = self.bottleneck(lm_y1)

        # add_y1 = bn_img_p_y1.add(bn_lm_y1)
        # #print(add_y1.size())
        # att_y1_ = self.fc(add_y1)
        # att_y1 = att_y1_.view(b,c,1,1)
        # self_img1 = img_y1 * att_y1
        # self_lm1 = lm_y1 * att_y1_
        
        # if 'P1' in self.location or 'P1'==self.location:

            
        #     cross_lm1 = self.cross_attne1(img_p_y1,lm_y1)
        #     lm_y1_c = self_lm1.add(cross_lm1)
        #     input_lm2 = self.fc(lm_y1_c)

        #     cross_img1 = self.cross_attne1(lm_y1,img_y1)
        #     img_y1_c = self_img1.add(cross_img1)
        #     input_2 = self.conv(img_y1_c)
        # else:
        #    input_lm2 = self_lm1
        #    input_2 =  self_img1

        # # #归一化

        # #print('after fusion inter',att_y1_.size(),lm_y1.size())
        
        # #放在后面的模态内attention
        # #lm_y 分枝是从头开始训练的网络，因此在网络开始时acc很低，训练后期acc才可以达到70,80
        # #lm_y1 = self.control_gate(input_2,lm_y1,self.threshold)
        # #input_2 = self.control_gate(lm_y1,input_2,self.threshold)
        
        # #print(input_2.size(),lm_y1.size())

        # #second attention
        # #print(input_2.size())
        # img_y2 = self.img_model_layer(input_2, 3)
        # b,c,_,_= img_y2.size()
        # img_p_y2= self.avg_pool_2(img_y2).view(b,c) #b*c

        # lm_y2 = self.lm_model2(input_lm2) #b*c
        

        # #归一化
        # bn_img_p_y2 = self.bottleneck2(img_p_y2)
        # bn_lm_y2 = self.bottleneck2(lm_y2)

        # # lm_y2 = bn_lm_y2 # self.locaa2(bn_lm_y2)
        # # input_3 = img_y2 #*self.locaa2(bn_img_p_y2).view(b,c,1,1)

        # add_y2 = bn_img_p_y2.add(bn_lm_y2)

        # att_y2_ = self.fc2(add_y2)
        # att_y2 = att_y2_.view(b,c,1,1)
        # self_img2 = img_y2 * att_y2
        # self_lm2 = lm_y2 * att_y2_

        # if 'P2' in self.location or 'P2'==self.location:
        #     #进行交互运行
        #     #self_lm2 = self.locaa1(lm_y2)
        #     cross_lm2 = self.cross_attne2(img_p_y2,lm_y2)
        #     lm_y2_c = torch.cat([self_lm2,cross_lm2],dim=1)
        #     input_lm3 = self.fc(lm_y2_c)

        #     #self_img2 = self.locaa1(img_y2)
        #     cross_img2 = self.cross_attne2(lm_y2,img_y2)
        #     img_y2_c = torch.cat([self_img2,cross_img2],dim=1)
        #     input_3 = self.conv2(img_y2_c)
        # else:
        #    input_lm3 = self_lm2
        #    input_3 =  self_img2
        # #print(lm_y2.size())


        # # lm_y2 = lm_y21 + lm_y2 * self.cross_attne2(input_31,lm_y2)
        # # input_3 = input_31 + img_y2 *self.cross_attne2(lm_y2,input_31).view(b,c,1,1)
       
        # #lm_y2 = self.control_gate(input_3,lm_y2,self.threshold)
        # #input_3 = self.control_gate(lm_y2,input_3,self.threshold)

        # #second attention
        # img_y3 = self.img_model_layer(input_3, 4)
        # b,c,_,_= img_y3.size()
        # img_p_y3 = self.avg_pool_2(img_y3).view(b,c) #b*c
        # #print(img_p_y3.size())

        # lm_y3 = self.lm_model3(input_lm3) #b*c
        
        # bn_img_p_y3 = self.bottleneck3(img_p_y3)
        # bn_lm_y3 = self.bottleneck3(lm_y3)

        # # lm_y3 = self.locaa3(bn_lm_y3)
        # # input_4 = img_y3 * self.locaa3(bn_img_p_y3).view(b,c,1,1)

        # add_y3 = bn_img_p_y3.add(bn_lm_y3)
        # att_y3_ = self.fc3(add_y3)
        # att_y3 = att_y3_.view(b,c,1,1)
        # self_img3 = img_y3 * att_y3
        # self_lm3 = lm_y3 * att_y3_

        # if 'P3' in self.location or 'P3'==self.location:
        #     #进行交互运行
        #     self_lm3 = self.locaa1(lm_y3)
        #     cross_lm3 = self.cross_attne3(img_p_y3,lm_y3)
        #     lm_y3_c = self_lm3.add(cross_lm3) #torch.cat([self_lm3,cross_lm3],dim=1)
        #     lm_y4 = self.fc(lm_y3_c)

        #     self_img3 = self.locaa1(img_y3)
        #     cross_img3 = self.cross_attne3(lm_y3,img_y3)
        #     img_y3_c = torch.cat([self_img3,cross_img3],dim=1)
        #     input_4 = self.conv(img_y3_c)
        # else:
        #     lm_y4 = self_lm3
        #     input_4 = self_img3
        
        # #归一化

        # # lm_y3 = lm_y31 + lm_y3 * self.control_gate(input_41,lm_y3,self.threshold)
        # # input_4 = input_41 + img_y3 *self.control_gate(lm_y3,input_41,self.threshold).view(b,c,1,1)


        # #分类层
        # img_pool = self.avg_pool_2(input_4).view(input_4.size(0),-1)

        # #img_pool1 = self.selecte_k(img_pool)
        # #img_pool1 = img_pool1.to(device)

        # #lm_y31 = self.selecte_k(lm_y3 )
        # #lm_y31 = lm_y31.to(device)

        # #img_pool_ = torch.cat((self.bottleneck4(img_pool1),self.bottleneck4(lm_y31)),1)
        # #img_pool_ = self.bottleneck3(img_pool).add(self.bottleneck3(lm_y3))
        # #average
        # #img_pool_ = self.bottleneck3(img_pool).add(self.bottleneck3(lm_y3))
        # # fusion after the gate control
        # img_p_y4 = self.bottleneck3(img_pool)
        # lm_y4 = self.bottleneck3(lm_y4)

        # if 'P4' in self.location or 'P4'==self.location:
        #     #进行交互运行
        #     #self_lm4 = self.locaa1(lm_y4)
        #     cross_lm4 = self.cross_attne3(img_p_y4,lm_y4)
        #     lm_y4_c = lm_y4.add(cross_lm4) #torch.cat([lm_y4,cross_lm4],dim=1)
        #     lm_y41 = self.fc3(lm_y4_c)

        #     #self_img4 = self.locaa1(img_p_y4)
        #     cross_img4 = self.cross_attne3(lm_y4,img_p_y4)
        #     img_y4_c = torch.cat([img_p_y4,cross_img4],dim=1)
        #     img_p_y41 = self.conv3(img_y4_c)

        # # lm_y3_ = self.locaa3(lm_y3_)
        # # img_pool_s = self.locaa3(img_pool)

        # # img_p_y3 = self.control_gate(lm_y3_,img_pool_s,self.threshold)
        # # lm_y3 = self.control_gate(img_pool_s,lm_y3_,self.threshold)

        # else:
        #     #print('---')
        #     lm_y41 = self.cross_attne3(img_p_y4,lm_y4)
        #     img_p_y41 = self.cross_attne3(lm_y4,img_p_y4)

        # img_pool_ = img_p_y41.add(lm_y41)
        
        # #img_pool_ = self.selecte_k(img_pool_)
        # #img_pool_ = img_pool_.to(device)
        # if tsne:
        #     #return self.classifier(img_pool_)
        #     return img_pool_

        # #print(img_pool_.size())
        # out_y = self.classifier(img_pool_)

        # out_lm = self.classifier_lm(lm_y41)

        # out_img = self.classifier_lm(img_p_y41)

        # return out_y, out_img, out_lm

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
        #print('after fusion inter',att_y1_.size(),lm_y1.size())
        #input_2 = img_y1
        #放在后面的模态内attention
        #lm_y 分枝是从头开始训练的网络，因此在网络开始时acc很低，训练后期acc才可以达到70,80
        #input_2 = self.conv(torch.cat([self.cross_attne1(lm_y1,img_y1),input_2],dim=1))
        #lm_y1 = self.fc11(torch.cat([self.cross_attne1(img_y1,lm_y1),lm_y1],dim=1))
        
        #print(input_2.size(),lm_y1.size())

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
        #input_3 = img_y2
       
        #input_3 = self.conv2(torch.cat([self.cross_attne2(lm_y2,img_y2),input_3],dim=1))
        #lm_y2 = self.fc21(torch.cat([self.cross_attne2(img_y2,lm_y2),lm_y2],dim=1))

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

        input_4 = self.conv3(torch.cat([self.cross_attne3(lm_y3,img_y3),input_4],dim=1)) #self.cross_attne3(lm_y3,input_4)#.add(input_4) #
        lm_y3 =  self.fc31(torch.cat([self.cross_attne3(img_y3,lm_y3),lm_y3],dim=1))
        #self.cross_attne3(input_4,lm_y3)#.add(lm_y3)#self.fc31(torch.cat([self.cross_attne3(img_y3,lm_y3),lm_y3],dim=1))

        #input_4 = img_y3
        #分类层
        img_pool = self.avg_pool_2(input_4).view(input_4.size(0),-1)

        #img_pool1 = self.selecte_k(img_pool)
        #img_pool1 = img_pool1.to(device)

        # lm_y31 = self.selecte_k(lm_y3 )
        # lm_y31 = lm_y31.to(device)

        # img_pool_ = torch.cat((self.bottleneck4(img_pool1),self.bottleneck4(lm_y31)),1)
        # img_pool_ = self.bottleneck3(img_pool).add(self.bottleneck3(lm_y3))
        # average
        # img_pool_ = self.bottleneck3(img_pool).add(self.bottleneck3(lm_y3))
        # fusion after the gate control
        img_pool_s = self.bottleneck3(img_pool)
        lm_y3_ = self.bottleneck3(lm_y3)

        lm_y31_ = self.locaa3(lm_y3_)
        img_pool_s1 = self.locaa3(img_pool)
        
        img_pool_s = self.fc31(torch.cat([self.cross_attne3(lm_y3_,img_pool_s1),img_pool_s],dim=1))
        lm_y3 = self.fc31(torch.cat([self.cross_attne3(img_pool_s,lm_y3_),lm_y31_],dim=1))

        #lm_y3 = lm_y3_
        
        img_pool_ = img_pool_s.add(lm_y3)
        #img_pool_ = torch.mean(torch.stack((img_pool_s, lm_y3)), dim=0)
        #img_pool_ = self.fc31(torch.cat([img_pool_s, lm_y3],dim=1))
        
        #img_pool_ = self.selecte_k(img_pool_)
        #img_pool_ = img_pool_.to(device)
  

        #print(img_pool_.size())
        out_y = self.classifier(img_pool_)

        out_lm = self.classifier_lm(lm_y3)

        out_img = self.classifier_lm(img_p_y3)

        if tsne:
            #return self.classifier(img_pool_)
            return out_y

        return out_y, out_img, out_lm


