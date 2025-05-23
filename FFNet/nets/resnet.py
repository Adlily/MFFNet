#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/9 21:53
# @Author  : Sun Chunli
# @Software: PyCharm

from torchvision import models
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import torch
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
from collections import OrderedDict
import math
from torchvision import models
import os
from config import config
weight_dir = config.WEIGHTS_MODEL


__all__ = ['se_resnet18', 'se_resnet34', 'se_resnet50', 'se_resnet101', 'se_resnet152']

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

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.se  = SELayer(planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        #out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        #self.se  = SELayer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        #out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x,prefla=1):
        x_ = x
        if prefla==1:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            output = self.layer1(x)
        elif prefla==2:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x_ = self.layer1(x)
            output = self.layer2(x_)
        elif prefla==3:
            output = self.layer3(x_)
        elif prefla==4:
            output = self.layer4(x_)
        else:
            raise ValueError
        return output

def se_resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    
    if pretrained:
        weight_res = os.path.join(weight_dir, 'resnet18-5c106cde.pth')
        if not os.path.exists(weight_res):
           raise ValueError('pre-trained model isnot exist!')
        net = models.resnet18(pretrained=False)
        net.load_state_dict(torch.load(weight_res))

        resnet50_param = net.state_dict()

        #checkpoint = remove_fc(resnet50_param)

        model.load_state_dict(resnet50_param, strict=False)

        return model
    return model


def se_resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    
    if pretrained:
        weight_res = os.path.join(weight_dir, 'resnet34-b627a593.pth')
        if not os.path.exists(weight_res):
           raise ValueError('pre-trained model isnot exist!')
        net = models.resnet34(pretrained=False)
        net.load_state_dict(torch.load(weight_res))

        resnet50_param = net.state_dict()

        #checkpoint = remove_fc(resnet50_param)

        model.load_state_dict(resnet50_param, strict=False)

        return model
    
    return model


def se_resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    if pretrained:
        weight_res = os.path.join(weight_dir, 'resnet50-19c8e357.pth')
        if not os.path.exists(weight_res):
           raise ValueError('pre-trained model isnot exist!')
        net = models.resnet50(pretrained=False)
        net.load_state_dict(torch.load(weight_res))

        resnet50_param = net.state_dict()

        #checkpoint = remove_fc(resnet50_param)

        model.load_state_dict(resnet50_param, strict=False)

        return model

    return model



def se_resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        weight_res = os.path.join(weight_dir, 'resnet101-63fe2227.pth')
        if not os.path.exists(weight_res):
           raise ValueError('pre-trained model isnot exist!')
        net = models.resnet101(pretrained=False)
        net.load_state_dict(torch.load(weight_res))

        resnet50_param = net.state_dict()

        #checkpoint = remove_fc(resnet50_param)

        model.load_state_dict(resnet50_param, strict=False)

        return model
        
    
    return model


def se_resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def remove_fc(params):
    """
    :param parems: 移除掉fc的权重
    :return:
    """

    for k,v in params.items():
        if k=='fc.weight' or 'fc.bias':
            del params[k]

def demo():
    net = se_resnet50(True,num_classes=5)
    print(net)
    #y = net(torch.randn(1, 3, 224,224))
    #print(y.size())

#demo()
