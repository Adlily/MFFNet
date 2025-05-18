import torch
import torch.nn as nn
from config import config
weight_dir = config.WEIGHTS_MODEL
import os
classnumbers = config.NUM_CLASSES

class VGG16(nn.Module):
    def __init__(self, nums = 10):
        super(VGG16, self).__init__()
        self.nums = nums
        vgg = []

        # 第一个卷积部分
        # 112, 112, 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 第二个卷积部分
        # 56, 56, 128
        self.conv2 = nn.Sequential(
           nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
           nn.ReLU(),
           nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
           nn.ReLU(),
           nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 第三个卷积部分
        # 28, 28, 256
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 第四个卷积部分
        # 14, 14, 512

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


        # 第五个卷积部分
        # 7, 7, 512
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 将每一个模块按照他们的顺序送入到nn.Sequential中,输入要么事orderdict,要么事一系列的模型，遇到上述的list，必须用*号进行转化
        vgg.append(self.conv1)
        vgg.append(self.conv2)
        vgg.append(self.conv3)
        vgg.append(self.conv4)
        vgg.append(self.conv5)
        self.main = nn.Sequential(*vgg)

        # 全连接层
        classfication = []
        # in_features四维张量变成二维[batch_size,channels,width,height]变成[batch_size,channels*width*height]
        classfication.append(nn.Linear(in_features=512 * 7 * 7, out_features=4096))  # 输出4096个神经元，参数变成512*7*7*4096+bias(4096)个
        classfication.append(nn.ReLU())
        classfication.append(nn.Dropout(p=0.5))
        classfication.append(nn.Linear(in_features=4096, out_features=4096))
        classfication.append(nn.ReLU())
        classfication.append(nn.Dropout(p=0.5))
        classfication.append(nn.Linear(in_features=4096, out_features=self.nums))

        self.classfication = nn.Sequential(*classfication)
  
    def forward(self, x,prefla=1):
        x_ = x
        if prefla==2:
            x = self.conv1(x)
            x = self.conv2(x)
            output = self.conv3(x)
        elif prefla==3:
            output = self.conv4(x_)
        elif prefla==4:
            
            output = self.conv5(x_)
        else:
            raise ValueError
        return output
        
def Vgg():
    """
    if pretrain:
        weights_res = os.path.join(weight_dir,'vgg16-397923af.pth')
        #net = models.vgg16(pretrained=True)   #从预训练模型加载VGG16网络参数
        
        if os.path.exists(weights_res):
            net = VGG16(classnumbers)
            net.load_state_dict(torch.load(weights_res))
        else:
            net = VGG16(classnumbers)   #从预训练模型加载VGG16网络参数
    """
    net = VGG16(classnumbers)
    return net