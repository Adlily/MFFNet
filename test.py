#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/16 19:33
# @Author  : Sun Chunli
# @Software: PyCharm
import sys
import numpy as np
import torch
from Mydataset import Mydataset_lm_img
import random
import os
from sklearn.model_selection import StratifiedKFold

batch_size = 8

class Logger(object):
    def __init__(self,fileN ="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN,"a")
 
    def write(self,message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass
 

from nets import ResNet_MLP_double_flow_cross_self as ResNet_MLP
device = "cpu"#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
layer = '' #需要提取的某一层的特征
model = model = ResNet_MLP.Resnet_MLP(256,threshold=0.2)
#model = nn.Sequential(*list(model_.children())[:-1])
#print(model)
from config import config
image_dir = config.IMAGE_DIR
lm_dir = config.LM_DIR
numclass = config.NUM_CLASSES

# model = models.resnet50(pretrained=True)
weight_path = '/FFNet/pretrained_models/1_fold_best.pt'
#f = open( os.path.join(path,'a.log'), 'a')
#sys.stdout = f

def get_k_fold_data(tra_index, test_index, img_xy_data,img_label,lm_data):

    """

    :param tra_index: 当前k 折的训练数据索引
    :param test_index: 当前k 折的测试数据索引
    :param img_xy_data: 数据
    :param img_label: label
    ：neuname: samples name
    :return:
    """

    # img_data
    tr_data_ = []
    te_data_ = []

    index = [i for i in range(len(tra_index))]
    random.shuffle(index)

    # 此时已经打乱顺序
    index1 = [i for i in range(len(test_index))]
    random.shuffle(index1)

    #for i in range(len(img_xy_data)):

    train_img_xy_set_i = img_xy_data[tra_index]
    test_img_xy_set_i = img_xy_data[test_index]    # 打乱顺序
    # print(train_set[0,-1])

    train_lm_set = lm_data[tra_index]
    test_lm_set = lm_data[test_index]


    train_data_xy_i = train_img_xy_set_i[index]
    test_data_xy_i = test_img_xy_set_i[index1]

    train_lm_i = train_lm_set[index]
    test_lm_i = test_lm_set[index1]

    y_tra = img_label[tra_index]
    y_tes = img_label[test_index]


    # if one_hot:
    #      # 转换label格式为多类标签
    #      train_y = label_to_one_hot(y_tra[index], num_class)
    #      test_y = label_to_one_hot(y_tes[index1], num_class)
    # else:
    train_y =  y_tra[index]
    test_y = y_tes[index1]
    print('train,test', len(tra_index), len(test_index))


    return train_data_xy_i,train_lm_i,train_y,test_data_xy_i[:-2],test_lm_i[:-2],test_y[:-2]

def test(model,weight_path,img_dir,save_path):

    """

    :param model_path: .pt
    :param img_path: image/1.png,2.png,3.png
    :param save_path:
    :return:
    """
    #model = GVCNN_resnet_se.GVCNN_resnet_se(,5)
    model = model.to(device)
    #model = torch.nn.DataParallel(model)
    check_poing = torch.load(weight_path,map_location='cpu')
    # 去除模型中所有权重的 module 关键词
    new_state_dict = {}
    for k, v in check_poing.items():
        name = k.replace('module.', '')  # 如果键中包含 module，去除掉
        new_state_dict[name] = v
        #print(check_poing)
    # if use_gpu:
        # state_dict = model.module.state_dict()
    # else:
        # tate_dict = model.state_dict()
        
    model.load_state_dict(new_state_dict)
    # for n in model.named_modules():
    #     print(n)
    torch.no_grad()

    #处理数据集

    #读取lm_data
    img_xy_data, img_label = multi_input(image_dir,lm_dir)
    #neu_n1,image1 = data_generator.read_image_path_label(image_dir)
    lm_data,lm_label  = read_excel_by_xlrd_neuname(img_xy_data,lm_dir)

    img_xy_data, img_label = multi_input(image_dir,lm_dir)
    neu_name, img_label_ = spilt_neuname(img_xy_data, img_label)

    img_xy_data, img_label = np.array(img_xy_data), np.array(img_label)
    neu_name, img_label_= np.array(neu_name), np.array(img_label_)
    #lm_data,lm_label=RFECV_feature_select.rfecv_cv(lm_data,lm_label)
    #img_xy_data, img_label = Mydataset.read_only_image_path_label(img_dir)
    split = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    

    i = 0
    best_y_t = []     
    f_tsne = []
    for tra_index, test_index in split.split(img_xy_data, img_label):

        print('*********')
        #i = i + 1
        #if not (i == best_fold):
        #    continue
        
        X_train, LM_train,Y_train, X_test, LM_test,Y_test = get_k_fold_data(tra_index, test_index, img_xy_data,img_label,lm_data)



        train_dataset =  Mydataset_lm_img(X_train, LM_train,Y_train)
        test_dataset = Mydataset_lm_img(X_test, LM_test,Y_test)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            # num_workers=args.workers,
            shuffle=False,
            pin_memory=False
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            # num_workers=args.workers,
            shuffle=False,
            pin_memory=False
        )
        #测试数据

        model.eval()
   
        with torch.no_grad():  # 不计算梯度
            iter_cnt = 0
            for step, (b_x,lm_x, b_y) in enumerate(test_loader):
                iter_cnt += 1
                
                # 转换数据形式为：[b,c,w,w]
                b_x0 = b_x.permute(0, 3, 1, 2)
                b_x1 = lm_x
                b_x0 = b_x0.to(device, dtype=torch.float32)
                b_x1 = b_x1.to(device, dtype=torch.float32)
                b_y = b_y#.to(device, dtype=torch.long)



                output  = model(b_x0, b_x1,tsne=True)

if __name__=='__main__':

    tsne_off_line(model,weight_path,image_dir,save_path)

