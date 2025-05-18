##!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/28 10:09
# @Author  : Sun Chunli
# @Software: PyCharm

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 22:00:05 2020

@author: Adlily
"""


import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
from sklearn.model_selection import StratifiedKFold

from visual import metric
from visual import save_results
from visual import results_analysis as rn

import numpy as np
import time, torch

from  train_lm_img import train_model
import logger_write

from utils import *
from config import config


repeat_time = config.REPEAT_TIME
k_fold_numbers = config.K_FLOD
num_class = config.NUM_CLASSES
epochs = config.EPOCHS
#input_flag = config.INPUT_FLAG
one_hot =config.ONE_HOT_ENCODE
image_dir = config.IMAGE_DIR
save_path = config.SAVE_PATH
lm_dir = config.LM_DIR

cv_acc = []

def label_to_one_hot(label,num_classes):
    """
    转换label到one-hot形式
    :param label:
    :return:
    """
    sam_num = len(label)
    label_new = np.zeros((sam_num,num_classes))
    for i in range(sam_num):
        label_new[i,int(label[i])]=1

    return label_new



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


    if one_hot:
         # 转换label格式为多类标签
         train_y = label_to_one_hot(y_tra[index], num_class)
         test_y = label_to_one_hot(y_tes[index1], num_class)
    else:
        train_y =  y_tra[index]
        test_y = y_tes[index1]
    print('train,test', len(tra_index), len(test_index))


    return train_data_xy_i,train_lm_i,train_y,test_data_xy_i,test_lm_i,test_y

#读取lm_data
img_xy_data, img_label = multi_input(image_dir,lm_dir)
#neu_n1,image1 = data_generator.read_image_path_label(image_dir)
lm_data,lm_label  = read_excel_by_xlrd_neuname(img_xy_data,lm_dir)
#lm_data,lm_label=RFECV_feature_select.rfecv_cv(lm_data,lm_label)

def main():

    p_acc_list = []
    #存放多次重复K_flod的结果
    #重复repeat_time次k_fold cross_validation
    for p in range(1,repeat_time+1):

        #获得所有的训练和验证样本

        img_xy_data, img_label = multi_input(image_dir,lm_dir)
        neu_name, img_label_ = spilt_neuname(img_xy_data, img_label)

        img_xy_data, img_label = np.array(img_xy_data), np.array(img_label)
        neu_name, img_label_= np.array(neu_name), np.array(img_label_)
        #print(img_xy_data[0:10],lm_data[0:10])
        assert img_label.all() == img_label_.all()
        assert lm_label.all() == img_label.all()

        # data = zip(img_xy_data,lm_data,img_label)
        # for i,data_ in enumerate(data):
        #     print(data_)

        # img_xy_data, img_label =img_xy_data[0:80], img_label[0:80]

        #neu_name, img_label_ = data_generator.spilt_neuname(img_xy_data, img_label)

        # neu_name, img_label = neu_name[0:80], img_label[0:80]

        #print(img_label, img_label_)
        #assert img_label.all() == img_label_.all()

        test_loss_sum = 0  # The all loss of 10-fold-cross-validation
        test_acc_sum = 0  # The all acc of 10-fold-cross-validation
        best_test_acc_sum = 0  # It is used to accumulate all the acc for 10 folds
        best_test_acc_fold = 0  # It is used to update the highest acc in 10-fold-cross-validation
        best_test_acc_all = []  # It is used to store the best acc in 10 folds
        total_name =[]
        total_pre =[]# It is used to store all the prediction in 10-fold-cross-validation
        total_img_pre =[]
        total_lm_pre =[]
        total_true = []# It is used to store all the ground of truth in 10-fold-cross-validation

        split = StratifiedKFold(n_splits=k_fold_numbers, random_state=42, shuffle=True)

        save_p_ = save_path
        i = 0
        for tra_index, test_index in split.split(img_xy_data, img_label):

            i=i+1

            logger.info('*' * 25 + '{} time {} fold'.format(p, i) + '*' * 25)
            # Partitioned data set and train : test = 9 : 1
            X_train, LM_train,Y_train, X_test, LM_test,Y_test = get_k_fold_data(tra_index, test_index, img_xy_data,img_label,lm_data)

            logger.info('{} time {} fold-cross-validation--The shape of train sets：{}'.format(p, i, len(X_train)))
            logger.info('{} time {} fold-cross-validation--The shape of test sets：{}\n'.format(p, i, len(X_test)))


            each_fold, train_loss_all, train_acc_all, test_loss_all, test_acc_all, best_test_acc, best_epoch_wts,y_true_,y_pre_,y_img_pre_,y_lm_pre_ = train_model(
                logger, X_train, LM_train,Y_train, X_test, LM_test,Y_test, save_p_,i)

            logger.info('{} time {} fold record:'.format(p, i))
            logger.info(each_fold)
            logger.info('The last epoch: ' + 'train_loss:{:.4f}, train_acc:{:.4f}'.format(train_loss_all[-1],
                                                                                          train_acc_all[-1]))
            logger.info(
                'The last epoch: ' + 'test loss:{:.4f}, test_acc:{:.4f}'.format(test_loss_all[-1], test_acc_all[-1]))
            logger.info('The best epoch: {:.4f}\n'.format(best_test_acc))

            #存储每一次十折交叉验证的分类结果，数值
            test_loss_sum += test_loss_all[-1]
            test_acc_sum += test_acc_all[-1]
            #以数组形式存储
            best_test_acc_all.append(best_test_acc)
            best_test_acc_sum += best_test_acc_all[-1]
            total_true.append(y_true_)
            total_pre.append(y_pre_)
            total_img_pre.append(y_img_pre_)
            total_lm_pre.append(y_lm_pre_)
            
            name_,_= data_generator.spilt_neuname(X_test,Y_test)
            total_name.append(name_)
            #total_name.append(Y_name)
            cv_acc.append(best_test_acc)

            # filename = os.path.join(save_path,input_flag,only_image)
            # if not os.path.exists(filename):
            #     os.makedirs(filename)
            # confusion matrix

            if best_test_acc >= best_test_acc_fold:
                best_test_acc_fold = best_test_acc
                fold_best = i
                best_fold_epoch_wts = best_epoch_wts
        # metric

        # The curves of loss and acc in each fold
        total_pre = np.array([total_pre[i][j] for i in range(len(total_pre)) for j in range(len(total_pre[i]))])
        total_img_pre = np.array([total_img_pre[i][j] for i in range(len(total_img_pre)) for j in range(len(total_img_pre[i]))])
        total_lm_pre = np.array([total_lm_pre[i][j] for i in range(len(total_lm_pre)) for j in range(len(total_lm_pre[i]))])
        
        
        total_true = np.array([total_true[i][j] for i in range(len(total_true)) for j in range(len(total_true[i]))])
        total_name = np.array([total_name[i][j] for i in range(len(total_name)) for j in range(len(total_name[i]))])
        save_results.write_all_acc(cv_acc, os.path.join(save_p_, 'cv_acc' + only_image + '.txt'))

        #可视化十折交叉验证的结果

        # 可视化当前k_flod的分类结果
        # save_ = os.path.join(save_path,input_flag,i_r)
        metric.cm_plot(total_true, total_pre, save_p_)
        metric.class_results(total_true, total_pre, save_p_)
        
        metric.cm_plot(total_true, total_img_pre, save_p_,'img')
        metric.class_results(total_true, total_img_pre, save_p_,'img')
        
        
        # each_fold1 = {}
        # each_fold1['Y_true'] = total_true
        # each_fold1['Y_pred']  = total_name
        # each_fold1['Y_score'] = total_score
        #write_data_excel(each_fold1,write_xls,sheetname=str(fold)+'pred_resullts')
        #visual_all_results(total_true,total_pred,config.SAVE_PATH,test_score=total_score)
        
        #metric.cm_plot(total_true, total_lm_pre, save_p_,'lm')
        #metric.class_results(total_true, total_lm_pre, save_p_,'lm')

        # save_ = os.path.join(save_path,input_flag,i_r)
        err_sm = rn.error_sample(total_pre, total_true, total_name)
        rn.save_err_sample(err_sm, save_p_)
        # save the model

        logger.info('{} time {}-fold-cross-validation--The best fold is {}'.format(p, k_fold_numbers, fold_best))
        logger.info('-' * 10 + '{} fold-cross-validation--acc_avg(The last epoch):'.format(k_fold_numbers) + '-' * 10)
        logger.info('average test loss:{:.4f}, average test accuracy:{:.4f}'.format(test_loss_sum / k_fold_numbers,                                                                                   test_acc_sum / k_fold_numbers))
        logger.info('-' * 10 + '{} fold-cross-validation--acc_avg(The best epoch):'.format(k_fold_numbers) + '-' * 10)
        logger.info('average test accuracy:{:.4f}\n'.format(best_test_acc_sum / k_fold_numbers))

        p_acc = best_test_acc_sum / k_fold_numbers
        p_acc_list.append(p_acc)


    logger.info('-' * 10 + '{} time 10-fold-cross-validation--acc_avg(best epoch)'.format(repeat_time) + '#' * 10)
    logger.info('average test accuracy:{:.4f}'.format(np.mean(p_acc_list)))



if __name__=='__main__':

    logger = logger_write.logger_init()
    device = torch.device('cuda:0')
    logger.info('This experiment is implemented by ' + str(torch.cuda.get_device_name(0)))

    tic = time.time()
    main()
    toc = time.time()
    time_all = toc - tic
    logger.info('Time for the whole experiment： {:.0f}m {:.0f}s\n'.format(time_all // 60, time_all % 60))





