#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/28 10:09
# @Author  : Sun Chunli
# @Software: PyCharm
# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/14 21:56
# @Author  : Sun Chunli
# @Software: PyCharm

import os, pdb
import time, torch, copy, math
import torch.nn as nn
import pandas as pd
from sklearn.svm import LinearSVC, SVC
from nets import ResNet_MLP_double_flow_cross_self as ResNet_MLP
from visual import plot_loss_acc
from visual import metric
import numpy as np
from collections import Counter
from Mydataset import Mydataset_lm_img
from config import config


epochs = config.EPOCHS
designated_initializer = config.DESIGNED_INIT
batch_size = config.BATCH_SIZE
optimizer_ = config.OPTIMIZER
lr = config.LEARNIG_RATE
threshold = config.THRESHOLD
momentum = config.MOMENTUM
top_k= config.TOP_K
p_joint = config.P_JOINT
p_img = config.P_IMG,
p_lm = config.P_LM,
p_joint = float(p_joint)
p_img = float(p_img[0])
p_lm = float(p_lm[0])


import threading

def check_io_blocking():
    for thread in threading.enumerate():
        if thread.getName() == "MainThread":
            continue
        if thread.is_alive():
            # 程序被阻塞 IO 
            print('Debuging')
            raise ValueError

##定义训练过程，loss函数，优化器等
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def initialize_weight_goog(m):
    # 卷积层初始化
    if isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()
    # BN 层初始化
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    # fc 层初始化
    elif isinstance(m, nn.Linear):
        fan_out = m.weight.size(0)  # fan-out
        fan_in = 0
        init_range = 1.0 / math.sqrt(fan_in + fan_out)
        m.weight.data.uniform_(-init_range, init_range)
        m.bias.data.zero_()


def train_model(logger, X_train,LM_X_train, Y_train, X_test,LM_X_test, Y_test, fig_dir, cv_num=0):
    """T
    he process of each 10-fold-cross-validation
    data_sizes: 为样本总量
    num_epochs:为总共迭代的轮数，一个epoch表示把整个训练集走一遍
    scheduler: 用来控制学习率
    dataloaders: 加载数据
    criterion: loss函数
    optimizer: 优化器

    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 转换数据格式
    #print('the shape of lm features',np.shape(LM_X_train))
    train_dataset =  Mydataset_lm_img(X_train, LM_X_train,Y_train)
    test_dataset = Mydataset_lm_img(X_test, LM_X_test,Y_test)

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

    #print(X_train[0],LM_X_train[0],Y_train[0])

    # 存储文件
    fig_dir1 = os.path.join(fig_dir, 'loss_re')
    cm_path = os.path.join(fig_dir, 'Comfusion_m')
    re_path = os.path.join(fig_dir, 'report')
    loss_acc_path = os.path.join(fig_dir, 'loss_acc_fig')
    weight_dir = os.path.join(fig_dir, 'weights model')
    best_model_save = os.path.join(fig_dir, 'best_model')
    log_dir = os.path.join(fig_dir, 'log_train')

    feature_dir = os.path.join(fig_dir, 'feature')
    fig_l = [fig_dir1, cm_path, re_path, loss_acc_path, weight_dir, log_dir, best_model_save, feature_dir]

    for fig_i in fig_l:
        if not os.path.exists(fig_i):
            os.makedirs(fig_i)

    # 打印，模型中定义了哪些层
    # print(model)
    #model = ResNet_MLP_top_k.Resnet_MLP(top_k)
    model = ResNet_MLP.Resnet_MLP(top_k,threshold=threshold)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    # 模型参数设置
    params = model.parameters()
    # 定义优化器
    if optimizer_ == 'adam':
        optimizer = torch.optim.Adam(
            [{"params":filter(lambda p: p.requires_grad, model.parameters())}],
            lr=lr,
            # 过滤掉paramters中requires_grad=Falsed的参数
        )  # 重要的是这一句
    elif optimizer_ == 'sgd':
        optimizer = torch.optim.SGD(
            [{"params":filter(lambda p: p.requires_grad, model.parameters())}],
            lr,
            momentum=momentum,
            weight_decay=1e-4,
        )
    else:
        raise ValueError('Optimizer not supported.')
    # 控制学习率
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    train_loss_all = []
    train_acc_all = []
    test_loss_all = []
    test_acc_all = []
    
    train_img_loss_all = []
    train_img_acc_all = []
    test_img_loss_all = []
    test_img_acc_all = []
    
    train_lm_loss_all = []
    train_lm_acc_all = []
    test_lm_loss_all = []
    test_lm_acc_all = []
    
    
    best_test_lm_acc = 0.0
    best_test_img_acc = 0.0
    best_test_acc = 0.0  # It is used to record the best acc in 10 folds in each time 10-fold-corss-validation
    #best_epoch_wts = copy.deepcopy(model.state_dict())  # It is used to record the corresponding parameters
    y_t_ = []
    y_p_ = []
    
    y_img_p =[]
    y_lm_p = []

    for epoch in range(epochs):

        tic = time.time()
        logger.info('Epoch {}/{}'.format(epoch, epochs - 1))
        logger.info('-' * 30)

        train_loss = 0.0
        train_corrects = 0
        test_loss = 0.0
        test_corrects = 0
        
        train_img_loss = 0.0
        train_img_corrects = 0
        test_img_loss = 0.0
        test_img_corrects = 0
        
        train_lm_loss = 0.0
        train_lm_corrects = 0
        test_lm_loss = 0.0
        test_lm_corrects = 0
        
        
        count_batch = 0
        train_num = 0
        test_num = 0
        logger.info('current epoch {}, lr = {:.4f}'.format(epoch, optimizer.param_groups[0]['lr']))

        # 训练模型
        # scheduler.step()
        for step, (b_x, lm_x, b_y) in enumerate(train_loader):
            #pre_ =[]
            #result每次仅存储当前执行的batcsize数据的预测结果，所有arr[0],arr[1],arr[2]都应该为相同的大小，等于b_y的个数
            count_batch += 1
            # move tensors to GPU if CUDA is available
            # print(b_x[:,0].shape)

            # 转换数据形式为：[b,c,w,w]
            #print(np.shape(b_x[0][:, 0]))
            #print(np.shape(b_x[0]))
            b_x0 = b_x.permute(0, 3, 1, 2)

            # 将数据放到GPU 上
            b_x0 = b_x0.to(device, dtype=torch.float32)
            b_x1 = lm_x.to(device, dtype=torch.float32)
            #print(np.shape(b_x0),np.shape(b_x1),np.shape(b_x2))
            b_y = b_y.to(device, dtype=torch.long)

            if step==0:
                assert Y_train[:batch_size].all()==b_y.cpu().detach().numpy().all()
            else:
                assert Y_train[step*batch_size:(step+1)*batch_size].all()==b_y.cpu().detach().numpy().all()

            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            model.train(True)
            # forward pass: compute predicted outputs by passing inputs to the model
            #print(b_x0.size(),b_x1.size())
            outputs,img_y, lm_y = model(b_x0, b_x1)
            # print(output.shape ,b_y.shape)
            # out = torch.argmax(output, dim=1)#输入的也是对每个类别的概率
            _, preds = torch.max(outputs, 1)
            # print(torch.argmax(outputs.data, 1))

            # print(b_y,output)
            # calculate the batch loss
            loss = criterion(outputs, b_y)
            img_loss = criterion(img_y, b_y)
            lm_loss  = criterion(lm_y, b_y)

            # backward pass: compute gradient of the loss with respect to model parameters
            #loss.backward(retain_graph=True)
            # perform a single optimization step (parameter update)
            #print(np.shape(preds))
            pred = torch.argmax(outputs, dim=1)
            img_pred = torch.argmax(img_y, dim=1)
            lm_pred = torch.argmax(lm_y, dim=1)

            #print(p_joint,p_img,p_lm)
            total_loss = p_joint*loss + p_img*img_loss + p_lm*lm_loss 
            #total_loss = 0.5*loss + 0.3*img_loss + 0.2*lm_loss
            total_loss.backward()

            optimizer.step()


            train_num += b_x0.size(0)

            train_loss += total_loss.item() * b_x0.size(0)
            train_img_loss += img_loss.item() * b_x0.size(0)
            train_lm_loss += lm_loss.item() * b_x0.size(0)
            
            
            train_corrects += torch.sum( pred== b_y.data).to(torch.float32).item()
            train_img_corrects += torch.sum( img_pred== b_y.data).to(torch.float32).item()
            train_lm_corrects += torch.sum( lm_pred== b_y.data).to(torch.float32).item()
            
            train_num += b_x0.size(0)
            """
            
            if count_batch%10==0:
                batch_loss = train_loss/train_num
                batch_acc = train_corrects / train_num
                time_ = time.time() - tic
                print('Training Epoch {} Batch Loss: {:.4f} Acc:{:.4f}% Time: {:.4f}s'.format(
                     epoch, batch_loss, batch_acc, time_
                ))
                begin_time = time.time()
            """
        # 计算平均训练损失和精准度
        epoch_loss = train_loss / train_num
        epoch_acc = train_corrects / train_num
        train_loss_all.append(epoch_loss)
        train_acc_all.append(epoch_acc)
        
        train_img_loss_all.append(train_img_loss / train_num)
        train_lm_loss_all.append(train_lm_loss / train_num)
        
        train_img_acc_all.append(train_img_corrects / train_num)
        train_lm_acc_all.append(train_lm_corrects / train_num)

        #print('Train Loss: {:.4f} Acc: {:.4f}'.format( epoch_loss, epoch_acc))
        # 保存模型
        # torch.save(model,os.path.join(best_model_save,'epoch{}.pkl'.format(epoch)))

        # 验证集的模型，验证模型
        vote_corrects = 0
        test_img_corrects = 0
        test_lm_corrects =0
        
        lm_clf=0
        y_preb_best =[]
        y_t_best = []
        y_score_best = []
        y_p_img = []
        y_p_lm = []
        model.eval()
        
        img_preb_best = []
        lm_preb_best = []
        
        with torch.no_grad():  # 不计算梯度
            iter_cnt = 0
            for step, (b_x,lm_x, b_y) in enumerate(test_loader):
                iter_cnt += 1
                # 转换数据形式为：[b,c,w,w]
                b_x0 = b_x.permute(0, 3, 1, 2)
                b_x1 = lm_x
                b_x0 = b_x0.to(device, dtype=torch.float32)
                b_x1 = b_x1.to(device, dtype=torch.float32)
                b_y = b_y.to(device, dtype=torch.long)



                output, img_y ,lm_y = model(b_x0, b_x1)

                loss = criterion(output, b_y)
                img_loss = criterion(img_y, b_y)
                lm_loss = criterion(lm_y, b_y)

                test_loss += p_joint*loss #+ p_img*img_loss + p_lm*lm_loss
                #0.5* loss+0.3*img_loss + 0.2*lm_loss

                result = torch.argmax(output, dim=1)
                _,score = torch.max(output,1)
                #pre_num = pre_lab.cpu().numpy()
                vote_corrects += torch.sum(result == b_y.data).item()

                test_loss += loss.item() * b_x0.size(0)
                test_img_loss += img_loss.item() * b_x0.size(0)
                test_lm_loss += lm_loss.item() * b_x0.size(0)
                
                
                test_corrects +=  torch.sum(result == b_y.data).item()
                test_img_corrects += torch.sum(torch.argmax(img_y, dim=1) == b_y.data).item()
                test_lm_corrects += torch.sum(torch.argmax(lm_y, dim=1) == b_y.data).item()
                
                test_num += b_x0.size(0)

                y_preb_best.extend(result.cpu().detach().numpy().tolist())
                y_t_best.extend(b_y.cpu().detach().numpy().tolist())
                y_score_best.extend(score.cpu().detach().numpy().tolist())
                #record the accuracy of img,lm
                img_preb_best.extend(torch.argmax(img_y, dim=1).cpu().detach().numpy().tolist())
                lm_preb_best.extend(torch.argmax(lm_y, dim=1).cpu().detach().numpy().tolist())

            #print('epoch: {}, model paramters is {},{},{}'.format(str(epoch), str(W_a), str(W_b), str(W_c)))
            if epoch%20==0:
                print('epoch: {}'.format(str(epoch) ))
                print('boost accuracy:'+str(vote_corrects/test_num))
                print('img accuracy:'+str(test_img_corrects/test_num))
                print('lm accuracy:'+str(test_lm_corrects/test_num))

            
            test_loss_all.append(test_loss.cpu() / test_num)
            test_acc_all.append(test_corrects / test_num)
            
            test_img_loss_all.append(test_img_loss / test_num)
            test_lm_loss_all.append(test_lm_loss / test_num)
            
            test_img_acc_all.append(test_img_corrects / test_num)
            test_lm_acc_all.append(test_lm_corrects / test_num)
            
            
            logger.info(
                 '{} Train Loss: {:.4f}  Train Acc: {:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))
            logger.info('{} Test Loss: {:.4f}  Test Acc: {:.4f}'.format(epoch, test_loss_all[-1], test_acc_all[-1]))
            logger.info('boost accuracy:' + str(vote_corrects / test_num))
            logger.info('img accuracy:' + str(test_img_corrects / test_num))
            logger.info('lm accuracy:' + str(test_lm_corrects / test_num))
           
        # save the best acc and corrosponding parameters in 10 folds
        if test_acc_all[-1] > best_test_acc:
            best_test_acc = test_acc_all[-1]
            best_test_img_acc = test_img_acc_all[-1]
            best_test_lm_acc = test_lm_acc_all[-1]
            
            best_epoch_wts = copy.deepcopy(model.state_dict())
            y_t_ = y_t_best
            y_p_ = y_preb_best
            y_score_ = y_score_best
            y_img_p = img_preb_best
            y_lm_p = lm_preb_best
            
            torch.save(model.state_dict(), os.path.join(best_model_save, '{}_fold_best.pt'.format(cv_num)))

        toc = time.time()
        time_use = toc - tic
        logger.info('current epoch complete in {:.0f}m {:.0f}s\n'.format(time_use // 60, time_use % 60))

        # model.load_state_dict(best_epoch_wts) #模型更新为最优的参数

    # h可视化训练结果和验证结果
    # metric
    # confusion matrix
    title_acc = 'The accuracy of {}-fold cross_val'.format(cv_num)
    title_loss = 'The  loss of {}-fold cross_val'.format(cv_num)
    save_acc_plot = loss_acc_path + '/' + 'acc_' + str(cv_num) + '.png'
    save_loss_plot = loss_acc_path + '/' + 'loss_' + str(cv_num) + '.png'
    #就是这里出了问题
    #print('write done!111')
    # pdb.set_trace()
    #plot_loss_acc.plot_acc_numpy(train_acc_all, test_acc_all, save_acc_plot, title_acc)
    # print('write done!222')
    # plot_loss_acc.plot_loss_numpy(train_loss_all, test_loss_all, save_loss_plot, title_loss)

    logger.info(
         'Cross_validation: {} fold best acc is {:.4f}\n'.format(cv_num, best_test_acc))
    logger.info(
         'Cross_validation: {} fold best img acc is {:.4f}\n'.format(cv_num, best_test_img_acc))
    logger.info(
         'Cross_validation: {} fold best lm acc is {:.4f}\n'.format(cv_num, best_test_lm_acc))
    #
    # # 可视化当前k_flod的分类结果
    # # save_ = os.path.join(save_path,input_flag,i_r)
    # # print(y_t_,y_p_)
    #print('write 111')
    #check_io_blocking()
    # metric.cm_plot(y_t_, y_p_, cm_path)
    # #check_io_blocking()
    metric.class_results(y_t_, y_p_, re_path,y_score=y_score_)

    pd.set_option('display.max_columns', None)
    each_fold = pd.DataFrame(
        data={
            "epoch": range(epochs),
            "train_loss_all": train_loss_all,
            "test_loss_all": test_loss_all,
            "train_acc_all": train_acc_all,
            "test_acc_all": test_acc_all,
            
            "train_img_loss_all": train_img_loss_all,
            "test_img_loss_all": test_img_loss_all,
            "train_img_acc_all": train_img_acc_all,
            "test_img_acc_all": test_img_acc_all,
            
            "train_lm_loss_all": train_lm_loss_all,
            "test_lm_loss_all": test_lm_loss_all,
            "train_lm_acc_all": train_lm_acc_all,
            "test_lm_acc_all": test_lm_acc_all,
        }
    )
    print('write 111')
    each_fold.to_excel(os.path.join(fig_dir1, 'img_lm_boost_'+str(cv_num) + '.xlsx'))

    # # test
    # metric.cm_plot(y_t_, y_p_, cm_path)
    # metric.class_results(y_t_, y_p_, re_path)

    return each_fold, train_loss_all, train_acc_all, test_loss_all, test_acc_all, best_test_acc,\
           best_epoch_wts, y_t_, y_p_, y_img_p, y_lm_p




