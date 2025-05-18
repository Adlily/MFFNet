#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/27 21:50
# @Author  : Sun Chunli
# @Software: PyCharm
import matplotlib.pyplot as plt
import pdb
import numpy as np
from pylab import rcParams
import matplotlib.font_manager as font_manager

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14,
}

font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 12,
}


def adujst_fig(ax,title):
    """
    设置图的大小和线型、粗细
    :return:
    """

    # 设置坐标刻度值的大小以及刻度值的字体

    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    ###设置坐标轴的粗细
    ax.spines['bottom'].set_linewidth(2)  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2)  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2)  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)  ####设置上部坐标轴的粗细

    # 刻度线
    ax.tick_params(which='major', width=2,labelsize=12)
    #设置标题
    ax.set_title( title,fontdict=font1)
    #设置tuli
    #ax.legend(prop=font2)




def plot_acc(result,save_path,title):
    """
    result:the return of model.fit
    """

    rcParams["figure.figsize"] = [6.4, 4.8]
    if 'acc' in result.history.keys():

         train_acc= result.history['acc']
         val_acc = result.history['val_acc']
    elif 'accuracy' in result.history.keys():

         train_acc= result.history['accuracy']
         val_acc = result.history['val_accuracy']
    else:
        print('KeyError!')
        return

    x_lim = [x_ for x_ in range(1,len(train_acc)+1)]
    y_lim =[y_ for y_ in range(1,len(val_acc)+1)]

    
    plt.plot(x_lim, train_acc, marker="o", color="r", label="Training")
    plt.plot(y_lim, val_acc, marker='^', color='blue',  label="Validation")

    plt.xlabel('Epochs',font1)
    plt.ylabel('Accuracy',font1)

    ax = plt.gca()
    adujst_fig(ax,title)

    plt.legend(prop=font2)
    #plt.title(title)
    plt.savefig(save_path)
    plt.close()
    # plt.show()

def plot_loss (result,loss_acc_path,title):
    """
    result:the return of model.fit
    """
    rcParams["figure.figsize"] = [6.4, 4.8]

    train_loss = result.history['loss']
    val_loss = result.history['val_loss']

    x_lim = [x_ for x_ in range(1,len(train_loss)+1)]
    y_lim =[y_ for y_ in range(1,len(val_loss)+1)]

    plt.plot(x_lim, train_loss, marker="o", color="r", label="Training")
    plt.plot(y_lim, val_loss, marker='^', color='blue',  label="Validation")

    plt.xlabel('Epochs',font1)
    plt.ylabel('Loss',font1)

    ax = plt.gca()
    adujst_fig(ax,title)

    plt.legend(prop=font2)
    #plt.title(title)
    plt.savefig(loss_acc_path)
    plt.close()


def plot_acc_numpy(train_acc,val_acc, save_path, title):
    """
    result:the np.array with (2,len(size))
    result[0,:] is the accuracy of training
    result[1,:] is the accuracy of validation
    """

    #result = np.array(result)
    #train_acc = result[0,:]*100
    #val_acc = result[1,:]*100

    #fig, ax = plt.subplots(figsize=(6.4, 4.8))
    rcParams["figure.figsize"] = [7.4, 5.6]
    
    # pdb.set_trace()
    x_lim = [x_ for x_ in range(1,len(train_acc)+1)]
    y_lim =[y_ for y_ in range(1,len(val_acc)+1)]
    plt.plot(x_lim, train_acc, marker="o", color="r", label="Training")
    plt.plot(y_lim, val_acc, marker='^', color='blue',  label="Validation")

    plt.xlabel('Epochs',font1)
    plt.ylabel('Accuracy',font1)

    ax = plt.gca()
    adujst_fig(ax,title)

    plt.legend(prop=font2)
    #plt.title(title)
    plt.savefig(save_path)
    plt.close()
    # plt.show()


def plot_loss_numpy(train_loss,val_loss, loss_acc_path, title):
    """
    result:the np.array with (2,len(size))
    result[0,:] is the loss of training
    result[1,:] is the loss of validation
    """
    #result = np.array(result)
    #train_loss = result[0,:]
    #val_loss  = result[1,:]

    rcParams["figure.figsize"] = [6.4, 4.8]

    x_lim = [x_ for x_ in range(1,len(train_loss)+1)]
    y_lim =[y_ for y_ in range(1,len(val_loss)+1)]
    plt.plot(x_lim, train_loss, marker="o", color="r", label="Training")
    plt.plot(y_lim, val_loss, marker='^', color='blue',  label="Validation")

    plt.xlabel('Epochs',font1)
    plt.ylabel('Loss',font1)

    ax = plt.gca()
    adujst_fig(ax,title)

    plt.legend(prop=font2)
    #plt.title(title)
    plt.savefig(loss_acc_path)
    plt.close()

if __name__=='__main__':
    a=[[1,2,3,4],[3,4,5,7]]
    plot_acc_numpy(a,r'C:\Users\USER\Desktop\test\acc2.png','Acc')