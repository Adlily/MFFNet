import sys
sys.path.insert(1,'/ghome/suncl/code/package')
import numpy as np
from sklearn.metrics import roc_curve,auc,confusion_matrix,classification_report,roc_auc_score,precision_score,recall_score,precision_recall_fscore_support,precision_recall_curve
import matplotlib.pyplot as plt #导入作图库
import os
import sys
import itertools
#from tensorflow import tensor as T
from config import config
import time
classes = config.CLASS_NAME

#classes = ['0','1','2']
# cm_plot.py 文件,包括了混淆矩阵可视化函数,
# 放置在python的site-packages 目录,供调用
# 例如:~/anaconda2/lib/python2.7/site-packages
#-*- coding: utf-8 -*-

class Logger(object):
    def __init__(self,fileN ="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN,"a")
 
    def write(self,message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass

def cm_plot_ratio(y, yp,save_p):
  '''
  plt the conmusion_contrix
  '''
  #y_true = T.as_tensor_variable(yp)
  # labels_pred = model.predict_classes(data,verbose=0)

  # Compute confusion matrix
  cnf_matrix = confusion_matrix(y, yp)
  #cnf_matrix = np.array([[254,99,2,17,128],[12,370,1,2,115],[12,28,12,0,43],[30,49,0,84,45],[55,118,10,1,315]])
  #np.set_printoptions(precision=2)
  cm_normalize = (cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis])
  plt.figure()
  plt.imshow(cm_normalize, interpolation='nearest', cmap=plt.cm.Greys)
  plt.title('Confusion Matrix')
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=25)
  plt.yticks(tick_marks, classes)
  thresh = cnf_matrix.max() / 3

  for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
      ratuo_ = cnf_matrix[i, j]/sum(cnf_matrix[i,:])*100
      plt.text(j, i, format(ratuo_, ".2f"),
               horizontalalignment="center",
               color="white" if i==j else "black")

  plt.ylabel('True Label')
  plt.xlabel('Predicted Label')
  #plt.tight_layout()

  # plt.savefig(fig_dir + 'Normalized_confusion_matrix.svg')
  time1 = time.localtime(time.time())
  str1 = str(time1.tm_mon) + '_' + str(time1.tm_mday) + '_' + str(time1.tm_hour) + '_' + str(time1.tm_min) + '_' + str(
      time1.tm_sec)
  fig = plt.gcf()
  fig.savefig(os.path.join(save_p,"{}_confusion_matrix_ratio.png".format(str1)))
  plt.close()

def cm_plot(y,yp,save_p):
    
    
    #labels_pred = model.predict_classes(data,verbose=0)
    
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y,yp)
    np.set_printoptions(precision=2)
    cm_normalize = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    plt.figure()
    plt.imshow(cm_normalize, interpolation='nearest', cmap=plt.cm.Greys)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=25)
    plt.yticks(tick_marks, classes)
    thresh = cnf_matrix.max() / 2
    
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], "d"),
                 horizontalalignment="center",
                 color="white" if  i==j  else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    #plt.tight_layout()
    
    #plt.savefig(fig_dir + 'Normalized_confusion_matrix.svg')
    time1 = time.localtime(time.time())
    str1 = str(time1.tm_mon)+'_'+str(time1.tm_mday)+'_'+str(time1.tm_hour)+'_'+str(time1.tm_min)+'_'+str(time1.tm_sec)
    fig = plt.gcf()
    fig.savefig(os.path.join(save_p,"{}_confusion_matrix.png".format(str1)))
    plt.close()
    cm_plot_ratio(y,yp,save_p)
    #plt.show()

def compute_sen(Y_test,Y_pred):
    sen = []
    concat = confusion_matrix(Y_test,Y_pred)
    n = len(set(Y_test))
    for i in range(n):
        tp = concat[i][i]
        fn = np.sum(concat[i:]) - tp
        sen1 = tp /(tp+fn)
        sen.append(sen1)
    avg_sen = sum(sen)/n
    return sen, avg_sen


def compute_spe(Y_test,Y_pred):
    spe= []
    concat =  confusion_matrix(Y_test,Y_pred)
    n = len(set(Y_test))
    for i in range(n):
        number = np.sum(concat[:,:])
        tp = concat[i][i]
        fn = np.sum(concat[i,:]) - tp
        fp = np.sum(concat[:,i]) - tp
        tn =  number - tp -fn - fp
        spe1 = tn / (tn+fp)
        spe.append(spe1)
    avg_spe  = sum(spe)/n
    return spe,avg_spe

def class_results(y_true,y_preb,save_p,title=None,y_score=None):
    """
    show the classfication results for the per class
    avg_loss_acc:[avg_tra_loss,acg_tra_acc,avg_test_loss,avg_test_acc]
    """
    time1 = time.localtime(time.time())
    str1 = str(time1.tm_mon)+'_'+str(time1.tm_mday)+'_'+str(time1.tm_hour)+'_'+str(time1.tm_min)+'_'+str(time1.tm_sec)
    results = classification_report(y_true, y_preb,digits=6)
    
    if title is not None:
        f_name = os.path.join(save_p,"{}_class_result_{}.txt".format(str1,title))
    else:
        f_name = os.path.join(save_p,"{}_class_result_.txt".format(str1))

    with open(f_name,"w") as f:
        f.write(results+"\n")
        #write confusion matrix to file
        cnf_matrix = np.array(confusion_matrix(y_true, y_preb))
        cm_normalize = np.array(cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]*100)

        f.write('confusiom matrix ratio is:\n')
        np.savetxt(f, np.reshape(np.array(classes), (1, len(classes))), fmt='%s', delimiter='\t')
        f.write('\n')

        np.savetxt(f, cm_normalize, fmt='%.2f', delimiter='\t')
        f.write('\n')
        f.write('confusiom matrix is:\n')
        f.write('confusiom matrix ratio is:\n')
        np.savetxt(f, cnf_matrix, fmt='%d', delimiter='\t')
        f.write('\n')
        
        #计算敏感度和特异性
        senti,avg_sen = compute_sen(y_true,y_preb)
        specity, avg_spec = compute_spe(y_true,y_preb)

        f.write('avg sensitity is:{}, sensitity for per class is \n'.format(avg_sen))
        np.savetxt(f,senti,fmt='%.6f',delimiter='   ')
        f.write('\n')
        f.write('avg spectity is:{}, spectity for per class is \n'.format(avg_spec))
        np.savetxt(f,specity,fmt='%.6f',delimiter='   ')
        f.write('\n')

        #计算ROC
        if y_score is not None:
            fpr,tpr,thresholds =  roc_curve(y_true,y_score,pos_label=2)
            roc = auc(fpr,tpr)
            f.write('auc is {}\n'.format(roc))

            f.write('fpr is \n')
            np.savetxt(f,np.array(fpr),fmt='%.6f',delimiter='   ')
            f.write('\n')
            f.write('tpr is \n')
            np.savetxt(f,np.array(tpr),fmt='%.6f',delimiter='   ')
            f.write('\n')
            f.write('threshold is \n')
            np.savetxt(f,np.array(thresholds),fmt='%.6f',delimiter='   ')
            f.write('\n')
            

    f.close()              


if __name__ == '__main__':
    pt = [1,1,2,3]
    pp = [1,2,1,3]

    class_results(pt,pp,r'G:\code\code\gvcnn_torch\results\10_select_randomly\comparison_methods\NeuroPath2Path1')

