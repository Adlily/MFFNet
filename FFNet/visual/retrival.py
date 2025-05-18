import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
import torch
import argparse
import os
import numpy as np
import time
import argparse
import random
import h5py
from torch.autograd import Variable
from collections import OrderedDict
import matplotlib.pyplot as plt
from sklearn import manifold,datasets
from torch.utils.data import DataLoader
from torch import nn
import h5py
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from mpl_toolkits.mplot3d import Axes3D
import Mydataset #数据集
import neurom as nm
from neurom import viewer
from neurom.view import matplotlib_impl, matplotlib_utils
from neurom.core.types import NeuriteType
#网络模型
from random import shuffle

from config import config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

best_fold = 1
weight_path = '/data/suncl/results/lm_img/10/cross_self_cat/lP4_cat11/best_model/{}_fold_best.pt'.format(best_fold)
save_path = config.SAVE_PATH
data_path = config.IMAGE_DIR
swc_path = config.IMAGE_ROOT_DIR

neuron_class = 'Granule' #针对某一个神经样本
swc_name = '227-1-3'

config.SAVE_PATH
class_dcit = config.CLASS_DICT

os.makedirs(save_path,exist_ok=True)

from nets import ResNet_MLP_double_flow_cross_self as ResNet_MLP
model = ResNet_MLP.Resnet_MLP(10,threshold=0.8)
model.load_state_dict(torch.load(weight_path),strict = False)


query_smaple = neuron_class+'/'+swc_name


def ReadH5py(dir,normalization=True):
    f = h5py.File(dir,'r')
    # print(f.keys())
    data = f['data'][:]
    if normalization:
        data = Normalization(data)
    label = f['label'][:]
    f.close()
    return data,label

def Normalization(data):
    data_normalized = np.zeros(data.shape)
    for i in range(0,data.shape[0]):
        temp = data[i]
        origin = np.zeros((1,3))
        origin[0][0] = (temp[:,0].max() + temp[:,0].min()) / 2
        origin[0][1] = (temp[:,1].max() + temp[:,1].min()) / 2
        origin[0][2] = (temp[:,2].max() + temp[:,2].min()) / 2
        temp[:,0] = temp[:,0] - origin[0][0]
        temp[:,1] = temp[:,1] - origin[0][1]
        temp[:,2] = temp[:,2] - origin[0][2]
        if temp[:,0].max()>1:
            temp[:,0] = temp[:,0] / temp[:,0].max()
        if temp[:, 1].max() > 1:
            temp[:,1] = temp[:,1] / temp[:,1].max()
        if temp[:, 2].max() > 1:
            temp[:,2] = temp[:,2] / temp[:,2].max()
        data_normalized[i] = temp
    return data_normalized

def WriteH5py(dir,data,label):
    f = h5py.File(dir,'w')
    f['data'] = data
    f['label'] = label


def ReadSWC(dir,thresold,CLIP=False,Padding=True):
    data = []
    with open(dir,'r') as f:
        lines = f.readlines()
        for i,line in enumerate(lines):
            if line[0] == '#'or line[0] == '\n':
                continue
            _,_,x,y,z,_,_ = [float(i) for i in line.split()]
            data.append([x,y,z])
        f.close()
        if Padding:
            while len(data)<thresold:data.append([0,0,0])
        if CLIP:
            length = math.floor(len(data) / thresold)
            data = np.array(data[0:(length*thresold)])
        else:data = np.array(data)
    return data

def GenerateH5py(dir_list,thresold):
    label10 = {'pyramidal':0,'aspiny':1,'cholinergic':2,'ganglion':3,'basket':4,'fast-spiking':5,'sensory':6,'neurogliaform':7,'martinotti':8,'mitral':9}
    label7 = {'amacrine':0,'aspiny':1,'basket':2,'bipolar':3,'pyramidal':4,'spiny':5,'stellate':6}
    i = 0
    for filename in os.listdir(dir_list):
        if filename.split('.')[-1] != 'swc':continue
        print(dir_list.split('/')[-1],'/',filename,' ',i)
        if ReadSWC(dir_list+'/'+filename,thresold).shape[0]<thresold:
            continue
        if i == 0:
            datas = ReadSWC(dir_list+'/'+filename,thresold)
            i = i + 1
            continue
        datas = np.concatenate((datas,ReadSWC(dir_list+'/'+filename,thresold)))
        i = i + 1
    if i==0:
        return 'continue','continue'
    datas = datas.reshape(-1,thresold,3)
    labels = np.ones((datas.shape[0], 1))*int(label7[dir_list.split('/')[-1]])
    return datas,labels

def GenerateNeuronDataset(neuron_list,thresold,proportion):
    for i,neuron_type in enumerate(os.listdir(neuron_list)):
        if i == 0:
            datas,labels = GenerateH5py(neuron_list+'/'+neuron_type,thresold)
            continue
        data,label = GenerateH5py(neuron_list+'/'+neuron_type,thresold)
        if data == 'continue':
            continue
        datas = np.concatenate((datas,data))
        labels = np.concatenate((labels,label))
    print(datas.shape,' ',labels.shape)
    state = np.random.get_state()
    np.random.shuffle(datas)
    np.random.set_state(state)
    np.random.shuffle(labels)
    datas = datas.astype(np.float32)
    labels = labels.astype(np.uint8)
    WriteH5py(dir = r'./TrainDatasets_6000.h5',data=datas[0:math.ceil(proportion*datas.shape[0])],
              label=labels[0:math.ceil(proportion*labels.shape[0])])
    WriteH5py(dir=r'./TestDatasets_6000.h5', data=datas[math.ceil(proportion * datas.shape[0]):-1],
              label=labels[math.ceil(proportion * labels.shape[0]):-1])


def VisualizeH5py(dir):
    datas,labels = ReadH5py(dir,normalization=False)
    datas2,labels2 = ReadH5py(dir,normalization=True)
    fig = plt.figure(dpi=180)
    ax1 = fig.add_subplot(121,projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    for i in range(0,200):
        data = datas[i]
        data2 = datas2[i]
        ax1.cla()
        ax1.scatter(data[:,0],data[:,1],data[:,2],c="b", marker=".", s=15, linewidths=0, alpha=1, cmap="spectral")
        ax2.cla()
        ax2.scatter(data2[:, 0], data2[:, 1], data2[:, 2], c="r", marker=".", s=15, linewidths=0, alpha=1,cmap="spectral")
        plt.pause(0.5)

def read_only_image_path_label(path):
    """
    read projected_images
    """
    # print(len(all_neu_n))
    all_data = []
    # print(np.shape(all_data))
    label = []
    # xy_data = np.zeros((height,width,3))
    i = 0
    for cl in class_name:
        cla_path = os.path.join(path, cl)
        neu_n_ = os.listdir(cla_path)
        neu_n_.sort()
        for ij,i_n in enumerate(neu_n_):
            #print(ij)
            # print(index_[0][0])
            all_data.append(os.path.join(cla_path, i_n))

            # cl_file = glob.glob(path + '/' + cl+'/*' )
            # cl_file.sort()
            # print(cl_file)
            # all_data.append(cl_file)
            label.append(class_dict[cl])
    # 保持样本顺序一致

    # print(all_data,label)
    # all_data1 = np.array([all_data[i][j] for i in range(len(all_data)) for j in range(len(all_data[i]))])
    # label1=np.array([label[i][j] for i in range(len(label)) for j in range(len(label[i]))])

    # print(np.shape(all_data))
    # a1 = np.array(all_data)
    # print(np.shape(a1[:,0]))
    # print(len(label))
    # pause

    assert np.shape(all_data)[0] == len(label)
    all_data_, label_ = shuffle(all_data, label, random_state=0)
    # print(all_data[0:10],label[0:10])
    # print(all_data_,label_)
    return np.array(all_data_), np.array(label_)


def ExtractFeature():
    #train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=8,
    #                          batch_size=args.batch_size, shuffle=True, drop_last=True)
    #test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
    #                        batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    #
    img_xy_data, img_label = read_only_image_path_label(data_path)
        
    train_dataset = Mydataset.Mydataset(img_xy_data, img_label)

    train_loader = DataLoader(train_dataset,num_workers=0,
                              batch_size=config.BATCH_SIZE, shuffle=True, drop_last=False)

    model.eval()
    NeuronVectorDatabase = {}

    time_str =  time.strftime('%M_%S')
    write_xls = os.path.join(config.SAVE_PATH,'train_test_curve'+time_str+'.xlsx')
    #提取特征,所有数据进行提取特征
    with torch.no_grad():
            for data, label,swc in train_loader:
                data, label = data.float(), label.squeeze()
                #print(data.size())
                _,feature= model(data,Tsne=True)
                feature = feature.numpy().squeeze()
                #此时的feature,swc均已batch_sizexianshi 
                for f,swcname in zip(feature,swc):
                    neuron_type = "/".join(swcname.split('/')[-2:])
                    NeuronVectorDatabase[neuron_type] = f

    np.save(os.path.join(save_path,'database.npy'), NeuronVectorDatabase)

def CaculateEucliDist(a,b):
    return np.sqrt(sum(np.power((a - b), 2)))

def CaculateCosineSimilarity(a,b):
    num = a.dot(b.T)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return num / denom

def CaculateManhattanDist(a,b):
    return sum(abs(x - y) for x, y in zip(a, b))

def ReturnKey(dict,value):
    '''input value and return key'''
    for k, v in dict.items():
        if all(v==value):
            return k

def CaculateAcc(query,result):
    query_type = query[0:query.find('/')]
    result_type = []
    for swc in result:
        result_type.append(swc[0:swc.find('/')])
    return result_type.count(query_type)/len(result_type),query_type

def QueryTest(database,query_swc,topk=10):
   
    '''{'swc_path':vector}'''
    database = np.load(database, allow_pickle=True).item()
    random.seed(3)
    #query_swc = random.sample(database.keys(), len(database))[i]
    #query_swc = 'spiny/Scnn1a-Tg3-Cre-Ai14-475124505.CNG.swc'
    #print(query_swc)
    query_vec = database[query_swc]
    del database[query_swc]
    value = [value for value in database.values()]

    dist = [CaculateCosineSimilarity(query_vec, val) for val in value]
    dist_value = dict(zip(dist, value))
    dist_value = sorted(dist_value.items(), key=lambda x: x[0], reverse=True)  # 按dist排序

    result = []
    for i in range(0, topk):
        result.append(ReturnKey(database, dist_value[i][-1]))
        print(result[i],' ',dist_value[i][0])
    print(CaculateAcc(query_swc, result))
    VisualizeQueryResult(query_swc,result,top_k=topk)    
    return CaculateAcc(query_swc, result)

def QueryTests(npy):
    acc = []
    class_acc = {}
    sum_acc = 0
    for i in range(0, len(np.load(npy, allow_pickle=True).item())): #一次查询
        print(i)
        instance_acc,type = QueryTest(npy,i)
        acc.append(instance_acc)
        if type not in class_acc:
            temp = []
            temp.append(instance_acc)
            class_acc[type] = temp
        else: class_acc[type].append(instance_acc)
    for ty in class_acc.keys():
        sum_acc = sum_acc + sum(class_acc[ty])/len(class_acc[ty])
    print('acc: ',sum(acc) / len(acc))
    print('avg acc: ',sum_acc/len(class_acc))

def ReturnLabel(database):
    label_list = []
    label10 = {'pyramidal': 0, 'aspiny': 1, 'cholinergic': 2, 'ganglion': 3, 'basket': 4, 'fast-spiking': 5, 'sensory': 6,
             'neurogliaform': 7, 'martinotti': 8, 'mitral': 9}
    label7 = config.CLASS_DICT #{'amacrine': 0, 'aspiny': 1, 'basket': 2, 'bipolar': 3, 'pyramidal': 4, 'spiny': 5, 'stellate': 6}
    for swc in database.keys():
        label_list.append(label7[swc[0:swc.find('/')]])
    return np.array(label_list)


def Tsne(database):
    label7 = config.CLASS_NAME#['Amacrine', 'Aspiny', 'Basket', 'Bipolar', 'Pyramidal', 'Spiny', 'Stellate']
    database = np.load(database, allow_pickle=True).item()
    feature = [value for value in database.values()]
    feature = np.array(feature)
    label = ReturnLabel(database)
    print(feature.shape)
    print(label.shape)
    tsne = manifold.TSNE(n_components=2,init='pca',random_state=13)
    X_tsne = tsne.fit_transform(feature)

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(6, 6))
    for i in range(X_norm.shape[0]):
        plt.scatter(X_norm[i, 0], X_norm[i, 1], color=plt.cm.Set2(label[i]),label = label7[label[i]])

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),loc='upper right',shadow=False,frameon=False,handletextpad=0.2,fontsize=10) #fontsize=14

    plt.title('Feature',size=20)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.show()

def VisualizeQueryResult(query,dir_list,top_k=10):
    #这里绘制的时候使用neuronm库包

    import matplotlib.pyplot as plt  
  
    # 创建11个神经元  
    # 定义布局  
    #print(top_k+1)
    import nibabel as nib
    from neuromorph import nmh

    # 读取SWC文件
    swc_file = os.path.join(swc_path,query+'.CNG.swc')
    swc_img = nmh.load_morphology(swc_file)

    # 提取SWC文件中的数据
    
    ax = plt.subplot()
    # 可视化神经元形态
    nmh.plot_morphology(swc_img, ax=ax)
    plt.show()

    # query_data = nm.load_morphology(os.path.join(swc_path,query+'.CNG.swc'))
  
    # # 创建图形对象和子图对象  
    # fig = plt.figure(figsize=(4,4))
    
    # # 使用plot_morph函数绘制神经元形态，不绘制胞体  
    # ax1 = fig.add_subplot(161)
    # ax1.axis('off')
    # matplotlib_impl.plot_morph(query_data, ax =ax1, soma_outline=False, linewidth=2)  
    
    # # 显示图形  
    # plt.show()
    # 创建图形对象和子图对象  
    # fig, axs = plt.subplots(nrows=2, ncols=top_k+1, figsize=(4,4))
    # ax = axs[0][0]
    # # 设置子图的位置和大小  
    # #ax.set_position([0, 0, 10, 10])  # [left, bottom, width, height]  
    # ax.axis('off')  # 关闭坐标轴 
    # matplotlib_impl.plot_morph(query_data,soma_outline=False,linewidth=2,color='blue')    

    # for i in range(top_k):
    #     data = nm.load_morphology(os.path.join(swc_path,dir_list[i]+'.CNG.swc'))
    #     ax = axs[0][i+1]
    #     ax.axis('off')
    #     #可视化单个样本
    #     matplotlib_impl.plot_morph(data,soma_outline=False,linewidth=2)
    # plt.show()

def ReturnClassFeature(npy,type):
    database = np.load(npy, allow_pickle=True).item()
    result = {}
    for key in database.keys():
        if key[0:key.find('/')] == type:
            result[key] = database[key]
    return result


if __name__ == '__main__':



    ExtractFeature()   
    QueryTest(os.path.join(save_path,'database.npy'), query_smaple,topk=5)
    
    #Tsne(os.path.join(save_path,'database.npy'))