import sys
import os
import numpy as np
from config import config
from PIL import Image
from sklearn.utils import shuffle
import pandas as pd
import xlrd
from sklearn.impute import SimpleImputer
from sklearn import preprocessing

class_dict = config.CLASS_DICT
image_dir_ = config.IMAGE_DIR
input_file_ = config.LM_DIR
fea_num = config.FEATURES_NUM
class_name = config.CLASS_NAME

class_dict = config.CLASS_DICT
path1 = config.IMAGE_DIR
class_name = config.CLASS_NAME
height = config.WIDTH
width = config.WIDTH
n_class = config.NUM_CLASSES


#all_neu_n = lm_data_process.read_neuron_name()

split_format = config.SPLIT_FORMAT
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
            all_data.append( os.path.join(cla_path, i_n))

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
def get_simple_name(image_dir):

    """

    :param image_dir:
    :return:
    """
    name =[]
    label=[]
    i = 0
    for cl in class_name:
        cla_path = os.path.join(image_dir,cl)
        neu_n_ = os.listdir(cla_path)
        neu_n_.sort()
        label_ = class_dict[cl]
        label.extend([label_]*len(neu_n_))

        name.extend(neu_n_)

    all_neu_name =  np.vstack((name,label)).T
    #print(all_neu_name)
    return np.array(all_neu_name)


all_neu_n = get_simple_name(path1)
def spilt_neuname(img_path, img_label):
    """
    split neuron name from dirpath
    """
    zipeed  = zip(img_path,img_label)
    all_name = []
    img_label_ = []
    #print(len(zipeed))
    for neu_path, label_ in zipeed:
        neu_name_ = neu_path.split(split_format)[-2:]
        neu_name = split_format.join(neu_name_)
        #print(neu_name)
        all_name.append(neu_name)
        img_label_.append(label_)
    #print(all_name,'\n',img_label_)
    return np.array(all_name),np.array(img_label_)
    
class Img:
    def __init__(self,image,rows,cols,center=[0,0]):
        self.src=image #原始图像
        self.rows=rows #原始图像的行
        self.cols=cols #原始图像的列
        self.center=center #旋转中心，默认是[0,0]

    def Move(self,delta_x,delta_y):      #平移
        #delta_x>0左移，delta_x<0右移
        #delta_y>0上移，delta_y<0下移
        self.transform=np.array([[1,0,delta_x],[0,1,delta_y],[0,0,1]])

    def Zoom(self,factor):               #缩放
        #factor>1表示缩小；factor<1表示放大
        self.transform=np.array([[factor,0,0],[0,factor,0],[0,0,1]])

    def Horizontal(self):                #水平镜像
        self.transform=np.array([[1,0,0],[0,-1,self.cols-1],[0,0,1]])

    def Vertically(self):                #垂直镜像
        self.transform=np.array([[-1,0,self.rows-1],[0,1,0],[0,0,1]])

    def Rotate(self,beta):               #旋转
        #beta>0表示逆时针旋转；beta<0表示顺时针旋转
        self.transform=np.array([[math.cos(beta),-math.sin(beta),0],
                                 [math.sin(beta), math.cos(beta),0],
                                 [    0,              0,         1]])
                                 
                                 
def image_genertor(image):
    """
    filp,roration
    image：PIL.Image as img 读取得到的图片
    """
    #图片水平翻转
    image = image.transpose(Image.FLIP_LEFT_RIGHT)
    #竖直方向的旋转
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    #旋转
    #image = image.rotate(45)#旋转45度
    image = image.transpose(Image.ROTATE_90)
    image = image.transpose(Image.ROTATE_180)
    image = image.transpose(Image.ROTATE_270)
    
    #改变图片的亮度
    
    return image


def label_to_one_hot(label):
    """
    转换label到one-hot形式
    :param label:
    :return:
    """
    sam_num = len(label)
    label_new = np.zeros((sam_num,n_class))
    for i in range(sam_num):
        label_new[i,int(label[i])]=1

    return label_new

def read_image_path_label(path=path1):
    """
    read projected_images
    """
    print(path)
    all_data = np.zeros((len(all_neu_n),1)).tolist()
    #print(np.shape(all_data))
    label = np.zeros((len(all_neu_n),1)).tolist()

    # xy_data = np.zeros((height,width,3))
    i = 0
    for cl in class_name:
        cla_path = os.path.join(path,cl)
        neu_n_ = os.listdir(cla_path)
        neu_n_.sort()
        for i_n in neu_n_:
            index_ = np.where(all_neu_n[:,0]==i_n)
            #print(index_[0][0])
            all_data [index_[0][0]]=os.path.join(cla_path,i_n)

        #cl_file = glob.glob(path + '/' + cl+'/*' )
        #cl_file.sort()
        # print(cl_file)
        #all_data.append(cl_file)
            label[index_[0][0]]=class_dict[cl]
    #保持样本顺序一致

    #print(all_data,label)
    #all_data1 = np.array([all_data[i][j] for i in range(len(all_data)) for j in range(len(all_data[i]))])
    #label1=np.array([label[i][j] for i in range(len(label)) for j in range(len(label[i]))])
    
    
    #print(np.shape(all_data))
    # a1 = np.array(all_data)
    # print(np.shape(a1[:,0]))
    # print(len(label))
    # pause

    assert np.shape(all_data)[0]==len(label)
    all_data_ , label_ = shuffle(all_data,label,random_state=0)
    #print(all_data[0:10],label[0:10])
    #print(all_data_,label_)
    return np.array(all_data_), np.array(label_)

def generator_img_1(train_img_path,image1='1'):
    #print()

    img_size = len(train_img_path)
    i=0
    for img_file in train_img_path:
        #print(img_file)
        #img_file = train_img_path[i]

        cur_1 = np.zeros([1, height, height, 3])
        if image1 == '1':
            img1 = Image.open(os.path.join(img_file, '1.png'))
            img1 = img1.resize((width, width))
            
            img1 = image_genertor(img1)
            # print(type(img1))
            # img1 = img1.tobytes()

            img1 = np.asarray(img1)
            cur_1[0, 0:height, 0:width, 0:3] = img1

        elif image1 == '2':
            img1 = Image.open(os.path.join(img_file, '2.png'))
            img1 = img1.resize((width, width))
            
            img1 = image_genertor(img1)
            
            # img2 = img2.tobytes()
            img1 = np.asarray(img1)
            cur_1[0, 0:height, 0:width, 0:3] = img1

        elif image1 == '3':
            img1 = Image.open(os.path.join(img_file, '3.png'))
            img1 = img1.resize((width, width))
            img1 = image_genertor(img1)
            
            # img3 = img3.tobytes()
            img1 = np.asarray(img1)

            cur_1[0, 0:height, 0:width, 0:3] = img1
        else:
            print('Error image1 choose!')
            return
        # img_ = [img1,img2,img3]
        # print(np.shape(img_[0]))

        # all_data.append(img_)
        if i == 0:
            xy_data = cur_1
            # print(np.shape(img1))
            #print(i)
            # yz_data = cur_2
            # xz_data = cur_3
        else:
            # print(np.shape(img1))
            #print('&&&&&&&&&',i)
            xy_data = np.append(xy_data, cur_1, axis=0)
            # yz_data = np.append(yz_data, cur_2, axis=0)
            # xz_data = np.append(xz_data, cur_3, axis=0)

        i=i+1


    #print('****',np.shape(xy_data))

        #i

    return np.array(xy_data)

def generator_img_3(train_img_path):

    for i in range(len(train_img_path)):

        img_file = train_img_path[i]
        cur_1 = np.zeros([1, height, height, 3])
        cur_2 = np.zeros([1, height, height, 3])
        cur_3 = np.zeros([1, height, height, 3])

        img1 = Image.open(os.path.join(img_file, '1.png'))
        img1 = img1.resize((width, width))
        # print(type(img1))
        # img1 = img1.tobytes()

        img1 = np.asarray(img1)
        cur_1[0, 0:height, 0:width, 0:3] = img1

        img2 = Image.open(os.path.join(img_file, '2.png'))
        img2 = img2.resize((width, width))
        # img2 = img2.tobytes()
        img2 = np.asarray(img2)
        cur_2[0, 0:height, 0:width, 0:3] = img2

        img3 = Image.open(os.path.join(img_file, '3.png'))
        img3 = img3.resize((width, width))
        # img3 = img3.tobytes()
        img3= np.asarray(img3)

        cur_3[0, 0:height, 0:width, 0:3] = img3

        # img_ = [img1,img2,img3]
        # print(np.shape(img_[0]))

        # all_data.append(img_)
        if i == 0:
            xy_data = cur_1
            # print(np.shape(img1))
            yz_data = cur_2
            xz_data = cur_3
        else:
            # print(np.shape(img1))
            xy_data = np.append(xy_data, cur_1, axis=0)
            yz_data = np.append(yz_data, cur_2, axis=0)
            xz_data = np.append(xz_data, cur_3, axis=0)
    return [np.array(xy_data),np.array(yz_data),np.array(xz_data)]

def generator_img_2(train_img_path,image2='12'):
    data = []
    for i in range(len(train_img_path)):

        img_file = train_img_path[i]

        cur_1 = np.zeros([1, height, height,3])
        cur_2 = np.zeros([1, height, height,3])

        if image2 == '12':
            img1 = Image.open(os.path.join(img_file, '1.png'))
            img1 = img1.resize((width, width))
            # print(type(img1))
            # img1 = img1.tobytes()

            img1 = np.asarray(img1)
            cur_1[0, 0:height, 0:width,0:3] = img1

            img2 = Image.open(os.path.join(img_file, '2.png'))
            img2 = img2.resize((width, width))
            # img2 = img2.tobytes()
            img2 = np.asarray(img2)
            cur_2[0, 0:height, 0:width,0:3] = img2
        elif image2 == '23':

            img1 = Image.open(os.path.join(img_file, '2.png'))
            img1 = img1.resize((width, width))
            # img2 = img2.tobytes()
            img1 = np.asarray(img1)
            cur_1[0, 0:height, 0:width, 0:3] = img1

            img2 = Image.open(os.path.join(img_file, '3.png'))
            img2 = img2.resize((width, width))
            # img3 = img3.tobytes()
            img2 = np.asarray(img2)

            cur_2[0, 0:height, 0:width, 0:3] = img2
        elif image2 == '13':
            img1 = Image.open(os.path.join(img_file, '1.png'))
            img1 = img1.resize((width, width))
            # print(type(img1))
            # img1 = img1.tobytes()

            img1 = np.asarray(img1)
            cur_1[0, 0:height, 0:width, 0:3] = img1

            img2 = Image.open(os.path.join(img_file, '3.png'))
            img2 = img2.resize((width, width))
            # img3 = img3.tobytes()
            img2 = np.asarray(img2)

            cur_2[0, 0:height, 0:width, 0:3] = img2
        else:
            print('Error image2 choose!')
            return
        # img_ = [img1,img2,img3]
        # print(np.shape(img_[0]))
        cur_ = np.stack((cur_1,cur_2),axis=2)
        data.append(cur_)
        # all_data.append(img_)
        if i == 0:
            xy_data = cur_1
            # print(np.shape(img1))
            yz_data = cur_2
            # xz_data = cur_3
            
        else:
            # print(np.shape(img1))
            xy_data = np.append(xy_data, cur_1, axis=0)
            #print(xy_data[0])
            yz_data = np.append(yz_data, cur_2, axis=0)
            # xz_data = np.append(xz_data, cur_3, axis=0)
        # print('****',np.shape(xy_data))
    return [np.array(xy_data), np.array(yz_data)]
    #return np.array(data)

def  myGenerator_1(x_train,y_train,batch_size,image_flag):

     total_size = len(y_train)
     while 1:
         for i in range(total_size//batch_size):
             train_img_path = x_train[i*batch_size:(i+1)*batch_size]

             tra_imgs = generator_img_1(train_img_path,image_flag)

             yield tra_imgs,y_train[i*batch_size:(i+1)*batch_size]


def myGenerator_2(x_train, y_train, batch_size, image_flag):
    total_size = len(y_train)
    while 1:
        for i in range(total_size // batch_size):
            train_img_path = x_train[i * batch_size:(i + 1) * batch_size]

            tra_imgs = generator_img_2(train_img_path, image_flag)

            yield tra_imgs, y_train[i * batch_size:(i + 1) * batch_size]

def myGenerator_3(x_train, y_train, batch_size):
    total_size = len(y_train)
    while 1:
        for i in range(total_size // batch_size):
            train_img_path = x_train[i * batch_size:(i + 1) * batch_size]

            tra_imgs = generator_img_3(train_img_path)

            yield tra_imgs, y_train[i * batch_size:(i + 1) * batch_size]


def myGenerator_1_lm(x_train, lm_train,y_train,batch_size, image_flag):
    total_size = len(y_train)
    while 1:
        for i in range(total_size // batch_size):
            train_img_path = x_train[i * batch_size:(i + 1) * batch_size]

            tra_imgs = generator_img_1(train_img_path, image_flag)
            tra_1 = [tra_imgs, np.array(lm_train[i * batch_size:(i + 1) * batch_size])]
            yield tra_1, y_train[i * batch_size:(i + 1) * batch_size]


def myGenerator_2_lm(x_train, lm_train,y_train, batch_size, image_flag):
    total_size = len(y_train)
    while 1:
        for i in range(total_size // batch_size):
            train_img_path = x_train[i * batch_size:(i + 1) * batch_size]

            tra_imgs = generator_img_2(train_img_path, image_flag)
            tra_2  = [tra_imgs[0],tra_imgs[1],lm_train[i * batch_size:(i + 1) * batch_size]]
            yield tra_2, y_train[i * batch_size:(i + 1) * batch_size]


def myGenerator_3_lm(x_train, lm_train,y_train, batch_size):
    total_size = len(y_train)
    while 1:
        for i in range(total_size // batch_size):
            train_img_path = x_train[i * batch_size:(i + 1) * batch_size]

            tra_imgs = generator_img_3(train_img_path)
            tra_3 = [tra_imgs[0],tra_imgs[1],tra_imgs[2],lm_train[i * batch_size:(i + 1) * batch_size]]
            yield tra_3, y_train[i * batch_size:(i + 1) * batch_size]

def myGenerator_lm(x_train,y_train,batch_size):
    total_size = len(y_train)
    while 1:
        for i in range(total_size // batch_size):
            train_img_path = x_train[i * batch_size:(i + 1) * batch_size]

            yield train_img_path, y_train[i * batch_size:(i + 1) * batch_size]


def myGenerator_1_tf(x_train, y_train, batch_size, i,image_flag='1'):
    total_size = len(y_train)
    if (i +1) * batch_size >= len(x_train):
        #print('***')
        #print(i * batch_size,len(y_train))
        train_img_path = x_train[i * batch_size:len(y_train)]
        #print(train_img_path)
        tra_imgs = generator_img_1(train_img_path, image_flag)

        return tra_imgs, y_train[i * batch_size:len(y_train)]
    else:
        train_img_path = x_train[i * batch_size:(i + 1) * batch_size]

        tra_imgs = generator_img_1(train_img_path, image_flag)

        return tra_imgs, y_train[i * batch_size:(i + 1) * batch_size]



def myGenerator_2_tf(x_train, y_train, batch_size, i,image_flag='12'):
    total_size = len(y_train)
    if (i +1) * batch_size >= len(x_train):
        train_img_path = x_train[i * batch_size:len(y_train)]

        tra_imgs = generator_img_2(train_img_path, image_flag)

        return tra_imgs, y_train[i * batch_size:len(y_train)]
    else:
        train_img_path = x_train[i * batch_size:(i + 1) * batch_size]

        tra_imgs = generator_img_2(train_img_path, image_flag)

        return tra_imgs, y_train[i * batch_size:(i + 1) * batch_size]


def myGenerator_3_tf(x_train, y_train, batch_size,i):
    if (i +1)* batch_size >= len(x_train):
        train_img_path = x_train[i * batch_size:len(y_train)]

        tra_imgs = generator_img_3(train_img_path)

        return tra_imgs, y_train[i * batch_size:len(y_train)]
    else:
        train_img_path = x_train[i * batch_size:(i + 1) * batch_size]

        tra_imgs = generator_img_3(train_img_path)

        return tra_imgs, y_train[i * batch_size:(i + 1) * batch_size]
def multi_input(img_datadir,lm_datadir):
    """
    判断两种输入中都存在的样本作为数据集的输入
    :param img_datadir:
    :param lm_datadir:
    :return:
    """
    common_name = []
    common_label = []

    #读取lm feature cxcel中所有的样本名字：
    wd =pd.read_excel(lm_datadir,usecols=[0],names=None)
    wd = wd.values.tolist()
    swc_name = []
    for s_li in wd:
        swc_name.append(s_li[0])
    #print(swc_name)
    #读取img的文件夹
    for c in class_name:
        class_path = os.path.join(img_datadir,c)
        img_file = os.listdir(class_path)
        for img in img_file:
            neuron_name = img +'.CNG.swc'
            if neuron_name in swc_name:
                common_name.append(os.path.join(class_path,img))
                common_label.append(class_dict[c])
    #print(common_name)
    assert len(common_name) == len(common_label)

    return common_name,common_label


def data_preprocess(data,Norm = 'standardize'):
    """
    对数据中存在的缺失值进行处理，对数据进行归一化
    """
    #Replace missing values 'nan' by column mean
    X=data
    imp = SimpleImputer(missing_values=np.NaN, strategy='mean')
    imp.fit(X)

    # Output replacement "Nan" values
    Y=imp.transform(X)
    #Data Standardization
    if Norm == 'normalize':
        Z=preprocessing.normalize(Y, axis=0, norm='l2') # Normalize
    else:
        if Norm == 'binarize':
            binarizer=preprocessing.Binarizer().fit(Y) # Binarize for Bernoulli
            Z = binarizer.transform(Y)
        else:
            if Norm == 'standardize':
                min_max_scaler = preprocessing.MinMaxScaler() # Normalize the data to [0,1]
                Z=min_max_scaler.fit_transform(Y)
            else:
                Z=preprocessing.scale(Y) #Scaling
    #save preprocessing data to the csv
    return Z

def read_neuron_name(image_dir= image_dir_):
    """

    :param image_dir: swc file
    :image_dir->class_name->neuron_name (dir not file)
    :return:
    """
    all_name = []
    all_label = []
    #class_name = os.listdir(image_dir)
    #print(class_name)
    assert  len(class_name)==len(class_dict)
    for c in class_name:
        class_path = os.path.join(image_dir,c)
        neuron_name = os.listdir(class_path)
        neuron_name.sort()
        neu_label = [class_dict[c]]*len(neuron_name)
        all_name.append(neuron_name)
        all_label.append(neu_label)
    all_name1 = np.array([all_name[j][i] for j in range(len(all_name)) for i in range(len(all_name[j]))])
    all_label1 = np.array([all_label[j][i] for j in range(len(all_label)) for i in range(len(all_label[j]))])

    neu_ni  = np.vstack((all_name1,all_label1)).T

    #print(np.shape(neu_ni))
    return neu_ni

def read_excel_by_xlrd(input_file = input_file_):
    """
    read data from the excel file, return np.array
    """
    #print(input_file)
    data1 = xlrd.open_workbook(input_file)
    table = data1.sheets()[0]  # open the first sheet
    # print(type(table),np.shape(table))
    
    neu_n = read_neuron_name()
    data = np.zeros((len(neu_n),fea_num))
    nrows = table.nrows
    #print('rows',nrows)
    # ncol = table.ncolums
    label = np.zeros((len(neu_n),1))
    for i in range(1,nrows):
        row_ = table.row_values(i)
        if row_ is None:
            return

        #print(row_)
        c_name = row_[0].split('.')[0]
        #print(type(neu_n[i][1]))
        index_ = np.where(neu_n[:,0]==c_name)
        #print(index_[0])
        #print(np.shape(row_[1:-1]))
        data[index_,:] = row_[1:-1]
        label.append(row_[-1])
        """
        if  c_name in neu_n[:,0] :
            #print(c_name)
            #print(type(neu_n[i-1][1]))
            if  int(row_[-1])==int(neu_n[i-1][1]):
                print(row_[-1])
                data.append(row_[1:-2])
        else:
                print(neu_n[i-1,:])
        """
    # print(type(data))
    data = np.array(data)
    #print('the shpe of data is', np.shape(data))
    #归一化
    data1= data_preprocess(data)
    return data1,np.array(label)


def read_excel_by_xlrd_neuname(neu_n,input_file):
    """
    read data from the excel file by the name from the image file, return np.array
    """
    #print(input_file)
    data1 = xlrd.open_workbook(input_file)
    table = data1.sheets()[0]  # open the first sheet
    # print(type(table),np.shape(table))

    data = np.zeros((len(neu_n),fea_num))
    nrows = table.nrows
    #print('rows',nrows)
    # ncol = table.ncolums
    label = np.zeros((len(neu_n),1))
    #for i in range(1,nrows):
    # 读取一整列的数据
    neu_name_excel = [str(table.cell_value(i, 0)) for i in range(1, nrows)]
    #print(np.shape(neu_name_excel))
    p_i = 0
    for i in range(0,len(neu_n)):
        #print(type(neu_n[i]))
        c_neu_name = str(neu_n[i])
        #print(os.path.basename(c_neu_name))
        name_i = os.path.basename(c_neu_name)
        c_name = name_i+'.CNG.swc'#和excel中存储的神经元名称对应起来
        #print(c_name)
        #print(neu_name_excel[0])
        #c_name = name_i + '.swc'
        index_ = neu_name_excel.index(c_name)+1

        #print(index_)
        if  index_ is None:
            continue

        row_ = table.row_values(index_)
        if row_ is None:
            continue
        #print(row_[1:-1])
        row_1 = [float(x) if x!='' else 0 for x in row_[1:-1] ]
        #row_1 = list(map(float,row_[1:-1]))
        data[p_i,:] = row_1
        label[p_i]= int(row_[-1])
        p_i = p_i+1

        """
        if  c_name in neu_n[:,0] :
            #print(c_name)
            #print(type(neu_n[i-1][1]))
            if  int(row_[-1])==int(neu_n[i-1][1]):
                print(row_[-1])
                data.append(row_[1:-2])
        else:
                print(neu_n[i-1,:])
        """
    # print(type(data))
    data = np.array(data)
    #print('the shpe of data is', np.shape(data))
    #归一化
    data1= data_preprocess(data)
    assert len(data1)==len(neu_n)
    return data1,np.array(label)

