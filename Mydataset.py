#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/14 22:14
# @Author  : Sun Chunli
# @Software: PyCharm
from torch.utils.data import DataLoader,Dataset
import read_data
from torchvision import transforms
width = 224
height = 224
#重写数据集的定义
def read_image(path,y_label):
    """
    path :str:the path of image
    """
    all_data = []
    label = []
    # xy_data = np.zeros((height,width,3))
    i = 0
    #for cl in range(len(path)):


    img_file =path

    cur_1 = np.zeros([ height, height, 3])
    samples = os.listdir(img_file)
    samples =  [str(file_name, encoding="utf-8") if not (isinstance(file_name, str)) else file_name for file_name in samples]

    img1 = Image.open(os.path.join(img_file, samples[0]))
    img1 = img1.resize((width, width))
    # print(type(img1))
    # img1 = img1.tobytes()

    img1 = np.asarray(img1).reshape((width,width,-1))
    cur_1[ 0:height, 0:width, 0] = img1[:,:,0]

    img2 = Image.open(os.path.join(img_file, samples[1]))
    img2 = img2.resize((width, width))
        # img2 = img2.tobytes()
    img2 = np.asarray(img2).reshape((width,width,-1))
    cur_1[0:height, 0:width, 1] = img2[:,:,0]


    img3 = Image.open(os.path.join(img_file, samples[2]))
    img3 = img3.resize((width, width))
        # img3 = img3.tobytes()
    img3 = np.asarray(np.array(img3)).reshape((width,width,-1))
    cur_1[0:height, 0:width, 2] = img3[:, :, 0]

    return np.array(cur_1), y_label
#重写数据集的定义
class Mydataset_lm_img(Dataset):

    def __init__(self,root_dir,lm_data,label,train_val='train',transform =None):
        """

        :param root_dir: 数据存放地址， data->class->sample->multi-images(.png or .tif)
        :param train_val:'train' or 'val'
        :param transform: 数据变化操作
        """

        self.data_path = root_dir
        #print(self.data_path)
        self.transform = transform
        self.label = label
        self.lm_data = lm_data

    def __getitem__(self, item):
       #item:索引值
       import numpy as np
       #print(self.data_path[item])
       #print('__________')
       image,label = read_data.read_image(self.data_path[item],self.label[item])
       lm_ = self.lm_data[item,:]
       #lm_ = np.reshape(lm_,(-1,np.shape(lm_)[0]))
       # print(label)
       # print(np.shape(image))

       if self.transform is not None:
           try:
               image = []
               for i in range(image):
                   img = image[i]
                   image_ = self.transform[self.train_val](img)
                   image.extend(image_)
           except:
               print('can not load image:{}'.format(self.data_path[item]))
       #此时的image为列表形式
       #print('the image shape',np.shape(image))
       #print('the lm shape',np.shape(lm_))
       return image,lm_, label


    def __len__(self):
        return len(self.data_path)

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomCrop(width),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        # transforms.Scale(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
