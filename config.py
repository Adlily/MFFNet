#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/18 14:45
# @Software: PyCharm


import os
import time


# BASE_DIR = os.path.dirname(os.path.abspath(__file__))



class Config():

    NUM_CLASSES = 12
    BATCH_SIZE = 8
    EPOCHS = 100

    K_FLOD = 10
    FEATURES_NUM = 43
    ONE_HOT_ENCODE = False
    DESIGNED_INIT = False
    DCA_FLAG = False
    THRESHOLD = 0.8

    REPEAT_TIME = 1
    SET_TRAINABLE = True
    
    
    MODEL = 'resnet50'
    NAME = 'se_resnet'
    ACTIVATE = 'relu'


    WIDTH = 256
    HEIGHT = 256

    TOP_K = 256
    P_JOINT = 0.5
    P_IMG = 0.3
    P_LM =  0.2


    OPTIMIZER = 'adam'
    LEARNIG_RATE = 0.0001
    WEIGHT_DECAY = 0.0001
    MOMENTUM = 0.9
    #OPTIMIZER = ReduceLROnPlateau(monitor= 'val_loss',factor=0.75,patience=4,mode='auto',min_lr=LEARNIG_RATE)

    CLASS_NAME = [
        'Chimpanzee_pyramidal',
        'Granule',
        'Human_pyramidal',
        'Medium_spiny',
        'Mouse_ganglion',
        'Mouse_pyramidal',
        'Rat_interneuron_gabaergic',
        'Rat_internuron_nitregic',
        'Rat_pyramidal_hippocampus',
        'Rat_pyramidal_neocortex'
    ]

    CLASS_DICT = {
        'Chimpanzee_pyramidal': 0,
        'Granule': 1,
        'Human_pyramidal': 2,
        'Medium_spiny': 3,
        'Mouse_ganglion': 4,
        'Mouse_pyramidal': 5,
        'Rat_interneuron_gabaergic': 6,
        'Rat_internuron_nitregic': 7,
        'Rat_pyramidal_hippocampus': 8,
        'Rat_pyramidal_neocortex': 9
    }


    LM_DIR = '/data/10_classes_data/LM_feature\10_43.xlsx'
    IMAGE_DIR = '/data/10_classes_data/images'
    SAVE_PATH = '/results/10_classes_data'
    # SAVE_PATH = os.path.join(SAVE_PATH_)


    # save best model
    if not os.path.exists(WEIGHTS_DIR):
        os.makedirs(WEIGHTS_DIR, exist_ok=True)


    # TensorBoard


config = Config()
