#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 16:49:08 2017

@author: ray
"""

import os
import random
import numpy as np
from scipy.misc import *
import scipy.io as sio
from keras.utils.np_utils import to_categorical
from keras.applications.inception_v3 import preprocess_input


path = '/home/ray/dataset/cars-jkrause/cars_train'


def gen_data(train_list,batch_size=8):
    with open(train_list) as f:
        img_list = f.readlines()
        random.shuffle(img_list)
        nb = len(img_list)
        iter_nb = nb / batch_size
        count = 0
        while True:
            if count < iter_nb:
                img_names = img_list[count*batch_size:(count+1)*batch_size]
                count += 1
            elif count == iter_nb:
                img_names = img_list[count*batch_size:]
                count = 0
            real_batch_size = len(img_names)
            train_data = np.zeros((real_batch_size,256,256,3))
            train_label = np.zeros((real_batch_size,))
            for i,j in enumerate(img_names):
                j = j.split()
                x1 = int(j[2])
                y1 = int(j[3])
                x2 = int(j[4])
                y2 = int(j[5])
                img = imread(j[0],mode='RGB')[y1:y2,x1:x2,:]
                train_data[i] = imresize(img,size=(256,256))
                train_label[i] = int(j[1])-1
            yield (train_data.astype('float32'),to_categorical(train_label,196).astype('uint8'))
            
    
def gen_data1(anno,batch_size):
    annotations = sio.loadmat(anno)
    annotations = annotations['annotations'].reshape(-1)
    nb = len(annotations)
    iter_nb = nb / batch_size
    count = 0
    while True:
        if count < iter_nb:
            imgs = annotations[count*batch_size:(count+1)*batch_size]
            count += 1
        elif count == iter_nb:
            imgs = annotations[count*batch_size:]
            count = 0
        real_batch_size = len(imgs)
        train_data = np.zeros((real_batch_size,256,256,3))
        train_label = np.zeros((real_batch_size,))
        for i in range(real_batch_size):
            img_name = str(imgs[i][5][0])
            label = int(imgs[i][4][0][0])-1
            train_data[i] = imresize(imread(os.path.join(path,img_name),mode='RGB'),size=(256,256))
            train_label[i] = label
        yield (train_data.astype('float32'),to_categorical(train_label,196).astype('uint8'))


        
if __name__ == '__main__':
    gen = gen_data('train_list.list')
    #gen = gen_data1('cars_train_annos.mat',8)
    for i in range(1):
        print i
        a = next(gen)
            
                