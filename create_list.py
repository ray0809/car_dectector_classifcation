#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 16:20:56 2017

@author: ray
"""

import scipy.io as sio
import os



label_path = '/home/ray/dataset/cars-16185/car_ims_16185/cars_annos.mat'
path = '/home/ray/dataset/cars-16185/car_ims_16185/car_ims_16185'


anno = sio.loadmat(label_path)

'''
      dtype=[('relative_im_path', 'O'), 
      ('bbox_x1', 'O'), ('bbox_y1', 'O'), 
      ('bbox_x2', 'O'), ('bbox_y2', 'O'), 
      ('class', 'O'), ('test', 'O')])
'''


annotations = anno['annotations'].reshape(-1)
class_names = anno['class_names'].reshape(-1)

train_txt = open('train_list.list','a')
test_txt = open('test_list.list','a')
for i in range(len(annotations)):
    a = annotations[i]
    name = str(a[0][0])
    label = str(a[5][0][0])
    train = int(a[6][0][0])
    x1 = str(a[1][0][0])
    y1 = str(a[2][0][0])
    x2 = str(a[3][0][0])
    y2 = str(a[4][0][0])
    
    if train == 0:
        train_txt.write(os.path.join(path,name)+' '+label+' '+x1+' '+y1+' '+x2+' '+y2)
        train_txt.write('\n')
    elif train == 1:
        test_txt.write(os.path.join(path,name)+' '+label+' '+x1+' '+y1+' '+x2+' '+y2)
        test_txt.write('\n')

train_txt.close()
test_txt.close()
