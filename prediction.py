#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 09:41:26 2017

@author: ray
"""

from train import creat_model
import cv2
import numpy as np
import scipy.io as sio
import scipy.misc as im
from SSD.ssd_detector import img_detect_vehicle
label_path = '/home/ray/dataset/cars-16185/car_ims_16185/cars_annos.mat'

def preprocess_input(x):
    x = x / 255.
    x = x - 0.5
    x = x * 2.
    return x

img = im.imread('/home/ray/dataset/cars-jkrause/cars_test/00018.jpg',mode='RGB')
recs = img_detect_vehicle(img)
rec_nb = len(recs)
imgs = np.zeros((rec_nb,256,256,3))
for i,rec in enumerate(recs):
    x1 = rec[0]
    y1 = rec[1]
    x2 = rec[2]
    y2 = rec[3]
    imgs[i] = preprocess_input(im.imresize(img[y1:y2,x1:x2,:],size=(256,256)))




model = creat_model()
model.load_weights('model_best_only_new')

pred = model.predict(imgs)
pred = np.argmax(pred,axis=1)


anno = sio.loadmat(label_path)
class_names = anno['class_names'].reshape(-1)
for i in range(pred.shape[0]):
    pred_class = str(class_names[pred[i]][0])
    cv2.rectangle(img,(recs[i][0],recs[i][1]),(recs[i][2],recs[i][3]),(255, 0, 0), 4)
    cv2.putText(img,pred_class,(recs[i][0],recs[i][1]),cv2.FONT_HERSHEY_SIMPLEX,  .6, (0, 255, 0), 1, 2)

im.imshow(img)
