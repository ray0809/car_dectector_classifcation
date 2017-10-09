#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 17:19:13 2017

@author: ray
"""


from keras.layers import Input,GlobalAveragePooling2D,Dense,Dropout
from keras.applications import InceptionV3,ResNet50

from keras.models import Model
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.optimizers import SGD
from generator import gen_data



def creat_model():
    inputs = Input(shape=(256,256,3))
    base_model = InceptionV3(include_top=False)
    conv = base_model(inputs)
    GAV = GlobalAveragePooling2D()(conv)
    outputs = Dense(196,activation='softmax')(GAV)
    model = Model(inputs,outputs)
    sgd = SGD(lr=0.0001,momentum=0.9,decay=1e-10)
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model




if __name__ == '__main__':
    model = creat_model()
    
    '''
    gen_train = gen_data('train_list.list',16)
    gen_val = gen_data('test_list.list',8)
    earlystopping = EarlyStopping()
    modelchenkpoint = ModelCheckpoint('model_best_only',save_best_only=True)
    model.load_weights('model_best_only')
    model.fit_generator(gen_train,
                        samples_per_epoch=64,
                        nb_epoch=1000,
                        callbacks=[modelchenkpoint],
                        validation_data=gen_val,
                        nb_val_samples=8)
    '''