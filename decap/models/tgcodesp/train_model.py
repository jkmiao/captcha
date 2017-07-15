#!/usr/bin/env python
# coding: utf-8

from keras.models import Model, Input, load_model 
from keras.layers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
import numpy as np
from PIL import Image
import random
import os
from utils import CharacterTable

max_len = 6
ctable = CharacterTable()

def load_data(path='img/train/origin_nacao', width=200, height=50, time_step=21):
    fnames = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith('jpg')]
    random.shuffle(fnames)
    data = np.zeros((len(fnames), width, height, 3))  # 数据类型
    input_len = np.ones(len(fnames), dtype=np.uint8) * time_step         # 23-2 reshape时的维度
    input_label = np.zeros((len(fnames), max_len), dtype=np.uint8)
    label_len = np.zeros(len(fnames), dtype=np.uint8)
    oriLabel = []
    for idx, fname in enumerate(fnames):
        img = Image.open(fname).convert('RGB').resize((width, height), Image.ANTIALIAS)
        data[idx] = np.array(img).transpose(1, 0, 2)
        imgLabel = (fname.split('/')[-1].split('_')[0])
        input_label[idx] = ctable.encode(imgLabel.lower())
        label_len[idx] = len(imgLabel)
        oriLabel.append(imgLabel)
      
    return [data, input_label, input_len, label_len], oriLabel


def ctc_lambda_func(args):
    y_pred, labels, input_len, label_len = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_len, label_len)


def build_model(width, height, max_len, n_class):
    input_tensor = Input((width, height, 3), name='input_tensor')
    x = input_tensor
    for i in range(3):
        x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2,2))(x)
    
    x = BatchNormalization()(x)
    conv_shape = x.get_shape()
    print '===' * 20
    print 'conv_shape', conv_shape
    print '===' * 20
    x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2]*conv_shape[3])))(x)
    x = Bidirectional(GRU(128, return_sequences=True), name='BiGRU1', merge_mode='sum')(x)
    x = Bidirectional(GRU(128, return_sequences=True), name='BiGRU2', merge_mode='concat')(x)
    x = Dropout(0.25)(x)
    x = Dense(n_class, activation='softmax')(x)
    base_model = Model(inputs=input_tensor, outputs=x)
    
    labels = Input(name='the_labels', shape=[max_len], dtype='float32')
    input_len = Input(name='input_len', shape=[1], dtype='int64')
    label_len = Input(name='label_len', shape=[1], dtype='int64')
    
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc_loss')([x, labels, input_len, label_len])
    ctc_model = Model(inputs=[input_tensor, labels, input_len, label_len], outputs=[loss_out])
    # 编译
    base_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    ctc_model.compile(loss={'ctc_loss':lambda y_true, y_pred: y_pred}, optimizer='adam')
    
    return base_model, ctc_model


def train_model(path, base_model, ctc_model, model_name, retrain=False, epochs=50):
    

    # 重新加载进行训练
    if retrain:
        ctc_model.load_weights('model/%s_weights.h5' % model_name)

    # 载入数据
    [X_data, labels, input_len, label_len], oriLabel = load_data(path)

    inputs = {
        'input_tensor': X_data,
        'the_labels': labels,
        'input_len': input_len,
        'label_len': label_len
        }
    outputs = {'ctc_loss': np.zeros([len(labels)])}  
	
    check_point = ModelCheckpoint(filepath='model/%s_weights.h5' % model_name, monitor='val_loss', save_best_only=True, save_weights_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    ctc_model.fit(inputs, outputs, batch_size=32, validation_split=0.2, epochs=epochs, callbacks=[check_point, early_stopping])

    return base_model


def test_model(base_model, path='img/test/origin_nacao', width=200, height=50):
    fnames = [os.path.join(path, fname) for fname in os.listdir(path)][:20]
    cnt = 0
    for fname in fnames:
        img = Image.open(fname).convert('RGB').resize((width, height), Image.ANTIALIAS)
        imgM = np.array(img, np.uint8).transpose(1, 0, 2)
        imgM = np.expand_dims(imgM, 0)
        y_pred = base_model.predict(imgM)
        y_out = ctable.decode(y_pred)
        y_true = fname.split('/')[-1].split('_')[0]
        print 'y_out', y_out, 'y_true',y_true
        if y_out==y_true:
            cnt += 1
    acc = float(cnt)/len(fnames)
    return acc
