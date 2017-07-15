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


def img_preprocess(img, image_w=200, image_h=50):
    """
    图片预处理, 统一规范化到 (200, 50) 宽高
    :type img: PIL.Image: 输入图片
    :type image_w: int 输出宽
    :type image_h: int 输出高
    :rtype : PIL.Image, 规范化后的图片
    """
    w, h = img.size
    new_width = w * image_h/h  # 等比例缩放
    img = img.resize((new_width, image_h), Image.ANTIALIAS) # 等高
    extra_blank = (image_w-img.width)/2
    img = img.crop((-extra_blank, 0, image_w - extra_blank, image_h))
    return img

def load_data(path, ctable,  width=200, height=50, code_len=5):
    """
    载入某个文件夹下的所有图像

    :type ctable: Charactable 标签编解码
    :type width: int: 预处理后输出宽
    :type height: int: 预处理后输出高
    :type code_len: int: 答案最大长度
    :rtype: [data, input_label, input_len, label_len], oriLabel
    """
    fnames = [os.path.join(path, fname) for fname in os.listdir(path) ]
    if len(fnames)>30000:
        fnames = random.sample(fnames, 30000)
    data = np.zeros((len(fnames), width, height, 3))  # 数据类型
    input_len = np.ones(len(fnames), dtype=np.uint8) * 19         # 21-2 reshape时的维度
    input_label = np.zeros((len(fnames), code_len), dtype=np.uint8)
    label_len = np.zeros(len(fnames), dtype=np.uint8)
    oriLabel = []
    for idx, fname in enumerate(fnames):
        try:
            img = Image.open(fname).convert('RGB')
            imgLabel = (fname.split('/')[-1].split('_')[0]).decode('utf-8')
            tmp = ctable.encode(imgLabel.lower())
        except Exception as e:
            print e
            os.system('mv %s ../../img/error/' % (fname))
            continue
        if len(imgLabel)<3:
            print 'too short label', fname
            os.system('mv %s ../../img/error/' % (fname))
            continue
        else:
            img = img_preprocess(img, width, height)
            input_label[idx] = ctable.encode(imgLabel.lower())
            data[idx] = np.array(img).transpose(1, 0, 2)
            label_len[idx] = len(imgLabel)
            oriLabel.append(imgLabel)
    return [data, input_label, input_len, label_len], oriLabel

def ctc_lambda_func(args):
    """
    内置 ctc 损失
    """
    y_pred, labels, input_len, label_len = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_len, label_len)


def build_model(width, height, code_len, n_class):
    """
    创建模型
    """
    input_tensor = Input((width, height, 3), name='input_tensor')
    x = input_tensor
    for i in range(3):
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2,2))(x)
    
    x = BatchNormalization()(x)
    conv_shape = x.get_shape()
    print 'conv_shape', conv_shape
    x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2]*conv_shape[3])))(x)
    x = Bidirectional(GRU(128, return_sequences=True), name='BiGRU1')(x)
    x = Dropout(0.25)(x)
    x = Dense(n_class, activation='softmax')(x)
    base_model = Model(inputs=input_tensor, outputs=x)
    
    labels = Input(name='the_labels', shape=[code_len], dtype='float32')
    input_len = Input(name='input_len', shape=[1], dtype='int64')
    label_len = Input(name='label_len', shape=[1], dtype='int64')
    
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc_loss')([x, labels, input_len, label_len])
    ctc_model = Model(inputs=[input_tensor, labels, input_len, label_len], outputs=[loss_out])
    # 编译
    base_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    ctc_model.compile(loss={'ctc_loss':lambda y_true, y_pred: y_pred}, optimizer='rmsprop')
    
    return base_model, ctc_model

def train_model(path, ctable, base_model, ctc_model, model_name, code_len, acc=0.92, nb_epoch=50, test_path=None):
    """
    载入创建的模型结构和数据,进行训练
    :type :object 训练好的模型
    """

    check_point = ModelCheckpoint(filepath='model/%s_weights.h5' % model_name, monitor='val_loss', save_best_only=True, save_weights_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    
    # 载入训练数据
    [X_data, labels, input_len, label_len], oriLabel = load_data(path, ctable, code_len=code_len)

    inputs = {
        'input_tensor': X_data,
        'the_labels': labels,
        'input_len': input_len,
        'label_len': label_len
    }
    outputs = {'ctc_loss': np.zeros([len(labels)])}  
        
    for epoch in range(nb_epoch/5):  
        hist = ctc_model.fit(inputs, outputs, batch_size=32, validation_split=0.1, epochs=1, callbacks=[check_point, early_stopping]) # 每次保存验证集中loss最小的模型
        train_acc = test_model(base_model, path, ctable)
        print epoch, 'train acc', train_acc, 'stop acc', acc
        if test_path:
            test_acc = test_model(base_model, test_path, ctable)
            print epoch, 'test acc', test_acc
        if train_acc > acc:   # 准确率达到期望的acc(默认0.92) 就提前退出
            break
        print ''
    return base_model


def test_model(base_model, path, ctable, cnt=100, width=200, height=50):
    """
    指定路径模型准确率测试
    :type path: 测试文件路径
    :type ctable: 编解码实例
    :rtype float: 指定文件夹下的图片准确率
    """
    fnames = [os.path.join(path, fname) for fname in os.listdir(path)]
    if len(fnames)<cnt:
        cnt = len(fnames)
    fnames = random.sample(fnames, cnt)  # 每次随机抽取100张做测试
    cnt = 0
    for idx, fname in enumerate(fnames):
        img = Image.open(fname).convert('RGB')
        img = img_preprocess(img, width, height)
        imgM = np.array(img).transpose(1, 0, 2)
        imgM = np.expand_dims(imgM, 0)
        y_pred = base_model.predict(imgM)
        y_out = ctable.decode(y_pred)
        y_true = fname.split('/')[-1].split('_')[0].lower()
        if y_out == y_true:
            cnt += 1
        if idx%10==0:
            print 'y_out', y_out, 'y_true',y_true
    acc = float(cnt)/len(fnames)
    return acc
