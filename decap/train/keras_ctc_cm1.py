#!/usr/bin/env python
# coding: utf-8

from keras.models import Model, Input, load_model 
from keras.layers import *
from keras import backend as K
import numpy as np
from PIL import Image
import random
import os


# 基本参数配置
charsets = '0123456789abcs'
rnn_size = 128
height, width, max_len, n_class = 50, 300, 5, len(charsets) 
char_dict = dict((c, i) for i, c in enumerate(charsets))
indic_dict = dict((i, c) for i, c in enumerate(charsets))
TIMESTEP = 23


def load_data(path='img/origin_jl'):
    fnames = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith('jpg')]
    random.shuffle(fnames)
    data = np.zeros((len(fnames), width, height, 3), dtype=np.uint8)  # 数据类型
    input_label = np.zeros((len(fnames), max_len), dtype=np.uint8)
    input_len = np.ones(len(fnames), dtype=np.uint8)*33         # 23-2 reshape时的维度
    label_len = np.zeros(len(fnames), dtype=np.uint8)
    oriLabel = []
    for idx, fname in enumerate(fnames):
        img = Image.open(fname).convert('RGB').resize((width, height), Image.ANTIALIAS)
        data[idx] = np.array(img).transpose(1, 0, 2)
        imgLabel = (fname.split('/')[-1].split('_')[0])
        tmp_vec = np.zeros(max_len, dtype=np.uint8)   # 默认为0
        for i, c in enumerate(imgLabel):
            tmp_vec[i] = char_dict[c]  # 填充相关正确参数
        input_label[idx] = tmp_vec
        label_len[idx] = len(imgLabel)
        oriLabel.append(imgLabel)
      
    return [data, input_label, input_len, label_len], oriLabel


def ctc_lambda_func(args):
    y_pred, labels, input_len, label_len = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_len, label_len)


DEBUG = True
if DEBUG:
    # 新定义
    input_tensor = Input((width, height, 3), name='input_tensor')
    x = input_tensor
    for i in range(3):
        x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2,2))(x)
    
    x = BatchNormalization()(x)
    conv_shape = x.get_shape()
    x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2]*conv_shape[3])))(x)
    x = Bidirectional(GRU(rnn_size, return_sequences=True), name='BiGRU1', merge_mode='sum')(x)
    x = Bidirectional(GRU(rnn_size, return_sequences=True), name='BiGRU2', merge_mode='concat')(x)
    x = Dropout(0.25)(x)
    x = Dense(n_class, activation='softmax')(x)
    base_model = Model(inputs=input_tensor, outputs=x)
    
    labels = Input(name='the_labels', shape=[max_len], dtype='float32')
    input_len = Input(name='input_len', shape=[1], dtype='int64')
    label_len = Input(name='label_len', shape=[1], dtype='int64')
    
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc_loss')([x, labels, input_len, label_len])
    
    model = Model(inputs=[input_tensor, labels, input_len, label_len], outputs=[loss_out])
	
	# 编译
    model.compile(loss={'ctc_loss':lambda y_true, y_pred: y_pred}, optimizer='adam')
    print model.summary()
    print 'conv_shape', conv_shape
    print 'x_shape', x.get_shape()
    
#    base_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


# 重新加载进行训练
# model.load_weights('model/tgcode_ctc_weights.h5')

# 载入数据
[X_data, labels, input_len, label_len], oriLabel = load_data()

print 'X_data', X_data.shape
print 'labels', labels.shape, labels[:5]
print 'input_len', input_len.shape, input_len[:5]
print 'label_len', label_len.shape, label_len[10:100:5]

inputs = {
    'input_tensor': X_data,
    'the_labels': labels,
    'input_len': input_len,
    'label_len': label_len
    }
outputs = {'ctc_loss': np.zeros([len(labels)])}  

def test(path='img/origin_jl'):
    fnames = [os.path.join(path, fname) for fname in os.listdir(path)][:5]
    cnt = 0
    for fname in fnames:
        img = Image.open(fname).convert('RGB').resize((width, height), Image.ANTIALIAS)
        img = np.array(img, np.uint8).transpose(1, 0, 2)
        img = np.expand_dims(img, 0)
        y_pred = base_model.predict(img)
        y_pred = y_pred[:, 2:, :]
        ctc_decode = (K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0])*y_pred.shape[1])[0][0])
        y_out = K.get_value(ctc_decode)[:, :5]
        y_out = ''.join([charsets[x] for x in y_out[0]])
        y_true = fname.split('/')[-1].split('_')[0]
        print y_out, y_true
        if y_out==y_true:
            cnt += 1
        print '=='*20
        print float(cnt)/len(fnames)
		
# test()

for i in range(50):
    hist = model.fit(inputs, outputs, batch_size=32, epochs=2)
    print '=='*10, i
    test()

y_pred = base_model.predict(X_data[:2])
y_pred = y_pred[:, 2:, :]
print 'y_pred.shape', y_pred.shape
ctc_decode = K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0])*y_pred.shape[1])[0][0]
y_out = K.get_value(ctc_decode)[:, :4]
print y_out
out = ''.join([charsets[x] for x in y_out[0]])
print out
print oriLabel[0]

model.save_weights('model/tgcode_ctc_weights.h5')
base_model.save('model/tgcode_ctc_cm1.h5')
