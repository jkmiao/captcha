#!/usr/bin/env python
# coding=utf-8

from keras.models import Model, Input, load_model 
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import merge, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
from keras.callbacks import EarlyStopping
import numpy as np
from PIL import Image
import random
from sklearn.model_selection import train_test_split
import os
from util import CharacterTable
from skimage.io import imread
import string

# 基本参数配置
height, width, n_len, n_class = 60, 200, 6, 36
chars = string.digits + string.lowercase

def load_data(path='img/origin'):
    fnames = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith('jpg')]
    random.shuffle(fnames)
    data = np.zeros((len(fnames), height, width, 3))
    label = []
    for idx, fname in enumerate(fnames):
        img = Image.open(fname).convert('RGB').resize((width, height), Image.ANTIALIAS)
        data[idx] = np.array(img)    
        label.append(fname.split('/')[-1].split('_')[0])
    return data, label



# 编解码
ctable = CharacterTable(chars)

data, label = load_data()
label_onehot = np.zeros((len(label), n_len*n_class))
for i, lb in enumerate(label):
    label_onehot[i, :] = ctable.encode(lb)

print data[0]
print data.shape
print label_onehot.shape
n_label = len(label_onehot[0])
print 'n_label', n_label

x_train, x_test, y_train, y_test = train_test_split(data, label_onehot)

# 图片动态生成，加扭曲、旋转
datagen = image.ImageDataGenerator(
    rescale = 1./255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=10,
    horizontal_flip=False,
    vertical_flip=False
    )

DEBUG = False
if DEBUG:
    # 新定义
    input_tensor = Input((height, width, 3))
    x = input_tensor
    for i in range(1, 5):
		x = Conv2D(32*2*i, (3, 3), padding='same', activation='relu')(x)
		x = Conv2D(32*2*i, (3, 3))(x)
		x = LeakyReLU()(x)
		x = MaxPooling2D((2,2))(x)
		x = BatchNormalization()(x)
		x = Dropout(0.25)(x)

	
    x = Flatten()(x)
    x = Dropout(0.25)(x)
    x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(n_len)]
    merged = merge(x, mode='concat', concat_axis=-1)
    model_cnn = Model(inputs=input_tensor, outputs=merged)

else:
    # 重新加载进行训练
    model_cnn = load_model('model/tgcode_ne6.h5')


# 编译
model_cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 开始训练
early_stopping = EarlyStopping(monitor='loss', patience=10)
model_cnn.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
            samples_per_epoch=len(x_train), epochs=500, callbacks=[early_stopping])

# model_cnn.fit(x_train, y_train, batch_size=32, epochs=3)
# 保存模型
model_cnn.save('model/tgcode_ne6.h5')

# 预测结果
y_pred = model_cnn.predict(x_test[:50], verbose=0)
print 'y_pred shape', y_pred.shape
cnt = 0
for i in range(len(y_pred)):
    guess = ctable.decode(y_pred[i], n_len)
    correct = ctable.decode(y_test[i], n_len)
    if guess==correct:
        cnt += 1
    if i%10==0:
        print '--'*10
        print 'y_pred', guess
        print 'y_test', correct

print 'total accuracy', 1.0*cnt/len(y_pred)
