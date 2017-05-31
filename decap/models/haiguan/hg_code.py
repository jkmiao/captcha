#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 14:08:11 2017

@author: bigwayne
"""

'''
conv96
pool
norm
conv256
pool
norm
conv384
conv384
conv256
pool
fc4096
fc4096
fc10
'''

from keras.models import load_model
from sklearn import metrics
import numpy as np
from PIL import Image
import string
import os
import time


class CharacterTable(object):
    """
    字符串编解码
    """

    def __init__(self):
        self.chars = string.digits + string.lowercase
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indice_chars = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, strs):
        """
        编码为向量
        """
        vec = np.zeros((len(strs), len(self.chars)))
        for i, c in enumerate(strs):
            vec[i, self.char_indices[c]] = 1
        return vec.flatten()

    def decode(self, vec, code_len=6):
        """
        解码为字符
        """
        vec = vec.reshape(code_len, -1)  # 默认为6个
        vec_idx = vec.argmax(axis=-1)
        res = ''.join(self.indice_chars[x] for x in vec_idx)
        return res.lower()


class HGCode(object):

    def __init__(self, model_path='./model/HG_CNN.h5'):
        '''
        code_type:
            "HG"-中国海关企业进出口信用信息公示平台 http://credit.customs.gov.cn/, X shape equals (-1,20,95,3)
        '''
        self.image_h = 20
        self.image_w = 95
        self.ctable = CharacterTable()
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, model_path)
        self.model = load_model(model_path)

    def predict(self, fname, code_len=6, detail=False):
        imgM = np.array(Image.open(fname).resize((self.image_w, self.image_h)))
        imgM = np.expand_dims(imgM, 0)
        y_pred = self.model.predict(imgM)
        y_pred = [self.ctable.decode(y, code_len) for y in y_pred]
        return ''.join(y_pred)


if __name__ == "__main__":

    test = HGCode()
    path = 'test_data'
    fnames = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith('jpg')]
    
    start_time = time.time()
    cnt = 0
    for fname in fnames:
        y_pred = test.predict(fname)
        y_test = fname.split('/')[-1].split('_')[0].lower()
        if y_pred == y_test:
            cnt += 1
        print y_pred, y_test

    print 'Accuracy: ', float(cnt)/len(fnames)
    print 'Time: ', (time.time() - start_time) / len(fnames)
