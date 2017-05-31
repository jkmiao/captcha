#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
> author: jkmiao
> Date: 2017-05-23
中文四则运算验证码


"""

from keras.models import load_model
from keras import backend as K
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
        self.chars = u'0123456789abcs'
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indice_chars = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, strs, maxlen=None):
        maxlen = maxlen if maxlen else len(strs)
        vec = np.zeros((maxlen, len(self.chars)))
        for i, c in enumerate(strs):
            vec[i, self.char_indices[c]] = 1
        return vec.flatten()

    def decode(self, vec, n_len=6):
        vec = vec[:, 2:, :]
        ctc_decode = K.ctc_decode(vec, input_length=np.ones(vec.shape[0])*vec.shape[1])[0][0]
        y_out = K.get_value(ctc_decode)

        res = ''.join([self.indice_chars[x] for x in y_out[0]])
        return res.lower()


class CMCode(object):

    def __init__(self, model_path='model/tgcode_ctc_cm.h5'):
        
        self.ctable = CharacterTable()
        
        self.image_h = 30
        self.image_w = 200
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, model_path)
        self.model = load_model(model_path)

    def predict(self, fname):
        img = Image.open(fname).convert('RGB').resize((self.image_w, self.image_h), Image.ANTIALIAS)
        imgM = np.array(img, dtype=np.uint8).transpose(1, 0, 2)
        imgM = np.expand_dims(imgM, 0)
        y_pred = self.model.predict(imgM)
        y_pred = self.ctable.decode(y_pred)
        return y_pred


if __name__ == "__main__":

    test = CMCode()
    path = 'img/test_data/cm21'
    fnames = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith('jpg')][:50]
    
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
