#!/usr/bin/env python
# encoding: utf-8

"""
__description__: 验证码个性化训练
__author__ : jkmiao
__date__: 2017-06
"""

import string, os
import numpy as np
from utils import CharacterTable
from PIL import Image
from keras.models import load_model

class TGcodesp(object):

    def __init__(self, model_name, width=200, height=50, code_len=6, charsets=None):
        
        if not charsets:
            self.charsets = string.digits + string.lowercase
        self.charsets += '-'
        self.n_class = len(self.charsets)
        self.width = width
        self.height = height
        self.code_len = code_len
        self.ctable = CharacterTable(code_len=code_len, charsets=self.charsets)
        
        base_path = os.path.dirname(__file__)
        self.model = load_model(os.path.join(base_path, 'model/%s.h5' % model_name))
    
    def img_preprocess(self, fname):
        """
        图片预处理
        """
        img = Image.open(fname).convert('RGB')
        w, h = img.size
        new_w = w * self.height / h
        img = img.resize((new_w, self.height), Image.ANTIALIAS)
        extra_blank = (self.width - img.width)/2
        img = img.crop((-extra_blank, 0, self.width-extra_blank, self.height))
        return img

    def predict(self, fname, code_len=None, detail=False):
       """
       使用模型进行预测
       """
       img = self.img_preprocess(fname)
       imgM = np.array(img).transpose(1, 0, 2)
       imgM = np.expand_dims(imgM, 0)
       y_pred = self.model.predict(imgM)
       y_pred = self.ctable.decode(y_pred, code_len)
       if detail:
           try:
               y_pred = self.convert_res(y_pred)
           except :
               y_pred = y_pred
       return str(y_pred)


    def convert_res(self, label):
        """
        对计算题类型最终结果进行转化
        """
        op_map = {'a':'+', 'b': '-', 'c': '*', 'd':'/'}
        res = ''
        for s in label:
            if s in op_map:
               s = op_map[s]
            res += s
        return eval(res)


if __name__ == '__main__':
    
     test = TGcodesp('sogou')
     path = 'img/test/sogou/'
     fnames = [os.path.join(path, fname) for fname in os.listdir(path)]
     cnt = 0
     for fname in fnames:
         y_true = fname.split('/')[-1].split('_')[0].lower()
         y_pred = test.predict(fname, detail=True)
         print y_true, y_pred
         if y_true == y_pred:
            cnt += 1
     print float(cnt)/len(fnames)
