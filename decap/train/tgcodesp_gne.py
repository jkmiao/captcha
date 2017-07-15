#!/usr/bin/env python
# encoding: utf-8

"""
__description__: 验证码个性化训练
__author__ : jkmiao

"""

import string
import codecs
from utils import CharacterTable
from train_model import *
import pdb

class Tgcodesp(object):


    def __init__(self, model_name, width=200, height=50, code_len=6, charsets=None):
        
        if not charsets:
            self.charsets = string.digits + string.lowercase
        else:
            self.charsets = charsets
        self.charsets += '-'
        self.n_class = len(self.charsets)
        self.width = width
        self.height = height
        self.code_len = code_len
 
        self.ctable = CharacterTable(code_len=code_len, charsets=self.charsets)
        self.model_name = model_name.strip().lower()

    def create_model(self, train_path, nb_epoch=50, test_path=None):
       """
       创建新模型
       """
       model_names = [fname.split('/')[-1].split('.')[0] for fname in os.listdir('model')]
       if self.model_name in model_names:
           raise NameError("model name %s has existed! please change another one." % model_name)
       
       model, ctc_model = build_model(self.width, self.height, self.code_len, self.n_class)
       model = train_model(train_path, self.ctable, model, ctc_model, self.model_name, code_len=self.code_len, nb_epoch=nb_epoch, test_path=test_path)
       model.save('model/%s.h5' % self.model_name)
       return model
    

    def update_model(self, train_path, from_gne=True, nb_epoch=50, test_path=None):
       """
       根据旧模型进行调优
       """
       model_names = [fname.split('/')[-1].split('.')[0] for fname in os.listdir('model')]
       if self.model_name not in model_names:
           raise NameError("model name %s has not existed! please create model first." % model_name)

       model, ctc_model = build_model(self.width, self.height, self.code_len, self.n_class)
       if from_gne:
           ctc_model.load_weights("model/gne_weights.h5")
       else:
           ctc_model.load_weights("model/%s_weights.h5" % self.model_name)

       model = train_model(train_path, self.ctable, model, ctc_model, self.model_name, code_len=self.code_len, nb_epoch=nb_epoch, test_path=test_path)
       model.save('model/%s.h5' % self.model_name)
       return model
    

    def predict(self, model, fname):
       """
       使用模型进行预测
       """
       img = Image.open(fname).convert('RGB')
       img = img_preprocess(img, self.width, self.height)
       imgM = np.array(img).transpose(1, 0, 2)
       imgM = np.expand_dims(img, 0)
       y_pred = base_model.predict(imgM)
       y_pred = y_pred[:, 2:, :]
       y_pred = self.ctable.decode(y_pred)
       return y_pred


if __name__ == '__main__':
    
   charsets = string.digits + string.lowercase
   print charsets[:50]
   test = Tgcodesp('gne', code_len=10, charsets=charsets)
   model = test.create_model('img/train/multisites')
   model = test.update_model('img/train/multisites', nb_epoch=10, test_path='img/test/multisites')
