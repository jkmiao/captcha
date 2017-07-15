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

class Tgcodesp(object):


    def __init__(self, model_name, width=200, height=50, code_len=6, charsets=None):
        """
        初始化模型的名字,相关字符集, 验证码最大长度
        """
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

    def create_model(self, train_path, acc=0.92, nb_epoch=50, test_path=None):
       """
       创建新模型
       """
       model_names = [fname.split('/')[-1].split('.')[0] for fname in os.listdir('model')]
       if self.model_name in model_names:
           raise NameError("model name %s has existed! please change another one." % model_name)
       
       model, ctc_model = build_model(self.width, self.height, self.code_len, self.n_class)

       model = train_model(train_path, self.ctable, model, ctc_model, self.model_name, self.code_len, acc=acc, nb_epoch=nb_epoch)
       model.save('model/%s.h5' % self.model_name)
       return model
    

    def update_model(self, train_path, acc=0.92, nb_epoch=50, from_gne=True, test_path=None):
       """
       根据旧模型进行调优
       :type str: train_path 训练数据集路径
       :type float: acc: 训练集准确率停止条件
       :rtype :object model 训练好的预测模型
       """
       model_names = [fname.split('/')[-1].split('.')[0] for fname in os.listdir('model')]
       if self.model_name not in model_names:
           raise NameError("model name %s does not exist! please create it first." % model_name)

       model, ctc_model = build_model(self.width, self.height, self.code_len, self.n_class)
       if from_gne:
           ctc_model.load_weights("model/gne_weights.h5")
           for layer in model.layers[:5]:
               layer.trainable = False
       else:
           ctc_model.load_weights("model/%s_weights.h5" % self.model_name)
       print model.summary()
       model = train_model(train_path, self.ctable, model, ctc_model, self.model_name, self.code_len, acc=acc, nb_epoch=nb_epoch, test_path=test_path)
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
       y_pred = self.ctable.decode(y_pred)
       return y_pred


if __name__ == '__main__':
    
   # charsets = codecs.open('data/common_chisim_1.8k.txt', encoding='utf-8').read()
   charsets = string.digits + string.lowercase
   print len(charsets)
   print charsets[:50]
   test = Tgcodesp('qq168', code_len=10, charsets=charsets)
  # model = test.create_model('img/train/qq168', acc=0.95, nb_epoch=20)
   model = test.update_model(train_path='img/train/qq168_auto', acc=0.92, nb_epoch=50, test_path='img/test/qq168', from_gne=True)

   # 测试集文件夹测试模型准确率
   print '==='*20
   print 'test acc'
   print test_model(model, 'img/test/qq168', test.ctable)
