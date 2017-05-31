# -*- coding: utf-8 -*-
"""
Created on Sun May 14 23:31:50 2017

@author: Administrator
"""
from keras import models
import keras.backend as K
import string
import numpy as np

from PIL import Image
import os

image_h = 32
image_r = 200
labelmaxn = 12
labelsn = 26 + 10 + 1
chars = string.lowercase + string.digits + '-'
char_indices = dict((c, i) for i, c in enumerate(chars))
indice_chars = dict((i, c) for i, c in enumerate(chars))


def imgresize(img, image_h, image_r):
    """
    图片大小规范化
    """
    w, h = img.size  # 原本宽高
    ratio = float(image_h)/h # 缩放比例
    img = img.resize((int(w * ratio), int(h*ratio)), Image.ANTIALIAS) # 等高
    extra_blank = (image_r-img.width)/2  # 图片宽差

    img = img.crop((-extra_blank, 0, image_r - extra_blank, image_h))

    return img


def load_data(filepath):
    """
    图片文件转换为输入矩阵
    :type filepath: str : image file path
    :rtype np.array
    """
    imgM = np.array(imgresize(Image.open(filepath).convert('RGB'), image_h, image_r))
    return np.array([imgM])


class TGGVCR(object):
    '''
    use for generally recognizing verify code
    type: number or english
    code len: 1~12, with or without code len is ok.
    '''

    def __init__(self, model_path='models/tggvcr1.0.h5'):
        
        # 答案字符集
        chars = string.lowercase + string.digits + '-'
        self.char_indices = dict((c, i) for i, c in enumerate(chars))
        self.indice_chars = dict((i, c) for i, c in enumerate(chars))

        # 模型参数
        self.TIMESTEP = 50
        self.image_r = 32
        self.image_h = 200
        self.labelmaxn = 12
        self.labelsn = 26 + 10 + 1
        
        # 加载模型
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, model_path)
        self.predict_model = models.load_model(model_path)
        
    def ctc_decode(self, y_pred):
        t_s=np.argmax(y_pred,1)
        ret=[]
        ret.append(t_s[0])
        for i,t in enumerate(t_s):
            if i>0 and t!=t_s[i-1]:
                ret.append(t)
#        ret=np.array(ret)
#        ret=ret[ret<ret.shape[0]-1]
        ans = ''
        for j in xrange(len(ret)):
            ans += self.indice_chars[int(ret[j])]
        ans=ans.replace('-','')
#        for t in t_s:
#            if t<y_pred.shape[1]-1:
#                ret.append(t)
        return ans
        
    def predict(self, fname, code_len=None, detail=False):
        """
        根据传入图片文件预测
        :type fname :str 图片文件路径
        :type code_len :int 验证码答案字符长度
        :rtype :str 最终答案
        """
        X_s = load_data(fname)
        y_pred_s = self.predict_model.predict(X_s)
#        t = K.get_value(K.ctc_decode(y_pred_s, [self.TIMESTEP])[0][0])
        ans = self.ctc_decode(y_pred_s[0,:,:])
        
        # 量化规范到最终长度
        if code_len and code_len != len(ans):
            code_len = int(code_len)
            y_pred = y_pred_s[0, :, :-1]
            ind_s = np.argsort(np.max(y_pred, 1))
            ind_s = np.sort(ind_s[-code_len:])
            ans = ''
            y_pred_max_id = np.argmax(y_pred, 1)
            for ind in ind_s:
                ans += self.indice_chars[y_pred_max_id[ind]]
        return ans


if __name__ == "__main__":
   
    model_path= 'models/tggvcrv1.0_model.h5'
    fname = 'test_data/pJOjN2FLh_34031.jpg'
    
    test = TGGVCR(model_path)
    print 'y_true is:', 'pJOjN2FLh'.lower()
    print 'unfix len:', test.predict(fname)
    print 'fix len6:', test.predict(fname, 6)
    print 'fix len6:', test.predict(fname, 7)
    print 'fix len9:', test.predict(fname, 9)
