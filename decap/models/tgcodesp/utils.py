#!/usr/bin/env python
# encoding=utf-8

import os
from PIL import Image
import string
import numpy as np
from keras.preprocessing import image


class CharacterTable(object):
    
    """
    字符串编码
    """
    
    def __init__(self, code_len, charsets):
        
        self.char_indice = dict((c, i) for i, c in enumerate(charsets))
        self.indice_char = dict((i, c) for i, c in enumerate(charsets))
        self.code_len = code_len

    def encode(self, imgLabel):
        """
        验证码标签转向量
        :type imgLabel: str
        :return img vector
        """
  
        code_len = self.code_len if self.code_len else len(imgLabel)
        vec = np.zeros(code_len, dtype=np.uint8)
        for i, c in enumerate(imgLabel):
            vec[i] = self.char_indice[c]
        return vec

    def decode(self, y_pred, code_len=None):
        """
        ctc结果解码为最终结果
        """
        y_pred = y_pred[:, 2:, :]
        res = [self.indice_char[i] for i in np.argmax(y_pred[0], axis=1)]
        y_out = str(res[0])
        for i, c in enumerate(res):
            if i>0 and res[i-1]!=c and c != '-':
                y_out += c
        y_out = y_out.replace('-', '')
        if code_len:
            code_len = int(code_len)
        if code_len and code_len!= len(y_out):
            y_pred = y_pred[0, :, :-1]
            ind_s = np.argsort(np.max(y_pred, 1))
            ind_s = np.sort(ind_s[-code_len:])
            y_out = ''
            y_pred_max_id = np.argmax(y_pred, 1)
            for ind in ind_s:
                y_out += self.indice_char[y_pred_max_id[ind]]
        return y_out


class ImgEnhance(object):
    
    def __init__(self):
        
        self.imggen = image.ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1, zoom_range=0.1)
    
    def gen_images(self, path, save_to_dir):
        
        fnames = [os.path.join(path, fname) for fname in os.listdir(path)]
        for idx, fname in enumerate(fnames):
            imgM = np.array(Image.open(fname))
            imgM = np.expand_dims(imgM, 0)
            imgLabel = fname.split('/')[-1].split('_')[0]
            cnt = 0
            for batch in self.imggen.flow(imgM, save_prefix=imgLabel, save_to_dir=save_to_dir):
                cnt += 1
                if cnt > 10:
                    break
            if idx%200==0:
                print idx



if __name__ == '__main__':
    
    test = ImgEnhance()
    test.gen_images('img/train/chisim_real', 'img/train/chisim_auto')
