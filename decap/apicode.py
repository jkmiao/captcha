#!/usr/bin/env python
# coding=utf-8

import os
from models.chimath.cm_code import CMCode
from models.tggvcr.tggvcr import TGGVCR
from models.tgcodesp.tgcodesp import TGcodesp
import base64
from cStringIO import StringIO
from urllib import urlopen

class Apicode(object):


    def __init__(self, path='model'):
        """
        模型预先加载
        """
        self.models = {}
        self.models['GNE'] = TGGVCR()  # 中英文通用模型
        self.models['CM'] = CMCode()  # 数学计算题验证码， 类似工商吉林山东或17小说网 http://passport.17k.com/mcode.jpg?r=8417905093765
        self.models['gne'] = TGcodesp('gne')


    def predict(self, codetype, fname, code_len=None, detail=False):

        if codetype not in self.models:
            raise ValueError("input captcha type error: %s " % type)

        # base64编码图片
        if len(fname)>1000:
            fname = StringIO(base64.b64decode(fname))

        return self.models[codetype].predict(fname, code_len=code_len, detail=detail)


    def predict_url(self, codetype, url, code_len=None, detail=False):
        if codetype not in self.models:
            raise ValueError("input captcha type error: %s " % type)

        # 图片url
        if url.startswith('http'):
            fname = StringIO(urlopen(url).read())

        return self.models[codetype].predict(fname, code_len=code_len, detail=detail)




if __name__ == '__main__':

    test = Apicode()
    ctype = 'gne'
    path = 'img/test/%s/' % ctype.lower()
    fnames = [os.path.join(path, fname) for fname in os.listdir(path)][:10]
    for fname in fnames:
        print fname
        # base64编码
        fname = base64.b64encode(open(fname).read())
        print test.predict(ctype, fname, detail=True)
