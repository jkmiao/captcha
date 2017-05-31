#!/usr/bin/env python
# coding=utf-8

import numpy as np
import string

class CharacterTable(object):

    def __init__(self, chars='0123456789abcs'):
 #       self.chars = string.uppercase + string.digits
        self.chars = chars
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, strs):
        """
        将验证码编码为向量
        """
        vec = np.zeros((len(strs), len(self.chars)))
        for i, c in enumerate(strs):
            vec[i, self.char_indices[c]] = 1
        return vec.flatten()

    def decode(self, vec, n_len=3, calc_argmax=True):
        """
        将向量解码为验证码
        """
        vec = vec.reshape(n_len, -1) # 6 个
        if calc_argmax:
            vec = vec.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in vec )

if __name__ == "__main__":
    import sys 
    test = CharacterTable()
    vec = test.encode(sys.argv[1])
    print vec.shape, vec
    print test.decode(vec)
