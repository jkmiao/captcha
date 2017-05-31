#!/usr/bin/env python
# coding=utf-8

import os, sys
import random
import argparse

def rdmv(src, dst, cnt=10):
    
    fnames = [os.path.join(src, fname) for fname in os.listdir(src)]
    random.shuffle(fnames)
    for fname in fnames[:cnt]:
        os.system("mv %s %s" % (fname, dst))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='random mv files from source to dest path')
    parser.add_argument('s', type=str, help='source path')
    parser.add_argument('d', type=str, help='dest path')
    parser.add_argument('-n', type=int, help='files number to move',  default=10)
    args = parser.parse_args()
    rdmv(args.src, args.dst, args.n)
