#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/4 下午5:54
# @Author  : Jame
# @Site    :
# @File    : libsvm_split.py
# @Software: PyCharm


import numpy as np
import gc
from tqdm import tqdm
import math
import os
from sklearn.feature_extraction.text import CountVectorizer
from io import open


class libsvm_split:

    def __init__(self,file, rate=0.2):
        self.file = file
        self.rate = rate

    def split(self):
        masks = np.random.rand(44000000)
        masks = masks < self.rate
        train_libvm_split =  open("train_libsvm.split","w")
        test_libvm_split = open("test_libsvm.split","w")
        #count = 0
        with open(self.file,'r') as f:
            for i, mask in tqdm(zip(f, masks)):
                if mask:
                    train_libvm_split.write(f.readline())
                else:
                    test_libvm_split.write(f.readline())
                #count += 1
                #if count == 200000:
                    #train_libvm_split.flush()
                    #test_libsvm_split.flush()
        train_libvm_split.close()
        test_libvm_split.close()
        gc.collect()

    def split2(self):
        train_libvm_split = open("train_libsvm.split", "w")
        test_libvm_split = open("test_libsvm.split", "w")
        with open(self.file, 'r') as f:
            data = f.readlines()
            random_list = np.random.randint(0,len(data),int(len(data)*self.rate))
            for idx,line in data:
                if idx in random_list:
                    test_libvm_split.write(line)
                else:
                    train_libvm_split.write(line)
        train_libvm_split.close()
        test_libvm_split.close()
        gc.collect()


train_data = libsvm_split("train_libsvm")
train_data.split2()