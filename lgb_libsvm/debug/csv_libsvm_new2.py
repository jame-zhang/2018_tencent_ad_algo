#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/7 上午9:31
# @Author  : Jame
# @Site    : 
# @File    : csv_libsvm_new2.py
# @Software: PyCharm

from tqdm import tqdm
import json
import os

#TODO:
#1. 优化fit_transform函数，将2n降为n
#2. 添加低频过滤
#3. 优化低内存运行和选择
#4. label被当做特征添加进去，应该删除, 2018年06月05日18:44:38，check!

class csv_to_libsvm:
    #说明：1000w数据大概占用10G内存
    csv_header = []
    feature_word_dic_id = 1
    feature_word_dic = {}

    def __init__(self, file, header=True, label='label', drop=[],kind ='' ):
        self.file = file
        self.label = label
        self.header = header
        self.idx = 1
        self.drop = drop
        self.drop_idx= []
        self.kind = kind

    def fit(self):
        print(self.kind+": fit start!")
        with open(self.file, 'r') as f:
            if self.header:
                self.csv_header = f.readline().strip().split(',')
                self.drop_idx = [self.csv_header.index(i) for i in self.drop]
                for seq,i in enumerate(self.drop_idx):
                    del self.csv_header[i-seq]
                for col in self.csv_header:
                    self.feature_word_dic[col] = {}
            for line in tqdm(f):
                line_split_list = line.strip().split(',')
                line_split_list = self.line_drop(line_split_list)
                for each, col in zip(line_split_list, self.csv_header):
                    for item in each.split():
                        if item not in self.feature_word_dic[col].keys():
                            self.feature_word_dic[col][item] = str( self.idx)
                            self.idx += 1
            if os.path.exists('feature_word_dic.dict'):
                os.renames("feature_word_dic.dict","old_feature_word_dic.dict")
            self.dump_dict(filename=self.kind+'_feature_word_dic.dict')
            print(self.kind + ": fit finished!\n")

    def trans_line(self, line):
        line_split_list = line.strip().split(',')
        line_split_list = self.line_drop(line_split_list)
        lib_svm_line = line_split_list[self.csv_header.index(self.label)] + " "
        for col, each in zip(self.csv_header, line_split_list):
            lib_svm_line += ' '.join(
                [self.feature_word_dic[col][_] + ':1' for _ in each.split()])
        return lib_svm_line

    def line_drop(self,line):
        for i,item in enumerate(self.drop_idx):
            del line[item-i]    #Attention, the old idx should move with del action
        return line
    def transform(self):
        print(self.kind+": transform start!")
        if not self.feature_word_dic:
            self.load_dict()
        line_libsvm_list = []
        with open(self.file, 'r') as f:
            if self.header:
                self.csv_header = f.readline().strip().split(',')
                self.drop_idx = [self.csv_header.index(i) for i in self.drop]
                for seq, i in enumerate(self.drop_idx):
                    del self.csv_header[i - seq]
            for line in tqdm(f):
                line_split_list = line.strip().split(',')
                line_split_list = self.line_drop(line_split_list)
                line_libsvm_label = ""
                if self.label:
                    line_libsvm_label = line_split_list[self.csv_header.index(self.label)] + " "
                for col, each in zip(self.csv_header, line_split_list):
                    if col == self.label:
                        continue
                    line_libsvm_label += ''.join([self.feature_word_dic[col][_] + ':1 ' for _ in each.split()])
                line_libsvm_list.append(line_libsvm_label)
            print(self.kind + ": transformed and save Start!")
            # print(line_libsvm)
            with open(self.kind+"_libsvm", 'w') as f:
                f.write('\n'.join(line_libsvm_list))
            print(self.kind + ": transformed and save Finished\n")

    def transform_new(self):
        line_svm_list = []
        with open(self.file, 'r') as f:
            if self.header:
                self.csv_header = f.readline().strip().split(',')

            # TODO : label !!!

            lines = f.readlines()
            line_svm_list = [self.trans_line(line) for line in lines]
            libsvm_content = '\n'.join(line_svm_list)
            print("Save Start!")
            with open("libsvm", 'w') as f:
                f.write(libsvm_content)

    def fit_transform(self):
        self.fit()
        self.transform()

    def load_dict(self,filename='feature_word_dic.dict' ):
        if self.kind:
            filename = self.kind + '_feature_word_dic.dict'
        with open(filename) as f:
            self.feature_word_dic = json.load(f)

    def dump_dict(self, filename='feature_word_dic.dict'):
        if self.kind:
            filename = self.kind + '_feature_word_dic.dict'
        with open(filename, 'w') as f:
            json.dump(self.feature_word_dic, f)

    def build_trans_word_dic(self):
        with open(self.file, 'r') as f:
            if self.header:
                self.csv_header = f.readline().strip().split(',')
                for col in self.csv_header:
                    self.feature_word_dic[col] = {}
            for line in f:
                line_split_list = line.strip().split(',')
                for each, col in zip(line_split_list, self.csv_header):
                    for item in each.split():
                        if item not in self.feature_word_dic[col].keys():
                            self.feature_word_dic[col][item] = str(self.idx)
                            self.idx += 1
            # print(self.feature_word_dic)
            with open('feature_word_dic.dict', 'w') as d:
                json.dump(self.feature_word_dic, d)


if __name__ == '__main__':
    print(os.getcwd())
    path1 = r'train_debug.csv'
    # path2 = r'data_test_sub_0_01.csv'

    ctol = csv_to_libsvm(path1, kind='train',label='label')
    ctol.fit_transform()

    # ctol = csv_to_libsvm(path2, drop=['aid', 'uid','label'], kind='test', label='')
    # ctol.fit_transform()

    # ctol = csv_to_libsvm(path2, drop=['aid','uid'],kind='test1')
    # ctol.transform()