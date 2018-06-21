#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/3 下午10:08
# @Author  : Jame
# @Site    : 
# @File    : csv_libsvm_new2.py
# @Software: PyCharm

from tqdm import tqdm
import json
import os
import gc
import time

#TODO:
#1. 优化fit_transform函数，将2n降为n
#2. 添加低频过滤
#3. 优化低内存运行和选择
#4. label被当做特征添加进去，应该删除, 2018年06月05日18:44:38，check!
#5. 将train和test放在一起转化，防止字典出错

#NOTE：犯了一个很愚蠢的问题，既然fit了两个字典，test1没用train的字典，导致特征属性全错了

class csv_to_libsvm:
    #说明：1000w数据大概占用10G内存
    csv_header = []
    feature_word_dic_id = 1
    feature_word_dic = {}

    def __init__(self, file, file2="", save="", dic_path="", header=True, label='label', drop=[], kind =''):
        self.file = file
        self.label = label
        self.header = header
        self.idx = 1
        self.drop = drop
        self.drop_idx= []
        self.kind = kind
        self.save = save
        self.dic_path= dic_path
        self.file2 = file2


        pathlist = ["save", "dic_path"]
        for i in range(len(pathlist)):
            if not getattr(self,pathlist[i]).endswith("/"):
                setattr(self,pathlist[i],getattr(self,pathlist[i])+"/")


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
            if os.path.exists(self.dic_path + 'feature_word_dic.dict'):
                os.renames(self.dic_path + "feature_word_dic.dict", self.dic_path + "old_feature_word_dic.dict")
            self.dump_dict()
            print(self.kind + ": fit finished!\n")

    def fit_transform_onetime(self):
        print(self.kind + ": transform start!")
        if not self.feature_word_dic:
            self.load_dict()
        print(self.kind+": fit start!")
        line_result_list = []
        line_libsvm_label = []
        with open(self.file, 'r') as f:
            #处理header的
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

                if self.label:
                    temp_label = line_split_list[self.csv_header.index(self.label)]
                    if temp_label == "0.0":
                        line_libsvm_label = "0 "
                    elif temp_label == "1.0":
                        line_libsvm_label = "1 "
                    else:
                        line_libsvm_label = temp_label + " "

                for each, col in zip(line_split_list, self.csv_header):
                    for item in each.split():
                        if item not in self.feature_word_dic[col].keys():
                            self.feature_word_dic[col][item] = str( self.idx)
                            self.idx += 1
                    #确保每一行建完字典后再进行转化
                    if col == self.label:
                        continue
                    try:
                        line_libsvm_label += ''.join([self.feature_word_dic[col][_] + ':1 ' for _ in each.split()])
                    except KeyError:  # KeyError是因为测试集出现了训练集没有的特征值，这里直接丢弃，因为在训练集并没有学习到
                        continue
                line_result_list.append(line_libsvm_label)
            if os.path.exists(self.dic_path + 'feature_word_dic.dict'):
                os.renames(self.dic_path + "feature_word_dic.dict", self.dic_path + "old_feature_word_dic.dict")
            self.dump_dict()
            print(self.kind + ": fit finished!\n")
            print(self.kind + ": transformed and save Start!")

            with open(self.save + self.kind + ".libsvm", 'w') as f:
                f.write('\n'.join(line_result_list))
                f.write("\n")  # end of line
            print(self.kind + ": transformed and save Finished\n")
            del line_result_list
            gc.collect()

    def fit_transform_onetime_twofile(self):
        print(self.kind + ": transform start!")
        if not self.feature_word_dic:
            self.load_dict()
        print(self.kind+": fit start!")
        line_result_list = []
        line_libsvm_label = []
        f2 =open(self.file2,"r")
        with open(self.file, 'r') as f:
            #处理header的
            if self.header:
                self.csv_header = f.readline().strip().split(',')
                self.drop_idx = [self.csv_header.index(i) for i in self.drop]
                self.csv_header = self.csv_header+f2.readline().strip().split(',')
                for seq,i in enumerate(self.drop_idx):
                    del self.csv_header[i-seq]
                for col in self.csv_header:
                    self.feature_word_dic[col] = {}
            for line in tqdm(f):
                line_split_list = line.strip().split(',')
                line_split_list = self.line_drop(line_split_list)
                line_split_list += f2.readline().strip().split(',')
                if self.label:
                    temp_label = line_split_list[self.csv_header.index(self.label)]
                    if temp_label == "0.0":
                        line_libsvm_label = "0 "
                    elif temp_label == "1.0":
                        line_libsvm_label = "1 "
                    else:
                        line_libsvm_label = temp_label + " "

                for each, col in zip(line_split_list, self.csv_header):
                    for item in each.split():
                        if item not in self.feature_word_dic[col].keys():
                            self.feature_word_dic[col][item] = str( self.idx)
                            self.idx += 1
                    #确保每一行建完字典后再进行转化
                    if col == self.label:
                        continue
                    try:
                        line_libsvm_label += ''.join([self.feature_word_dic[col][_] + ':1 ' for _ in each.split()])
                    except KeyError:  # KeyError是因为测试集出现了训练集没有的特征值，这里直接丢弃，因为在训练集并没有学习到
                        continue
                line_result_list.append(line_libsvm_label)
            if os.path.exists(self.dic_path + 'feature_word_dic.dict'):
                os.renames(self.dic_path + "feature_word_dic.dict", self.dic_path + "old_feature_word_dic.dict")
            self.dump_dict()
            print(self.kind + ": fit finished!\n")
            print(self.kind + ": transformed and save Start!")

            with open(self.save + self.kind + ".libsvm", 'w') as f:
                f.write('\n'.join(line_result_list))
                f.write("\n")  # end of line
            print(self.kind + ": transformed and save Finished\n")
            del line_result_list
            gc.collect()


    def fit_two_file(self):
        print(self.kind+": fit start!")
        #file2为较小特征文件，长度与f1一直，即直接进行列拼接，所以直接打开,默认f2里面的字段没有需要丢弃的，即全部保留
        f2 = open(self.file2,"r")
        with open(self.file, 'r') as f:
            if self.header:
                self.csv_header = f.readline().strip().split(',')
                self.csv_header += f2.readline().strip().split(',')
                self.drop_idx = [self.csv_header.index(i) for i in self.drop]
                for seq,i in enumerate(self.drop_idx):
                    del self.csv_header[i-seq]
                for col in self.csv_header:
                    self.feature_word_dic[col] = {}
            for line in tqdm(f):
                line_split_list = line.strip().split(',')
                line_split_list += f2.readline().strip().split(',')
                line_split_list = self.line_drop(line_split_list)
                for each, col in zip(line_split_list, self.csv_header):
                    for item in each.split():
                        if item not in self.feature_word_dic[col].keys():
                            self.feature_word_dic[col][item] = str( self.idx)
                            self.idx += 1
            if os.path.exists(self.dic_path + 'feature_word_dic.dict'):
                os.renames(self.dic_path + "feature_word_dic.dict", self.dic_path + "old_feature_word_dic.dict")
            self.dump_dict()
            print(self.kind + ": fit finished!\n")
            f2.close()









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
                #对label继续转化，0.0->0,1.0->0，为了防止出现多分类且两个字符以上，因此不用str[0]的方法进行处理
                if self.label:
                    temp_label = line_split_list[self.csv_header.index(self.label)]
                    if temp_label == "0.0":
                        line_libsvm_label = "0 "
                    elif temp_label == "1.0":
                        line_libsvm_label = "1 "
                    else:
                        line_libsvm_label = temp_label + " "
                for col, each in zip(self.csv_header, line_split_list):
                    if col == self.label:
                        continue
                    try:
                        line_libsvm_label += ''.join([self.feature_word_dic[col][_] + ':1 ' for _ in each.split()])
                    except KeyError:  #KeyError是因为测试集出现了训练集没有的特征值，这里直接丢弃，因为在训练集并没有学习到
                        continue
                line_libsvm_list.append(line_libsvm_label)
            print(self.kind + ": transformed and save Start!")
            with open(self.save + self.kind+".libsvm", 'w') as f:
                f.write('\n'.join(line_libsvm_list))
                f.write("\n") #end of line
            print(self.kind + ": transformed and save Finished\n")
            del line_libsvm_list
            gc.collect()

    def transform_two_file(self):
        print(self.kind+": transform start!")
        if not self.feature_word_dic:
            self.load_dict()
        line_libsvm_list = []
        f2 = open(self.file2,"r")
        with open(self.file, 'r') as f:
            if self.header:
                self.csv_header = f.readline().strip().split(',')
                self.drop_idx = [self.csv_header.index(i) for i in self.drop]
                for seq, i in enumerate(self.drop_idx):
                    del self.csv_header[i - seq]
                self.csv_header+= f2.readline().strip().split(',')
            for line in tqdm(f):
                line_split_list = line.strip().split(',')
                line_split_list = self.line_drop(line_split_list)
                line_split_list+= f2.readline().strip().split(',')
                line_libsvm_label = ""
                #对label继续转化，0.0->0,1.0->0，为了防止出现多分类且两个字符以上，因此不用str[0]的方法进行处理
                if self.label:
                    temp_label = line_split_list[self.csv_header.index(self.label)]
                    if temp_label == "0.0":
                        line_libsvm_label = "0 "
                    elif temp_label == "1.0":
                        line_libsvm_label = "1 "
                    else:
                        line_libsvm_label = temp_label + " "
                for col, each in zip(self.csv_header, line_split_list):
                    if col == self.label:
                        continue
                    try:
                        line_libsvm_label += ''.join([self.feature_word_dic[col][_] + ':1 ' for _ in each.split()])
                    except KeyError:  #KeyError是因为测试集出现了训练集没有的特征值，这里直接丢弃，因为在训练集并没有学习到
                        continue
                line_libsvm_list.append(line_libsvm_label)
            print(self.kind + ": transformed and save Start!")
            with open(self.save + self.kind+".libsvm", 'w') as f:
                f.write('\n'.join(line_libsvm_list))
                f.write("\n") #end of line
            print(self.kind + ": transformed and save Finished\n")
            del line_libsvm_list
            gc.collect()

    def transform_limited_mem(self):
        """
        逐行写入，不需要list存储所有行，然后再全部写入
        :return:
        """
        print(self.kind + ": transform start!")
        if not self.feature_word_dic:
            self.load_dict()
        libsvm = open(self.save+self.kind + ".libsvm", 'w')
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
                # 对label继续转化，0.0->0,1.0->0，为了防止出现多分类且两个字符以上，因此不用str[0]的方法进行处理
                if self.label:
                    temp_label = line_split_list[self.csv_header.index(self.label)]
                    if temp_label == "0.0":
                        line_libsvm_label = "0 "
                    elif temp_label == "1.0":
                        line_libsvm_label = "1 "
                    else:
                        line_libsvm_label = temp_label + " "
                for col, each in zip(self.csv_header, line_split_list):
                    if col == self.label:
                        continue
                    try:
                        line_libsvm_label += ''.join([self.feature_word_dic[col][_] + ':1 ' for _ in each.split()])
                    except KeyError:
                        continue
                libsvm.write(line_libsvm_label+"\n")
                libsvm.flush()
            libsvm.write("\n")  # end of line
            libsvm.close()
            print(self.kind + ": transformed and save Finished\n")




    def transform_new(self):
        print(self.kind + ": transform start!")
        if not self.feature_word_dic:
            self.load_dict()
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
    def fit_transform_limited_mem(self):
        self.fit()
        self.transform_limited_mem()

    def load_dict(self,filename='.dict' ):
        if not self.dic_path:
            self.dic_path = self.save
        with open(self.dic_path + self.kind + filename) as f:
            self.feature_word_dic = json.load(f)

    def dump_dict(self, filename='.dict'):
        if not self.dic_path:
            self.dic_path = self.save
        with open(self.dic_path + self.kind + filename, 'w') as f:
            json.dump(self.feature_word_dic, f)



if __name__ == '__main__':

    path1 = r'/home/ubuntu/tencent_final/datasets/merge/full_train.csv'
    path2 = r'/home/ubuntu/tencent_final/datasets/new_feature_ljh/wh_part1_2/wh_train.csv'
    path3 = r'/home/ub36192/tencent_final/datasets/merge/full_test2.csv'
    path4 = r'/home/ub36192/tencent_final/datasets/new_feature_ljh/wh/wh_test2.csv'
    save = r"/home/ub36192/tencent_final/datasets/new_feature_ljh/wh"

    # s = time.time()
    # ctol = csv_to_libsvm(path1, path2, save=save, dic_path=save, drop=['aid', 'uid'], kind='wh_train', label='label')
    # ctol.fit_transform_onetime_twofile()
    # print(time.time()-s)
    s = time.time()
    ctol = csv_to_libsvm(path3,path4, save=save, dic_path=save, drop=['aid', 'uid'], kind='wh_train_test2',label='label')
    ctol.transform_two_file()
    print(time.time()-s)


