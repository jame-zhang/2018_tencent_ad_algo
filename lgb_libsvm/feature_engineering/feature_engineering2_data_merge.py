#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/7 下午9:35
# @Author  : Jame
# @Site    : 
# @File    : feature_engineering2_data_merge.py
# @Software: PyCharm


import pandas as pd


train_path = r""
valid_path = r""
test1_path = r""
test2_path = r""


data = pd.read_csv(train_path)
data.loc[data['label']==-1,'label']=0





# merge test1 dataset
temp = pd.read_csv(test1_path)
data["n_parts"] = 6
data = pd.concat([data,temp])
# merge test2 dataset
temp = pd.read_csv(test2_path)
data["n_parts"] = 7
data = pd.concat([data,temp])
