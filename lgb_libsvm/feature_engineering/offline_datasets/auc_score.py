#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/9 下午9:59
# @Author  : Jame
# @Site    : 
# @File    : auc_score.py
# @Software: PyCharm

import pandas as pd
from sklearn.metrics import roc_auc_score


predict_path = r"/home/ubuntu/tencent_final/code/Jame/feature_engineering/offline_datasets/sub_add3/sub_add3.prediction"
result_path = r"/home/ubuntu/tencent_final/code/Jame/feature_engineering/offline_datasets/sub_add3/sub_valid_0.1_add3.csv"

reslut = pd.read_csv(result_path)["label"].values
predict = pd.read_csv(predict_path,header=None)
print("score is %f" % roc_auc_score(reslut,predict))

