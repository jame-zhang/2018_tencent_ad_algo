#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/6 上午8:48
# @Author  : Jame
# @Site    : 
# @File    : predict_result_manipulate.py
# @Software: PyCharm

import pandas as pd
import os
import time




path = r"/home/ubuntu/tencent_final/datasets/merge/full_test2.csv"
result = r"/home/ubuntu/tencent_final/datasets/merge/add3/add3.prediction"

data = pd.read_csv(path)
data = data[["aid","uid"]]
print("data.shape[0]: "+str(data.shape[0]))
score = pd.read_csv(result,header=None)
score.columns = ['score']
print("score.shape[0]: "+str(score.shape[0]))
score['score'] = score['score'].apply(lambda x: '%.6f' % x)
if data.shape[0] == score.shape[0]:
    print("Same shape!!")
    data = pd.concat([data,score],axis=1)
    data.to_csv('submission/submission.csv',index=False)
    # os.system("zip submission/submission_"+time.strftime('%Y-%m-%d',time.localtime())+".zip" + "submission/submission.csv")
