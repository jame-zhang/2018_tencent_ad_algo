# -*- coding:utf-8 -*-

import os
import numpy as np
from sklearn.metrics import roc_auc_score
from time import time
import sys
import pandas as pd
import tf_DeepFM
import gc
# import tf_deepffm_data_preprocess
# from utils import tf_deepffm_data_preprocess
# from utils import tf_deepffm_data_preprocess_split
# from deepmodels.tf_DeepFM import DeepFM

"""
static_index的维度: 行数*特征数量
dynamic_index的维度: 行数*[特征数量 * max_len]
dynamic_lengths的维度: 行数*特征数量
"""

# sys.path.append('../')
# from utils.args import args
# from utils import tencent_data_functions
import pickle



# print('read data...')
# sys.stdout.flush()

print('read transfromed data...')
sys.stdout.flush()
time1 = time()
train_data = pickle.load(open('../../datasets/deep_ffm_data/tf_deepmodel_train_with_ft20_ratio_cvr', 'rb'))
test_data = pickle.load(open('../../datasets/deep_ffm_data/tf_deepmodel_test_with_ft20_ratio_cvr', 'rb'))

print('first static index ', train_data['static_index'][0])
sys.stdout.flush()
print('process cost: ', time()-time1)
sys.stdout.flush()
time1 = time()
      
print('read transfromed dynamic data...')
sys.stdout.flush()
train_data_dynamic = pickle.load(open('../../datasets/deep_ffm_data/tf_deepmodel_dynamic_train_with_ft20_ratio_cvr', 'rb'))
test_data_dynamic = pickle.load(open('../../datasets/deep_ffm_data/tf_deepmodel_dynamic_test_with_ft20_ratio_cvr', 'rb'))

total_feature_sizes = [train_data['st_total_feature_size'], train_data_dynamic['dy_total_feature_size']]
field_sizes = train_data['field_sizes']
print('process cost: ', time()-time1)
sys.stdout.flush()

time1 = time()
print('total_feature_sizes', total_feature_sizes)
sys.stdout.flush()
print('field_sizes', field_sizes)
sys.stdout.flush()
print('split data...')
sys.stdout.flush()

train_size = round(len(train_data['static_index']) * 0.9)
static_index = train_data['static_index'][:train_size]
valid_static_index = train_data['static_index'][train_size:]
dynamic_index = train_data_dynamic['dynamic_index'][:train_size]
valid_dynamic_index = train_data_dynamic['dynamic_index'][train_size:]
dynamic_lengths = train_data_dynamic['dynamic_lengths'][:train_size]
valid_dynamic_lengths = train_data_dynamic['dynamic_lengths'][train_size:]
y = train_data['label'][:train_size]
valid_y = train_data['label'][train_size:]

test_static_index = test_data['static_index']
test_dynamic_index = test_data_dynamic['dynamic_index']
test_dynamic_lengths = test_data_dynamic['dynamic_lengths']
test_y = pd.read_csv('../../datasets/rawdata/test2.csv')
      
print('process cost: ', time()-time1)
sys.stdout.flush()
time1 = time()
print('first static index ', static_index[0])
sys.stdout.flush()
print('first dynamic length ', dynamic_lengths[0])
sys.stdout.flush()

del train_data
del test_data
del train_data_dynamic
del test_data_dynamic
gc.collect()

y_pred = np.array([0.0] * len(test_y))
print('start train...')
sys.stdout.flush()
for i in range(1):
    dfm = tf_DeepFM.DeepFM(field_sizes=field_sizes, total_feature_sizes=total_feature_sizes, embedding_size=9,
                 dynamic_max_len=10, learning_rate=0.0005, epoch=1, batch_size=2048)
    dfm.fit(static_index, dynamic_index, dynamic_lengths, y,
            valid_static_index, valid_dynamic_index, valid_dynamic_lengths, valid_y, combine=False)
    y_pred += dfm.predict(test_static_index, test_dynamic_index, test_dynamic_lengths)

y_pred /= 4.0

result = test_y[['aid', 'uid']]

print('save result...')
sys.stdout.flush()
result['score'] = y_pred
result['score'] = result['score'].apply(lambda x: float('%.6f' % x))
result.to_csv('../../datasets/submissions/submission_model_tf_deepffm_fl20_ratio_cvr.csv', index=False)
print('finish')
sys.stdout.flush()

# print('start jame program...')
# sys.stdout.flush()
# os.system("python /home/ub36192/tencent_final/code/Jame/csv_libsvm_new2.py")
# f = open('./datasets/submissions/submission_model_tf_deepffm.csv', 'wb')
# for y in y_pred:
#     f.write('%.6f' % (y) + '\n')
# f.close()







