import numpy as np
import pandas as pd
import os
from scipy import sparse
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from time import time
import gc

one_hot_feature=['creativeSize', 'LBS','age','carrier','consumptionAbility', 'ct', 'education','gender','house','os','marriageStatus','advertiserId','campaignId', 'creativeId',
       'adCategoryId', 'productId', 'productType']

vector_feature=['interest1','interest2','interest5','kw1','kw2','topic1','topic2']


one_hot_cvr = ['cvr_of_aid_and_age',
 'cvr_of_aid_and_gender',
 'cvr_of_aid_and_consumptionAbility',
'cvr_of_aid_and_education',
'cvr_of_aid_and_house',
'cvr_of_aid_and_LBS',
'cvr_of_aid',
'cvr_of_creativeSize_and_gender',
 'cvr_of_creativeSize_and_productType'
 ]

one_hot_ratio = [
 'ratio_click_of_age_in_aid',
 'ratio_click_of_age_in_creativeSize',
 'ratio_click_of_consumptionAbility_in_aid',
 'ratio_click_of_productType_in_consumptionAbility',
 'ratio_click_of_productType_in_age',
 'ratio_click_of_gender_in_consumptionAbility',
 'ratio_click_of_gender_in_aid',
 'ratio_click_of_creativeSize_in_productType',
 'ratio_click_of_aid_in_creativeSize',
 'ratio_click_of_productId_in_uid',
 'ratio_click_of_gender_in_productId',
 'ratio_click_of_consumptionAbility_in_age',
 ]

one_hot_feature = one_hot_feature + one_hot_cvr + one_hot_ratio

def get_raw_data(train_path, test_path):
    time1 = time()
    print('read train data...')
    train_data = pd.read_csv(train_path)
    print('read test data...')
    test_data = pd.read_csv(test_path)
    train_y = train_data['label'].map(lambda x: int(float(x))).values.squeeze()
    train_data = train_data.fillna('-1')
    test_data = test_data.fillna('-1')
    print('read raw data cost: {} s'.format(time()-time1))
    return train_data, train_y, test_data


def get_new_lisan_feature(train_path, test_path, train_path_2, test_path_2):
    time1 = time()
    print('get new feature...')
    new_feature_train1 = pd.read_csv(train_path)
    new_feature_test1 = pd.read_csv(test_path)
    new_feature_train_2 = pd.read_csv(train_path_2)
    new_feature_test2 = pd.read_csv(test_path_2)
    new_feature_train = pd.concat([new_feature_train1, new_feature_train_2], axis=1)
    new_feature_test = pd.concat([new_feature_test1, new_feature_test2], axis=1)
    train_size = len(new_feature_train)
    del new_feature_train1
    del new_feature_test1
    del new_feature_train_2
    del new_feature_test2
    gc.collect()
    all_new_fea = pd.concat([new_feature_train, new_feature_test], axis=0, ignore_index=True)
    del new_feature_train
    del new_feature_test
    gc.collect()
    new_fea_lisan = pd.DataFrame()
    for col in all_new_fea.columns:
        time1 = time()
        print('process: ', col)
        fea_lisan = pd.cut(all_new_fea[col], bins=10, labels=False).astype(str)
        new_fea_lisan = pd.concat([new_fea_lisan, fea_lisan], axis=1)
        print('process cost: ', time()-time1)
    print('read new feature cost: {} s'.format(time() - time1))
    return new_fea_lisan.iloc[:train_size], new_fea_lisan.iloc[train_size:].reset_index()


def get_all_index(train_data, test_data, label):
    result_train = {}
    result_test = {}
    ffm_train = pd.DataFrame()
    ffm_test = pd.DataFrame()
    print('get feature index...')
    idx = 0
    for col in one_hot_feature:
        time1 = time()
        print('process ', col)
        col_value = np.unique(train_data[col].unique().tolist()+test_data[col].unique().tolist())
        feat_dict = dict(zip(col_value, range(idx, idx+len(col_value))))
        se_train = train_data[col].apply(lambda x: feat_dict.get(x, idx+len(col_value)))
        se_test = test_data[col].apply(lambda x: feat_dict.get(x, idx+len(col_value)))
        ffm_train = pd.concat([ffm_train, se_train], axis=1)
        ffm_test = pd.concat([ffm_test, se_test], axis=1)
        idx = idx + len(col_value) + 1
        print('process cost: {} s'.format(time() - time1))
    result_test['static_index'] = ffm_test.values
    result_train['static_index'] = ffm_train.values
    result_train['label'] = label
    result_train['st_total_feature_size'] = idx

    print('get interest index...')
    all_dy_ids = []
    all_dy_length = []
    all_dy_ids_test = []
    all_dy_length_test = []
    ids_dy = 0
    for index, col in enumerate(vector_feature):
        time1 = time()
        print('processing ', col)
        dict_ids_num = {}
        for item in train_data[col] + ' ' + test_data[col]:
            for inter in item.strip().split(' '):
                if inter in dict_ids_num:
                    dict_ids_num[inter] += 1
                else:
                    dict_ids_num[inter] = 1
        dict_ids_num = {i: vals for i, vals in dict_ids_num.items() if vals >= 20}
        col_value = dict(zip(dict_ids_num.keys(), range(ids_dy, ids_dy + len(dict_ids_num))))
        train_se = []
        test_se = []
        buf_length = []
        buf_length_test = []
        print('transform train data...')
        for m in train_data[col]:
            ses = set()
            for n in m.split(' '):
                if n in col_value:
                    se = col_value[n]
                else:
                    se = len(col_value) + ids_dy
                ses.add(se)
            buf_length.append(len(ses))
            train_se.append(list(ses))

        all_dy_ids.append(train_se)
        all_dy_length.append(buf_length)

        print('transform test data...')
        for m in test_data[col]:
            ses = set()
            for n in m.split(' '):
                if n in col_value:
                    se = col_value[n]
                else:
                    se = len(col_value) + ids_dy
                ses.add(se)
            buf_length_test.append(len(ses))
            test_se.append(list(ses))

        all_dy_ids_test.append(test_se)
        all_dy_length_test.append(buf_length_test)
        ids_dy = ids_dy + len(col_value) + 1
        print('process cost: {} s'.format(time() - time1))

    result_test['dynamic_index'] = np.array(all_dy_ids_test).transpose()
    result_test['dynamic_lengths'] = np.array(all_dy_length_test).transpose()

    result_train['dynamic_index'] = np.array(all_dy_ids).transpose()
    result_train['dynamic_lengths'] = np.array(all_dy_length).transpose()
    result_train['dy_total_feature_size'] = ids_dy
    result_train['field_sizes'] = [len(one_hot_feature), len(vector_feature)]

    print('saving train data...')
    pickle.dump(result_train, open('../../datasets/deep_ffm_data/tf_deepmodel_train_with_ft20_ratio_cvr', 'wb'))
    print('saving test data...')
    pickle.dump(result_test, open('../../datasets/deep_ffm_data/tf_deepmodel_test_with_ft20_ratio_cvr', 'wb'))


print('read the raw data...')

train_data, train_label, test_data = get_raw_data('../../datasets/merge/full_train.csv', '../../datasets/merge/full_test2.csv')
new_fea_lisan_train, new_fea_lisan_test = get_new_lisan_feature('../../datasets/new_feature_ljh/new_feature_wh_cvr_full_train.csv',
                                                                '../../datasets/new_feature_ljh/new_feature_wh_cvr_full_test2.csv',
                                                                '../../datasets/new_feature_ljh/new_feature_wh_ratio_full_train.csv',
                                                                '../../datasets/new_feature_ljh/new_feature_wh_ratio_full_test2.csv'
                                                                )

train_data = pd.concat([train_data, new_fea_lisan_train], axis=1)
test_data = pd.concat([test_data, new_fea_lisan_test], axis=1)
del new_fea_lisan_train
del new_fea_lisan_test

import gc
gc.collect()

print('get the feature index...')
get_all_index(train_data, test_data, train_label)



