import numpy as np
import pandas as pd
import os
from scipy import sparse
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import pickle



one_hot_feature=['creativeSize', 'LBS','age','carrier','consumptionAbility', 'ct', 'education','gender','house','os','marriageStatus','advertiserId','campaignId', 'creativeId',
       'adCategoryId', 'productId', 'productType']
vector_feature=['interest1','interest2','interest5','kw1','kw2','topic1','topic2']


def get_raw_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    train_y = pd.read_csv('../datasets/train_label.csv', names=['label'])  # 直接读入label文件，省去了读取train源文件时间
    train_y[train_y == -1] = 0
    train_y = train_y.values.squeeze()
    return train_data, train_y, test_data


def get_all_index(train_data, test_data, label):
    result_train = {}
    result_test = {}
    ffm_train = pd.DataFrame()
    ffm_test = pd.DataFrame()
    feature_size = []
    print('get feature index...')
    for col in one_hot_feature:
        idx = 0
        col_value = train_data[col].unique()
        feature_size.append(len(col_value)+1)
        feat_dict = dict(zip(col_value, range(idx, len(col_value))))
        se_train = train_data[col].apply(lambda x: feat_dict[x])
        se_test = test_data[col].apply(lambda x: feat_dict.get(x, len(col_value)))
        ffm_train = pd.concat([ffm_train, se_train], axis=1)
        ffm_test = pd.concat([ffm_test, se_test], axis=1)

    result_train['index'] = ffm_train.values.tolist()
    result_test['index'] = ffm_test.values.tolist()
    result_train['label'] = label
    result_train['feature_size'] = feature_size

    ct_encoder = CountVectorizer(min_df=0.0009)
    train_interest = {}
    test_interest = {}
    feature_size_interest = []
    print('get interest index...')
    for index, col in enumerate(vector_feature):
        print('processing ', col)
        ct_encoder.fit(train_data[col])
        col_value = ct_encoder.vocabulary_
        feature_size_interest.append(len(col_value) + 2)
        train_se = []
        test_se = []
        for m in train_data[col]:
            ses = set()
            for n in m.split(' '):
                if n in col_value:
                    se = col_value[n]
                else:
                    se = len(col_value)
                ses.add(se)
            train_se.append(list(ses))
        for m in test_data[col]:
            ses = set()
            for n in m.split(' '):
                if n in col_value:
                    se = col_value[n]
                else:
                    se = len(col_value)
                ses.add(se)
            test_se.append(list(ses))
        train_interest[index] = train_se
        test_interest[index] = test_se
    result_train['interest'] = train_interest
    result_train['feature_size_interest'] = feature_size_interest
    result_test['interest'] = test_interest

    print('saving data...')
    pickle.dump(result_train, open('../datasets/deepmodel_train', 'wb'))
    pickle.dump(result_test, open('../datasets/deepmodel_test', 'wb'))

print('read the raw data...')
train_data, train_label, test_data = get_raw_data('../datasets/train_raw_data.csv', '../datasets/test2_raw_data.csv')

print('get the feature index...')
get_all_index(train_data, test_data, train_label)



