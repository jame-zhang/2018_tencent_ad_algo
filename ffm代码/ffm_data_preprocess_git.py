import pandas as pd
from pandas import get_dummies
import lightgbm as lgb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import numpy as np
import os
import gc
from time import  time

path='./datasets/'


one_hot_feature=['LBS','age', 'carrier', 'consumptionAbility','education','gender','advertiserId','campaignId', 'creativeId',
       'adCategoryId', 'productId', 'productType', 'creativeSize']

vector_feature = ['interest1', 'interest2', 'interest5','kw1','kw2','topic1','topic2','os','ct','marriageStatus']

continus_feature = []

# ad_feature=pd.read_csv(path+'adFeature.csv')
# user_feature=pd.read_csv(path+'userFeature.csv')

# train = pd.read_csv(path+'ffm_train.csv')
# valid = pd.read_csv(path+'ffm_valid.csv')

# data=pd.concat([train,test])
# data=pd.merge(data,ad_feature,on='aid',how='left')
# data=pd.merge(data,user_feature,on='uid',how='left')

# print('read csv...')
# data = pd.read_csv('./datasets/split_data.csv')
# label = data['label']
# data=data.fillna(-1)
# data = data[one_hot_feature+vector_feature]

class FFMFormat:
    def __init__(self,vector_feat,one_hot_feat,continus_feat):
        self.field_index_ = None
        self.feature_index_ = None
        self.vector_feat=vector_feat
        self.one_hot_feat=one_hot_feat
        self.continus_feat=continus_feat


    def get_params(self):
        pass

    def set_params(self, **parameters):
        pass

    def fit(self, df, y=None):
        self.field_index_ = {col: i for i, col in enumerate(df.columns)}
        self.feature_index_ = dict()
        last_idx = 0
        for col in df.columns:
            if col in self.one_hot_feat:
                print(col)
                df[col]=df[col].astype('int')
                vals = np.unique(df[col])
                for val in vals:
                    if val==-1: continue
                    name = '{}_{}'.format(col, val)
                    if name not in self.feature_index_:
                        self.feature_index_[name] = last_idx
                        last_idx += 1
            elif col in self.vector_feat:
                print(col)
                # vals=[]
                dict_ids_num = {}
                for data in df[col].apply(str):
                    for word in data.strip().split(' '):
                        if word in dict_ids_num:
                            dict_ids_num[word] += 1
                        else:
                            dict_ids_num[word] = 1
                dict_ids_num = {i: vals for i, vals in dict_ids_num.items() if vals >= 10}
                # vals = np.unique(vals)
                for val in dict_ids_num.keys():
                    # if val == "-1": continue
                    name = '{}_{}'.format(col, val)
                    if name not in self.feature_index_:
                        self.feature_index_[name] = last_idx
                        last_idx += 1
            self.feature_index_[col] = last_idx
            last_idx += 1
        return self

    def fit_transform(self, df, y=None):
        self.fit(df, y)
        print('start transform...')
        time1 = time()
        result = self.transform(df)
        print('transform cost: s'.format(time()-time1))
        return result

    def transform_row_(self, row):
        ffm = []

        for col, val in row.loc[row != 0].to_dict().items():
            if col in self.one_hot_feat:
                name = '{}_{}'.format(col, val)
                if name in self.feature_index_:
                    ffm.append('{}:{}:1'.format(self.field_index_[col], self.feature_index_[name]))
                # ffm.append('{}:{}:{}'.format(self.field_index_[col], self.feature_index_[col], 1))
            elif col in self.vector_feat:
                count = 0
                for word in str(val).split(' '):
                    name = '{}_{}'.format(col, word)
                    if name in self.feature_index_:
                        ffm.append('{}:{}:1'.format(self.field_index_[col], self.feature_index_[name]))
                    elif count == 0:
                        ffm.append('{}:{}:1'.format(self.field_index_[col], self.feature_index_[col]))
                        count = 1
            elif col in self.continus_feat:
                if val != -1:
                    ffm.append('{}:{}:{}'.format(self.field_index_[col], self.feature_index_[col], val))
        return ' '.join(ffm)

    def transform(self, df):
        # val=[]
        # for k,v in self.feature_index_.items():
        #     val.append(v)
        # val.sort()
        # print(val)
        # print(self.field_index_)
        # print(self.feature_index_)
        result = dict()
        for idx, row in df.iterrows():
            result[idx] = self.transform_row_(row)
            if idx % 500000 == 0:
                print('processed 50w')
        return pd.Series(result)
        # return pd.Series({idx: self.transform_row_(row) for idx, row in df.iterrows()})

# tr = FFMFormat(vector_feature,one_hot_feature,continus_feature)
# user_ffm=tr.fit_transform(data)
# user_ffm.to_csv('./datasets/ffm_all_git.csv', index=False)

# train = pd.read_csv(path + 'train.csv')
# test = pd.read_csv(path+'test1.csv')
#
# Y = np.array(train.pop('label'))
# len_train=len(train)

# len_train = round((len(user_ffm) * 0.78))
# X_train = user_ffm[:len_train]
# X_eval = user_ffm[len_train:]
#
# with open('./datasets/ffm_all_git.csv') as fin:
#     f_train_out=open('./datasets/ffm_train_git.csv', 'w')
#     f_test_out = open('./datasets/ffm_valid_git.csv', 'w')
#     for (i,line) in enumerate(fin):
#         if i<len_train:
#             f_train_out.write(str(label.values.squeeze()[i])+' '+line)
#         else:
#             f_test_out.write(str(label.values.squeeze()[i])+' '+line)
#     f_train_out.close()
#     f_test_out.close()
