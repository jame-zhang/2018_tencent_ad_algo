# coding=utf-8
# @author:bryan
# blog: https://blog.csdn.net/bryan__
# github: https://github.com/YouChouNoBB/2018-tencent-ad-competition-baseline
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os

# ad_feature=pd.read_csv('./data/adFeature.csv')
# if os.path.exists('./data/userFeature.csv'):
#     user_feature=pd.read_csv('./data/userFeature.csv')
# else:
#     userFeature_data = []
#     with open('./data/userFeature.data', 'r') as f:
#         for i, line in enumerate(f):
#             line = line.strip().split('|')
#             userFeature_dict = {}
#             for each in line:
#                 each_list = each.split(' ')
#                 userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
#             userFeature_data.append(userFeature_dict)
#             if i % 100000 == 0:
#                 print(i)
#         user_feature = pd.DataFrame(userFeature_data)
#         user_feature.to_csv('./data/userFeature.csv', index=False)
# train=pd.read_csv('./data/train.csv')
# predict=pd.read_csv('./data/test1.csv')
# train.loc[train['label']==-1,'label']=0
# predict['label']=-1
# data=pd.concat([train,predict])
# data=pd.merge(data,ad_feature,on='aid',how='left')
# data=pd.merge(data,user_feature,on='uid',how='left')
# data=data.fillna('-1')
# one_hot_feature=['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','marriageStatus','advertiserId','campaignId', 'creativeId',
#        'adCategoryId', 'productId', 'productType']
# vector_feature=['appIdAction','appIdInstall','interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3']
# for feature in one_hot_feature:
#     try:
#         data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
#     except:
#         data[feature] = LabelEncoder().fit_transform(data[feature])
#
# train=data[data.label!=-1]
# train_y=train.pop('label')
# # train, test, train_y, test_y = train_test_split(train,train_y,test_size=0.2, random_state=2018)
# test=data[data.label==-1]
# res=test[['aid','uid']]
# test=test.drop('label',axis=1)
# enc = OneHotEncoder()
# train_x=train[['creativeSize']]
# test_x=test[['creativeSize']]
#
# for feature in one_hot_feature:
#     enc.fit(data[feature].values.reshape(-1, 1))
#     train_a=enc.transform(train[feature].values.reshape(-1, 1))
#     test_a = enc.transform(test[feature].values.reshape(-1, 1))
#     train_x= sparse.hstack((train_x, train_a))
#     test_x = sparse.hstack((test_x, test_a))
# print('one-hot prepared !')
#
# cv=CountVectorizer()
# for feature in vector_feature:
#     cv.fit(data[feature])
#     train_a = cv.transform(train[feature])
#     test_a = cv.transform(test[feature])
#     train_x = sparse.hstack((train_x, train_a))
#     test_x = sparse.hstack((test_x, test_a))
# print('cv prepared !')
#
# def LGB_predict(train_x,train_y,test_x,res):
#     print("LGB test")
#     clf = lgb.LGBMClassifier(
#         boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
#         max_depth=-1, n_estimators=10000, objective='binary',
#         subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
#         learning_rate=0.1, min_child_weight=50, random_state=2018, n_jobs=-1
#     )
#     clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='auc',early_stopping_rounds=100)
#     res['score'] = clf.predict_proba(test_x)[:,1]
#     res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
#     res.to_csv('./data/submission.csv', index=False)
#     os.system('zip ./data/baseline.zip ./data/submission.csv')
#     return clf
#
# model=LGB_predict(train_x,train_y,test_x,res)

train_x=sparse.load_npz('train_x.npz')
evals_x=sparse.load_npz('evals_x.npz')
res=pd.read_csv('res.csv')
train_y=pd.read_csv('train_y.csv')
evals_y=pd.read_csv('evals_y.csv')
test_x=sparse.load_npz('test_x.npz')
print("LGB test")
clf = lgb.LGBMClassifier(
    boosting_type='gbdt', num_leaves=38, reg_alpha=0.0, reg_lambda=0.9,
    max_depth=-1, n_estimators=15000, objective='binary',
    subsample=0.8, colsample_bytree=0.7, subsample_freq=1,
    learning_rate=0.002, min_child_weight=40, random_state=2018, n_jobs=-1
)
clf.fit(train_x, train_y, eval_set=[(evals_x, evals_y)], eval_metric='auc',early_stopping_rounds=20)
res['score'] = clf.predict_proba(test_x)[:,1]
res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
res.to_csv('./submission.csv', index=False)
os.system('zip ./baseline2.zip ./submission.csv')
