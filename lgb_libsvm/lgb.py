# coding=utf-8
# @author:walter000
# github: https://github.com/Walter000/tencent_competition


"""
 复赛训练数据集中缺失率较情况：
appIdAction	  0.984490
interest4	  0.984309
appIdInstall  0.979607
interest3	  0.973314
kw3	          0.954391
topic3     	  0.954274
house	      0.823856
interest2	  0.347845
interest5	  0.249349
kw1	          0.099491
interest1	  0.090264
topic1	      0.084911
topic2	      0.038272
kw2	          0.036616
LBS	          0.000004
 训练数据中剔除了缺失率过高的特征: appIdAction, interest4, interest3,kw3, appIdInstall, appIdAction, topic3, 同时将除了ct(上网类型)之外的
 特征进行相同的处理，额外增加了四个组合特征


"""

import numpy as np
import pandas as pd
import os
from scipy import sparse
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import lightgbm as lgb
import time
from tqdm import tqdm


# 下面根据前面的特征分析，剔除掉缺失率过高的特征: interest3, interest4, kw3, appIdInstall, topic3
base_path = './'

# #1 处理未合并的数据
# ad_feature=pd.read_csv('lgb_file/adFeature.csv')
# user_feature = pd.read_csv('lgb_file/userFeature.csv')
# train = pd.read_csv('lgb_file/train.csv')
# predict = pd.read_csv('lgb_file/test2.csv')
# train.loc[train['label']==-1,'label']=0
# predict['label']=-1
# data = pd.concat([train,predict])
# data = pd.merge(data,ad_feature,on='aid',how='left')
# data = pd.merge(data,user_feature,on='uid',how='left')
# data = data.fillna('-1')
f = open("lgb_file/log.txt","w")

#2 处理已合并的数据
train = pd.read_csv('/home/lab/Desktop/tencent_algo/final/Jame/data_preprocess/data_5fold/train_5fold_0.csv')
train = train.drop('testid',axis=1)
predict = pd.read_csv('/home/lab/Desktop/tencent_algo/final/Jame/full_test1.csv')
predict['label']=-1
data = pd.concat([train,predict])
data.fillna('-1')
print("load finish!")
f.write("load finish!")
f.flush()
#缺失值处理--删除缺失率高的特征: appIdAction, interest4, interest3,kw3, appIdInstall, appIdAction, topic3
data = data.drop(['appIdAction', 'interest4', 'interest3','kw3', 'appIdInstall', 'appIdAction', 'topic3'],axis=1)

one_hot_feature=['creativeSize', 'LBS','age','carrier','consumptionAbility','education','gender','house','os','marriageStatus','advertiserId','campaignId', 'creativeId',
       'adCategoryId', 'productId', 'productType']

vector_feature=['interest1','interest2','interest5','kw1','kw2','topic1','topic2']
print('LabelEncoder start!')
f.write("LabelEncoder start!")
f.flush()
for feature in tqdm(one_hot_feature):
    try:
        data[feature] = LabelEncoder().fit_transform(data[feature].apply(str))
    except:
        data[feature] = LabelEncoder().fit_transform(data[feature])

train = data[data.label != -1]
data_clicked = train[train['label'] == 1]

'''
# 增加广告点击率特征

print('开始加入广告点击率特征')

num_ad = train['aid'].value_counts().sort_index()
num_ad_clicked = data_clicked['aid'].value_counts().sort_index()

ratio = num_ad_clicked / num_ad

ratio_clicked = pd.DataFrame({
    'aid': ratio.index,
    'ratio_clicked' : ratio.values
})
data = pd.merge(data, ratio_clicked, on=['aid'], how='left')

'''
# 增加每个广告推送给不同的用户数

print('开始加入广告推送给不同用户的数特征')
f.write("开始加入广告推送给不同用户的数特征")
f.flush()

num_advertise_touser = train.groupby('aid').uid.nunique()
num_advertise_touser = pd.DataFrame({
    'aid': num_advertise_touser.index,
    'num_advertise_touser' : num_advertise_touser.values
})
data = pd.merge(data, num_advertise_touser, on=['aid'], how='left')

# 加入推广计划转化率

print('开始加入推广计划转化率特征')
f.write("开始加入推广计划转化率特征")
f.flush()

num_campaign = train['campaignId'].value_counts().sort_index()
num_campaign_clicked = data_clicked['campaignId'].value_counts().sort_index()
ratio_num_campaign = num_campaign_clicked / num_campaign
ratio_num_campaign = pd.DataFrame({
    'campaignId': ratio_num_campaign.index,
    'ratio_num_campaign' : ratio_num_campaign.values
})

data = pd.merge(data, ratio_num_campaign, on=['campaignId'], how='left')


# 加入学历所对应转化率

print('开始加入学历所对应转化率特征')
f.write("开始加入学历所对应转化率特征")
f.flush()
num_education = train['education'].value_counts().sort_index()
num_education_clicked = data_clicked['education'].value_counts().sort_index()
ration_num_education = num_education_clicked / num_education
ration_num_education = pd.DataFrame({
    'education': ration_num_education.index,
    'ration_num_education' : ration_num_education.values
})
data = pd.merge(data, ration_num_education, on=['education'], how='left')


# 分离测试集
train = data[data.label != -1]
test = data[data.label == -1]
res = test[['aid','uid']]
test = test.drop('label', axis=1)
train_y = train.pop('label')

# 处理联网类型特征
ct_train = train['ct'].values
ct_train = [m.split(' ') for m in ct_train]
ct_trains = []
for i in ct_train:
    index = [0, 0, 0, 0, 0]
    for j in i:
        index[int(j)] = 1
    ct_trains.append(index)

ct_test = test['ct'].values
ct_test = [m.split(' ') for m in ct_test]
ct_tests = []
for i in ct_test:
    index = [0, 0, 0, 0, 0]
    for j in i:
        index[int(j)] = 1
    ct_tests.append(index)


# 将上面新加入的特征进行归一化

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(train[['num_advertise_touser', 'ratio_num_campaign',
                          'ration_num_education']].values)
train_x = scaler.transform(train[['num_advertise_touser', 'ratio_num_campaign',
                                                  'ration_num_education']].values)

test_x = scaler.transform(test[['num_advertise_touser', 'ratio_num_campaign',
                                                  'ration_num_education']].values)
train_x = np.hstack((train_x, ct_trains))
test_x = np.hstack((test_x, ct_tests))


# 特征进行onehot处理
enc = OneHotEncoder()

oc_encoder = OneHotEncoder()
print("onehot start")
f.write("onehot start")
f.flush()
for feature in tqdm(one_hot_feature):
    oc_encoder.fit(data[feature].values.reshape(-1, 1))
    train_a=oc_encoder.transform(train[feature].values.reshape(-1, 1))
    test_a = oc_encoder.transform(test[feature].values.reshape(-1, 1))
    train_x = sparse.hstack((train_x, train_a))
    test_x = sparse.hstack((test_x, test_a))
print('one-hot prepared !')
f.write("one-hot prepared !")
f.flush()

# 处理count特征向量

ct_encoder = CountVectorizer(min_df=0.001, tokenizer = str.split)  #传递函数
print("CV start")
f.write("CV start")
f.flush()
for feature in tqdm(vector_feature):
    ct_encoder.fit(data[feature])
    train_a = ct_encoder.transform(train[feature])
    test_a = ct_encoder.transform(test[feature])
    train_x = sparse.hstack((train_x, train_a))
    test_x = sparse.hstack((test_x, test_a))
print('cv prepared !')
f.write("cv prepared !")
f.flush()
# print('ths shape of train data:', test_x.shape)

sparse.save_npz('lgb_file/model_fea_add1_train.npz', train_x)
sparse.save_npz('lgb_file/model_fea_add1_test.npz', test_x)


def LGB_predict(train_x, train_y, test_x, res):
    print("LGB test")
    f.write("LGB test")
    f.flush()
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=10000, objective='binary',
        subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
        learning_rate=0.02, min_child_weight=50, random_state=2018, n_jobs=-1
    )
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='auc',early_stopping_rounds=10)
    res['score'] = clf.predict_proba(test_x)[:, 1]
    res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
    res.to_csv('lgb_file/submission_model_fea_add1'+str(time.strftime("%Y%m%d-%H%M%S", time.localtime()))+'.csv', index=False)
    return clf


model = LGB_predict(train_x,train_y,test_x,res)
joblib.dump(model, 'model_fea_add1.model')