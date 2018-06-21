#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/8 上午11:54
# @Author  : Jame
# @Site    : 
# @File    : data_merge_en_jiahui.liang.py
# @Software: PyCharm

import pandas as pd
import os
import gc
from tqdm import tqdm

basepath = "/home/ub36192/tencent_final/datasets/rawdata/final_competition_data/final_competition_data/"


def get_user_feature():
    if os.path.exists(basepath+'userFeature.csv'):
        user_feature=pd.read_csv(basepath+'userFeature.csv')
    else:
        userFeature_data = []
        with open(basepath+'userFeature.data', 'r') as f:
            print("get user_feature!!")
            for i, line in tqdm(enumerate(f)):
                line = line.strip().split('|')
                userFeature_dict = {}
                for each in line:
                    each_list = each.split(' ')
                    userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
                userFeature_data.append(userFeature_dict)
                if i % 100000 == 0:
                    print(i)
            user_feature = pd.DataFrame(userFeature_data)
            user_feature.to_csv(basepath+'userFeature.csv', index=False)
        gc.collect()
    return user_feature

def get_data():
    if os.path.exists(basepath+'data.csv'):
        return pd.read_csv(basepath+'data.csv')
    else:
        ad_feature = pd.read_csv(basepath+'adFeature.csv')
        train=pd.read_csv(basepath+'train.csv')
        predict=pd.read_csv(basepath+'test2.csv')
        train.loc[train['label']==-1,'label']=0
        predict['label']=-1
        user_feature=get_user_feature()
        data=pd.concat([train,predict])
        data=pd.merge(data,ad_feature,on='aid',how='left')
        data=pd.merge(data,user_feature,on='uid',how='left')
        data=data.fillna('-1')
        del user_feature
        return data

f = open("feature_engineering_jiahui.log","w")

data = get_data()



#缺失值处理--删除缺失率高的特征: appIdAction, interest4, interest3,kw3, appIdInstall, appIdAction, topic3
data = data.drop(['appIdAction', 'interest4', 'interest3','kw3', 'appIdInstall', 'appIdAction', 'topic3'],axis=1)

one_hot_feature=['creativeSize', 'LBS','age','carrier','consumptionAbility','education','gender','house','os','marriageStatus','advertiserId','campaignId', 'creativeId',
       'adCategoryId', 'productId', 'productType']

vector_feature=['interest1','interest2','interest5','kw1','kw2','topic1','topic2']
print('feature start!')
f.write("LabelEncoder start!")
f.flush()
# for feature in tqdm(one_hot_feature):
#     try:
#         data[feature] = LabelEncoder().fit_transform(data[feature].apply(str))
#     except:
#         data[feature] = LabelEncoder().fit_transform(data[feature])

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
temp = data[data["label"]!=-1]
temp.to_csv(basepath+"full_train_model_feature_jiahui.csv")
temp = data[data["label"] == -1]
temp.to_csv(basepath+"full_test2_model_feature_jiahui.csv")


gc.collect()
f.close()

