#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/8 上午8:41
# @Author  : Jame
# @Site    : 
# @File    : add3_jiahui.py
# @Software: PyCharm

import pandas as pd
import tqdm
import gc


train_path = r"/home/ub36192/tencent_final/datasets/merge/full_train.csv"
# valid_path = r""
# test1_path = r""
test2_path = r"/home/ub36192/tencent_final/datasets/merge/full_test2.csv"
train_save_path = r"/home/ub36192/tencent_final/datasets/merge/full_train_add3.csv"
test2_save_path = r"/home/ub36192/tencent_final/datasets/merge/full_test2_add3.csv"


#full_test有testid
#full_train和full_test1以及和full_test2没有testid

# merge train dataset
data = pd.read_csv(train_path)
data["n_parts"] = 0
# merge validation dataset
# temp = pd.read_csv(valid_path)
# temp["n_parts"] = 1
# data = pd.concat([data,temp])

#统一处理label，merge是已处理过
data.loc[data['label']==-1,'label']=0
data.loc[data['label']==0.0,'label']=0

# merge test1 dataset
# temp = pd.read_csv(test1_path)
# temp["n_parts"] = 6
# temp['label'] = -1
# data = pd.concat([data,temp])

# merge test2 dataset
temp = pd.read_csv(test2_path)
temp["n_parts"] = 2
data = pd.concat([data,temp])
del temp
gc.collect()

f = open("add3.log","w")
#缺失值处理--删除缺失率高的特征: appIdAction, interest4, interest3,kw3, appIdInstall, appIdAction, topic3
data = data.drop(['appIdAction', 'interest4', 'interest3','kw3', 'appIdInstall', 'appIdAction', 'topic3'],axis=1)

one_hot_feature=['creativeSize', 'LBS','age','carrier','consumptionAbility','education','gender','house','os','marriageStatus','advertiserId','campaignId', 'creativeId',
       'adCategoryId', 'productId', 'productType']

vector_feature=['interest1','interest2','interest5','kw1','kw2','topic1','topic2']
# print('LabelEncoder start!')
# f.write("LabelEncoder start!")
# f.flush()
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

#
# # 加入组合特征，Jame
# print('开始加入相关组合特征')
# f.write("开始加入相关组合特征")
# f.flush()
# data['cross_feature'] = data["age"].astype(str)+pd.Series(["_"]*len(data))+data["education"]+"_"+data["consumptionAbility"]+"_"+data["consumptionAbility"]
# "_"+data["productId"]+"_"+data["productType"]



#写入结果
temp = data[data["n_parts"] == 0]
temp = temp.drop("n_parts",axis=1)
temp.to_csv(train_save_path)
print("训练集写入完成！")
temp = data[data["n_parts"] == 2]
temp = temp.drop("n_parts",axis=1)
temp.to_csv(test2_save_path)
print("测试集写入完成！")
f.close()