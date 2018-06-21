#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/6 下午7:58
# @Author  : Jame
# @Site    : 
# @File    : feature_engineering.py
# @Software: PyCharm


import pandas as pd

path1= r""
path2= r""

train = pd.read_csv(path1)
test1 = pd.read_csv(path2)

train.loc[train['label']==-1,'label'] = 0
test1['label'] = -1

data = pd.concat([train,test1],ignore_index=True)
data = data.fillna("-1")


train = data[data.label != -1]
data_clicked = train[train['label'] == 1]



train = data[data.label != -1]
data_clicked = train[train['label'] == 1]


#jiahui.liang add_feature_mode6


# 增加每个广告推送给不同的用户数

print('开始加入广告推送给不同用户的数特征')

num_advertise_touser = train.groupby('aid').uid.nunique()
num_advertise_touser = pd.DataFrame({
    'aid': num_advertise_touser.index,
    'num_advertise_touser' : num_advertise_touser.values
})
data = pd.merge(data, num_advertise_touser, on=['aid'], how='left')

# 每个LBS投放的广告数
num_advertise_LBS = train.groupby('LBS').aid.nunique()
num_advertise_LBS = pd.DataFrame({
    'LBS': num_advertise_LBS.index,
    'num_advertise_LBS' : num_advertise_LBS.values
})
data = pd.merge(data, num_advertise_LBS, on=['LBS'], how='left')

# 获取每个用户的兴趣总数
train_all_interest = data[['interest1', 'interest2', 'interest5', 'kw1', 'kw2', 'topic1', 'topic2']]
num_all_interest_train = []
aids = []
train_array = np.array(train_all_interest)

for i in range(train_array.shape[0]):
    num = 0
    inter = train_array[i]
    for j in inter:
        inter_lis = j.split(' ')
        if inter_lis[0] == '-1':
            continue
        num += len(inter_lis)
    num_all_interest_train.append(num)
num_all_interest_train = pd.DataFrame(num_all_interest_train, columns=['num_all_interest'])

data = pd.concat([data, num_all_interest_train], axis=1)

# 增加各种兴趣ID，和主题ID的比例值

def get_common_interest(type_name, ratio):
    num_adid = data_clicked['aid'].value_counts().sort_index().index
    num_aid_clicked = dict(data_clicked['aid'].value_counts().sort_index())
    num_user_clicksameAd_interest = data_clicked.groupby('aid')[type_name].value_counts()
    dict_interest = {}
    for adid in num_adid:
        dict_buf = {}
        for interest in num_user_clicksameAd_interest.items():
            index = interest[0]
            if index[0] == adid:
                number = interest[1]
                detail = index[1]
                detail = detail.split(' ')
                for det in detail:
                    if det not in dict_buf:
                        dict_buf[det] = number
                    else:
                        dict_buf[det] += number
        dict_interest[adid] = dict_buf
    dict_common_interest = []
    for adid, dict_inter in dict_interest.items():
        dict_common_buf = {}
        dict_common_buf['aid'] = adid
        common_inter = []
        ad_total = num_aid_clicked[adid] - dict_inter.get('-1', 0)
        if '-1' in dict_inter:
            dict_inter.pop('-1')
        for id_inter, num in dict_inter.items():
            if num >= ad_total*ratio:
                common_inter.append(id_inter)
        str_name = 'common_'+type_name
        dict_common_buf[str_name] = common_inter
        dict_common_interest.append(dict_common_buf)
    return dict_common_interest


# 获取相同的兴趣ID2
print('开始加入兴趣ID2')
dict_common_interest2 = get_common_interest('interest2', 0.1)
df_common_interest2 = pd.DataFrame(dict_common_interest2)
data = pd.merge(data, df_common_interest2, on=['aid'], how='left')
data['num_common_interest2'] = [len(set(i.split(' ')).intersection(set(j))) / (len(j)+1) for i, j in data[['interest2', 'common_interest2']].values]

# 获取相同的兴趣ID1
print('开始加入兴趣ID1')
dict_common_interest1 = get_common_interest('interest1', 0.1)
df_common_interest1 = pd.DataFrame(dict_common_interest1)
data = pd.merge(data, df_common_interest1, on=['aid'], how='left')
data['num_common_interest1'] = [len(set(i.split(' ')).intersection(set(j))) / (len(j)+1) for i, j in data[['interest1', 'common_interest1']].values]

# 获取相同的兴趣ID5
print('开始加入兴趣ID5')
dict_common_interest5 = get_common_interest('interest5', 0.1)
df_common_interest5 = pd.DataFrame(dict_common_interest5)
data = pd.merge(data, df_common_interest5, on=['aid'], how='left')
data['num_common_interest5'] = [len(set(i.split(' ')).intersection(set(j))) / (len(j)+1) for i, j in data[['interest5', 'common_interest5']].values]

# 获取相同的主题1
print('开始加入主题1')
dict_common_topic1 = get_common_interest('topic1', 0.1)
df_common_topic1 = pd.DataFrame(dict_common_topic1)
data = pd.merge(data, df_common_topic1, on=['aid'], how='left')
data['num_common_topic1'] = [len(set(i.split(' ')).intersection(set(j))) / (len(j)+1) for i, j in data[['topic1', 'common_topic1']].values]


# 增加广告对应的年龄分布，消费能力分布，是否有房分布

def get_ad_toother(typename):
    num_ad_totype = train.groupby('aid')[typename].value_counts()
    num_ad_totype_clicked = data_clicked.groupby('aid')[typename].value_counts()
    ratio_num_ad_totype = num_ad_totype_clicked / num_ad_totype
    list_num_ad_totype = []
    num_adid = train['aid'].value_counts().sort_index().index
    for aid_out in num_adid:
        dict_buf = {}
        dict_num_ad_totype = {}
        dict_num_ad_totype['aid'] = aid_out
        for i, j in ratio_num_ad_totype.items():
            aid = i[0]
            feature = i[1]
            if(aid == aid_out):
                dict_buf[feature] = float("%.5f" % j)
        fea_name = 'num_ad_to'+typename
        dict_num_ad_totype[fea_name] = dict_buf
        list_num_ad_totype.append(dict_num_ad_totype)
    return list_num_ad_totype


print('开始加入年龄分布!')
list_num_ad_toage = get_ad_toother('age')
list_num_ad_toage = pd.DataFrame(list_num_ad_toage)
data = pd.merge(data, list_num_ad_toage, on=['aid'], how='left')
data['ratio_num_ad_toage'] = [j.get(i, 0) for i, j in data[['age', 'num_ad_toage']].values]

print('开始加入消费能力分布!')
list_num_ad_toconsume = get_ad_toother('consumptionAbility')
list_num_ad_toconsume = pd.DataFrame(list_num_ad_toconsume)
data = pd.merge(data, list_num_ad_toconsume, on=['aid'], how='left')
data['ratio_num_ad_toconsume'] = [j.get(i, 0) for i, j in data[['consumptionAbility', 'num_ad_toconsumptionAbility']].values]

print('开始加入是否有房分布!')
list_num_ad_tohouse = get_ad_toother('house')
list_num_ad_tohouse = pd.DataFrame(list_num_ad_tohouse)
data = pd.merge(data, list_num_ad_tohouse, on=['aid'], how='left')
data['ratio_num_ad_tohouse'] = [j.get(i, 0) for i, j in data[['house', 'num_ad_tohouse']].values]



