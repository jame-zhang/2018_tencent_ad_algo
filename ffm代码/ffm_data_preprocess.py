import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

one_hot_feature=['creativeSize', 'LBS','age','carrier', 'consumptionAbility','education','gender','house','os','marriageStatus','advertiserId','campaignId', 'creativeId',
       'adCategoryId', 'productId', 'productType', 'ct']
vector_feature=['interest1','interest2','interest5','kw1','kw2','topic1','topic2']

print('read csv...')
data = pd.read_csv('./datasets/split_data.csv')

print('len of split data: ', len(data))

label = data['label']
data = data[one_hot_feature+vector_feature]
# processing one hot feature
idx = 0
field_dict = dict(zip(one_hot_feature, range(len(one_hot_feature))))
ffm = pd.DataFrame()
for col in one_hot_feature:
    print('process', col)
    col_value = np.unique(data[col])
    feat_dict = dict(zip(col_value, range(idx, idx+len(col_value))))
    se = data[col].apply(lambda x: "{0}:{1}:{2}".format(field_dict[col]+1, feat_dict[x], 1))
    ffm = pd.concat([ffm,se], axis=1)
    idx += len(col_value)

print('processing one-hot feature finished!')

# processing count vector feature

field_index = len(one_hot_feature)
field_dict2 = dict(zip(vector_feature, range(len(vector_feature))))
# ct_encoder = CountVectorizer(min_df=0.0009)
# ct_encoder = CountVectorizer(min_df=10)  # 改变min_df大小
# ct_encoder = CountVectorizer()  # 不设定min_df

for col in vector_feature:
    print('process', col)
    # ct_encoder.fit(data[col])
    # col_value = ct_encoder.vocabulary_
    dict_ids_num = {}
    # feat_dict = {}
    for item in data[col]:
        for inter in item.strip().split(' '):
            if inter in dict_ids_num:
                dict_ids_num[inter] += 1
            else:
                dict_ids_num[inter] = 1
    dict_ids_num = {i: vals for i, vals in dict_ids_num.items() if vals >= 5}
    col_value = dict(zip(dict_ids_num.keys(), range(idx, idx + len(dict_ids_num))))
    # for k, v in col_value.items():
    #     feat_dict[k] = v+idx
    all_se = []
    for m in data[col]:
        count = 0
        ses = str()
        buf = {}
        for n in m.split(' '):
            if n in col_value:
                se = "{0}:{1}:{2}".format(field_dict2[col]+field_index+1, col_value[n], 1)
                ses = ses + se + ' '
            elif count == 0:
                se = "{0}:{1}:{2}".format(field_dict2[col] + field_index + 1, idx+len(col_value), 1)
                ses = ses + se + ' '
                count = 1
        buf[col] = ses
        all_se.append(buf)
    final = pd.DataFrame(all_se)
    ffm = pd.concat([ffm, final], axis=1)
    idx = idx + len(col_value) + 1

print('processing vector feature finished!')


# processing ct feature

# ct_train = data['ct'].values
# ct_train = [m.split(' ') for m in ct_train]
# dict_ct = {'0':0, '1':1, '2':2, '3':3, '4':4}
# all_ct = []
# for i in ct_train:
#     buf_dict = {}
#     ses = str()
#     for j in i:
#         se = "{0}:{1}:{2}".format(len(field_dict)+len(field_dict2)+1,dict_ct[j]+idx, 1)
#         ses = ses + se + ' '
#     buf_dict['ct'] = ses
#     all_ct.append(buf_dict)
#
# all_ct = pd.DataFrame(all_ct)
# ffm = pd.concat([ffm, all_ct], axis=1)

# add label to ffm data

ffm.insert(0, 'label', label)
print('save to csv...')
ffm.to_csv('./datasets/ffm_all_nomindf_filter5.csv', index=False)

# split train,val datasets

len_train = round((len(ffm) * 0.78))
X_train = ffm[:len_train]
X_eval = ffm[len_train:]

print('first item index: ', ffm.iloc[0])

def save_data_txt(path, raw_data):
    with open(path, 'w') as f:
        for i, data in enumerate(raw_data.values):
            for index, item in enumerate(data):
                if index <= 16:
                    f.write(str(item) + ' ')
                else:
                    f.write(item)
            if i < len(raw_data) - 1:
                f.write('\n')


print('save to text')

save_data_txt('./datasets/ffm_train_filter5.txt', X_train)
save_data_txt('./datasets/ffm_valid_filter5.txt', X_eval)
