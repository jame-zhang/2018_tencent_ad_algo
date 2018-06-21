import pandas as pd

from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import warnings
import time


col_new = ['cvr_of_aid_and_age',
 'cvr_of_aid_and_gender',
 'cvr_of_aid_and_consumptionAbility',
'cvr_of_aid_and_education',
'cvr_of_aid_and_house',
'cvr_of_aid_and_LBS',
'cvr_of_aid',
'cvr_of_creativeSize',
'cvr_of_creativeSize_and_LBS',
'cvr_of_creativeSize_and_gender',
 'cvr_of_creativeSize_and_productType',

 'cvr_of_uid_and_creativeSize',
 'cvr_of_uid_and_productType',
 'cvr_of_uid_and_productId',
'cvr_of_uid_and_advertiserId',

 'cvr_of_advertiserId_and_LBS',
 'cvr_of_consumptionAbility_and_os',
 'cvr_of_advertiserId_and_creativeSize',
 'cvr_of_productType',
 'cvr_of_advertiserId',
 'cvr_of_productType_and_gender',
 'cvr_of_age_and_consumptionAbility',
 'cvr_of_creativeSize_and_consumptionAbility',
]

warnings.filterwarnings("ignore")
##读取数据
print("read data...")
time_start = time.time()
train_data = pd.read_csv('/home/ubuntu/tencent_final/code/Jame/feature_engineering/offline_datasets/sub_train_0.1.csv')
train_data['label'] = train_data['label'].map(lambda x: int(float(x)))
test_data = pd.read_csv('/home/ubuntu/tencent_final/code/Jame/feature_engineering/offline_datasets/sub_valid_0.1.csv')
test_data['label'] = test_data['label'].apply(lambda x: -1)
data = pd.concat([train_data, test_data], ignore_index=True)
print('read data cost: %s ' % (time.time()-time_start))

average_size = int(len(train_data)/5)
n_parts = pd.DataFrame()
for i in range(1, 6):
    buf_parts = pd.DataFrame()
    start = (i-1) * average_size
    if i < 5:
        end = start + average_size
    else:
        end = len(data)
    buf_parts['n_parts'] = [i]*(end-start)
    n_parts = pd.concat([n_parts, buf_parts], axis=0)

buf_parts = pd.DataFrame()
buf_parts['n_parts'] = [6]*len(test_data)
n_parts = pd.concat([n_parts, buf_parts], axis=0)
n_parts = n_parts.reset_index(drop=True)
data = pd.concat([data, n_parts], axis=1)

print('Index...')
# train_part_index = list(data[(data['label']!=-1)&(data['n_parts']!=1)].index)
# evals_index = list(data[(data['label']!=-1)&(data['n_parts']==1)].index)
test2_index = list(data[data['n_parts']==6].index)
print('LabelEncoder...')
label_feature=['aid','uid', 'advertiserId', 'campaignId', 'creativeId',
       'creativeSize', 'adCategoryId', 'productId', 'productType', 'age',
       'gender','education', 'consumptionAbility', 'LBS',
       'os', 'carrier', 'house']
for feature in label_feature:
    s = time.time()
    try:
        data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
    except:
        data[feature] = LabelEncoder().fit_transform(data[feature])
    print(feature,int(time.time()-s),'s')
print('Done')
col_type = label_feature.copy()
label_feature.append('label')
label_feature.append('n_parts')
data = data[label_feature]
df_feature = pd.DataFrame()
data['cnt']=1
print('Begin stat...')
n_parts = 6
num = 0
for co in col_type:
    s = time.time()
    col_name = 'cvr_of_'+co
    if col_name in col_new:
        se = pd.Series()
        for i in range(n_parts):
            if i==0:
                df = data[data['n_parts']==i+1][[co]]
                stat = data[(data['n_parts']!=i+1)&(data['n_parts']<=5)][[co,'label']].groupby(co)['label'].mean()
                se = se.append(pd.Series(df[co].map(stat).values,index=df.index))
            elif i<=4 and 1<=i:
                df = data[data['n_parts']==i+1][[co]]
                stat = data[(data['n_parts']!=i+1)&(data['n_parts']<=5)&(data['n_parts']>=2)][[co,'label']].groupby(co)['label'].mean()
                se = se.append(pd.Series(df[co].map(stat).values,index=df.index))
            elif i>=5:
                df = data[data['n_parts']==i+1][[co]]
                stat = data[data['n_parts']<=5][[co,'label']].groupby(co)['label'].mean()
                se = se.append(pd.Series(df[co].map(stat).values,index=df.index))
        df_feature[col_name] = ((pd.Series(data.index).map(se)*9999)-400).fillna(value=-1).astype(int)
        num+=1
        print(num,col_name,int(time.time()-s),'s')
n = len(col_type)
for i in range(n):
    for j in range(n-i-1):
        s = time.time()
        col_name = 'cvr_of_'+col_type[i]+"_and_"+col_type[i+j+1]
        if col_name in col_new:
            se = pd.Series()
            for k in range(n_parts):
                if k==0:
                    stat = data[(data['n_parts']!=k+1)&(data['n_parts']<=5)].groupby([col_type[i],col_type[i+j+1]])['label'].mean()
                    dt = data[data['n_parts']==k+1][[col_type[i],col_type[i+j+1]]]
                    dt.insert(0,'index',list(dt.index))
                    dt = pd.merge(dt,stat.reset_index(),how='left',on=[col_type[i], col_type[i+j+1]])
                    se = se.append(pd.Series(dt['label'].values,index=list(dt['index'].values)))

                elif 1<=k and k<=4:
                    stat = data[(data['n_parts']!=k+1)&(data['n_parts']<=5)&(data['n_parts']>=2)].groupby([col_type[i],col_type[i+j+1]])['label'].mean()
                    dt = data[data['n_parts']==k+1][[col_type[i],col_type[i+j+1]]]
                    dt.insert(0,'index',list(dt.index))
                    dt = pd.merge(dt,stat.reset_index(),how='left',on=[col_type[i],col_type[i+j+1]])
                    se = se.append(pd.Series(dt['label'].values,index=list(dt['index'].values)))
                elif k>=5:
                    stat = data[data['n_parts']<=5].groupby([col_type[i],col_type[i+j+1]])['label'].mean()
                    dt = data[data['n_parts']==k+1][[col_type[i],col_type[i+j+1]]]
                    dt.insert(0,'index',list(dt.index))
                    dt = pd.merge(dt,stat.reset_index(),how='left',on=[col_type[i],col_type[i+j+1]])
                    se = se.append(pd.Series(dt['label'].values, index=list(dt['index'].values)))
            df_feature[col_name] = (pd.Series(data.index).map(se)*9999-400).fillna(value=-1).astype(int)
            num+=1
            print(num,col_name,int(time.time()-s),'s')


print('Saving...')
print('train_part...')
df_feature.iloc[:len(train_data)].to_csv('/home/ubuntu/tencent_final/code/Jame/feature_engineering/offline_datasets/new_feature_wh_sub_train.csv', index=False)

# print('evals...')
# df_feature.loc[evals_index].to_csv('evals_x_cvr_select.csv',index=False)

print('test2...')
df_feature.loc[test2_index].to_csv('/home/ubuntu/tencent_final/code/Jame/feature_engineering/offline_datasets/new_feature_wh_sub_valid.csv', index=False)
df_feature = pd.DataFrame()
print('Over')
