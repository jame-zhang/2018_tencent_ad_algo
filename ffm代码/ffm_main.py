import xlearn as xl
import pandas as pd
import numpy as np
import ffm_data_preprocess_git
import gen_new_static_fea

data = gen_new_static_fea.get_raw_data('./datasets/split_data.csv')

print('get new feature...')
# ad_fea = ['aid']
# user_fea = ['age', 'gender', 'consumptionAbility']

# new_features = None
# static_fea = gen_new_static_fea.get_statistic_feature(data)
# print('static_fea shape: ', static_fea.shape)
# print(static_fea.head())
# cross_fea = gen_new_static_fea.get_cross_fea(data, ad_fea, user_fea)
# print('cross_fea shape: ', cross_fea.shape)
# new_features = pd.concat([new_features, static_fea], axis=1)
# new_features = pd.concat([new_features, cross_fea], axis=1)
# new_features.to_csv('./datasets/new_feature/new_features_false_labels.csv', index=False)
# print(new_features.head())

one_hot_feature=['creativeSize', 'LBS','age','carrier','consumptionAbility','education','gender', 'houst', 'advertiserId','campaignId', 'creativeId',
       'adCategoryId', 'productId', 'productType',]

# one_hot_new = ['ratio_clicked_lisan', 'num_ad_clicked_lisan', 'num_ad_touser_lisan', 'ratio_common_interest1_lisan',
#                'ratio_common_interest2_lisan', 'ratio_common_interest5_lisan', 'aid_age_number_lisan', 'aid_age_ratio_lisan',
#                'aid_gender_number_lisan', 'aid_gender_ratio_lisan',
#                'aid_consumptionAbility_number_lisan', 'aid_consumptionAbility_ratio_lisan']

one_hot_new = ['num_ad_touser_lisan']

one_hot_feature = one_hot_feature + one_hot_new

vector_feature=['interest1','interest2','interest5','kw1','kw2','topic1','topic2','os','ct','marriageStatus']

continus_feature = []

# new_one_hot_feature = ['aid_age_lisan', 'aid_gender_lisan', 'aid_consumptionAbility_lisan']

# one_hot_feature = one_hot_feature + new_one_hot_feature

# continus_feature=['creativeSize']

# print('read raw data...')
# data = pd.read_csv('./datasets/split_data.csv')
print('read new feature data...')
new_fea = pd.read_csv('./datasets/new_feature/new_features.csv')

data = pd.concat([data, new_fea], axis=1)
print(data.head())
label = data['label']
# data=data.fillna(-1)
data = data[one_hot_feature+vector_feature]

tr = ffm_data_preprocess_git.FFMFormat(vector_feature, one_hot_feature, continus_feature)
user_ffm = tr.fit_transform(data)
user_ffm.to_csv('./datasets/ffm_all_git_filter10_addfea2.csv', index=False)
len_train = round((len(user_ffm) * 0.78))
X_train = user_ffm[:len_train]
X_eval = user_ffm[len_train:]

with open('./datasets/ffm_all_git_filter10_addfea2.csv') as fin:
    f_train_out=open('./datasets/ffm_train_git_filter10_addfea2.csv', 'w')
    f_test_out = open('./datasets/ffm_valid_git_filter10_addfea2.csv', 'w')
    for (i,line) in enumerate(fin):
        if i < len_train:
            f_train_out.write(str(label.values.squeeze()[i])+' '+line)
        else:
            f_test_out.write(str(label.values.squeeze()[i])+' '+line)
    f_train_out.close()
    f_test_out.close()


path='./datasets/'
ffm_model = xl.create_ffm()
ffm_model.disableEarlyStop()
ffm_model.setTrain(path+'ffm_train_git_filter10_addfea2.csv')
ffm_model.setValidate(path+'ffm_valid_git_filter10_addfea2.csv')
param = {'task':'binary', 'lr':0.01, 'lambda':0.001,'metric': 'auc','opt':'ftrl','epoch':10,'k':4,
         'alpha': 1.5, 'beta': 0.01, 'lambda_1': 0.0, 'lambda_2': 0.0}
ffm_model.fit(param, "./datasets/model_addfea2.out")
# ffm_model.predict("./model.out","./output.txt")
# sub = pd.DataFrame()
# sub['aid']=test_df['aid']
# sub['uid']=test_df['uid']
# sub['score'] = np.loadtxt("./output.txt")
# sub.to_csv('submission.csv',index=False)
# os.system('zip baseline_ffm.zip submission.csv')

pd.qcut()