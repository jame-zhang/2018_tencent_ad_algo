import pandas as pd
from time import time
import gc
from sklearn.model_selection import KFold
import numpy as np

new_fea_dict = {}


def get_raw_data(train_path, test_path=None):
    time1 = time()
    print('read train data...')
    train_data = pd.read_csv(train_path)
    print('read test data...')
    test_data = pd.read_csv(test_path)
    # train_y = train_data['label'].values.squeeze()
    train_data = train_data.fillna('-1')
    test_data = test_data.fillna('-1')
    print('read raw data cost: {} s'.format(time()-time1))
    return train_data, test_data


def get_statistic_feature(train_data, test_data):

    """
        获取转化率特征并离散化, 广告点击率，兴趣占比，广告出现次数，广告点击次数，
    """
    time1 = time()

    df_new_fea = None
    df_new_fea_test = None
    # 加入广告转化率
    print('add ad clicked ratio')
    ratio_clicked = train_data.groupby('aid').label.agg(['sum', 'count']).reset_index()
    ratio_clicked['ratio_clicked'] = (ratio_clicked['sum'] + 0.0001) / (ratio_clicked['count'] + 0.0001)
    ratio_clicked_train = pd.merge(train_data, ratio_clicked, on=['aid'], how='left')[['ratio_clicked']]
    ratio_clicked_test = pd.merge(test_data, ratio_clicked, on=['aid'], how='left')[['ratio_clicked']]

    # ratio_clicked['ratio_clicked_lisan'] = pd.cut(ratio_clicked_train['ratio_clicked'], bins=10, labels=False)
    # ratio_clicked_test['ratio_clicked_lisan'] = pd.cut(ratio_clicked_test['ratio_clicked'], bins=10, labels=False)
    #
    # df_new_fea = pd.concat([df_new_fea, ratio_clicked_train[['ratio_clicked', 'ratio_clicked_lisan']]], axis=1)
    # df_new_fea_test = pd.concat([df_new_fea_test, ratio_clicked_test[['ratio_clicked', 'ratio_clicked_lisan']]], axis=1)

    df_new_fea = pd.concat([df_new_fea, ratio_clicked_train['ratio_clicked']], axis=1)
    df_new_fea_test = pd.concat([df_new_fea_test, ratio_clicked_test['ratio_clicked']], axis=1)

    # 加入广告点击次数
    print('add ad clicked num')
    num_clicked = train_data.groupby('aid').label.agg(['sum']).reset_index()
    num_clicked_train = pd.merge(train_data, num_clicked, on=['aid'], how='left')[['sum']]
    num_clicked_test = pd.merge(test_data, num_clicked, on=['aid'], how='left')[['sum']]

    num_clicked_train.columns = ['num_ad_clicked']
    num_clicked_test.columns = ['num_ad_clicked']

    # num_clicked_train['num_ad_clicked_lisan'] = pd.cut(num_clicked_train['num_ad_clicked'], bins=10, labels=False)
    # num_clicked_test['num_ad_clicked_lisan'] = pd.cut(num_clicked_test['num_ad_clicked'], bins=10, labels=False)

    df_new_fea = pd.concat([df_new_fea, num_clicked_train['num_ad_clicked']], axis=1)
    df_new_fea_test = pd.concat([df_new_fea_test, num_clicked_test['num_ad_clicked']], axis=1)

    # 加入广告推送给不同的用户数
    print('add ad num to uer')
    num_advertise_touser = train_data.groupby('aid').uid.nunique()
    num_advertise_touser = pd.DataFrame({
        'aid': num_advertise_touser.index,
        'num_ad_touser': num_advertise_touser.values
    })
    num_advertise_touser_train = pd.merge(train_data, num_advertise_touser, on=['aid'], how='left')[['num_ad_touser']]
    num_advertise_touser_test = pd.merge(test_data, num_advertise_touser, on=['aid'], how='left')[['num_ad_touser']]

    # num_advertise_touser_train['num_ad_touser_lisan'] = pd.cut(num_advertise_touser_train['num_ad_touser'], bins=10, labels=False)
    # num_advertise_touser_test['num_ad_touser_lisan'] = pd.cut(num_advertise_touser_test['num_ad_touser'], bins=10, labels=False)

    df_new_fea = pd.concat([df_new_fea, num_advertise_touser_train['num_ad_touser']], axis=1)
    df_new_fea_test = pd.concat([df_new_fea_test, num_advertise_touser_test['num_ad_touser']], axis=1)

    # 加入兴趣1，2，5和topic1的兴趣分布
    print('add common interest1')

    dict_common_interest1 = get_common_interest(train_data, 'interest1', 0.25)
    df_common_interest1 = pd.DataFrame(dict_common_interest1)
    train_data = pd.merge(train_data, df_common_interest1, on=['aid'], how='left')
    test_data = pd.merge(test_data, df_common_interest1, on=['aid'], how='left')

    ratio_common_inter1 = pd.DataFrame()
    ratio_common_inter1_test = pd.DataFrame()
    ratio_common_inter1['ratio_common_interest1'] = [(len(set(i.split(' ')).intersection(set(j))) + 0.0001) / (len(j) + 0.0001) for i, j in
                                       train_data[['interest1', 'common_interest1']].values]
    ratio_common_inter1_test['ratio_common_interest1'] = [(len(set(i.split(' ')).intersection(set(j))) + 0.0001) / (len(j) + 0.0001) for i, j in
                                       test_data[['interest1', 'common_interest1']].values]

    # ratio_common_inter1['ratio_common_interest1_lisan'] = pd.cut(ratio_common_inter1['ratio_common_interest1'], bins=10, labels=False)
    # ratio_common_inter1_test['ratio_common_interest1_lisan'] = pd.cut(ratio_common_inter1_test['ratio_common_interest1'], bins=10, labels=False)

    df_new_fea = pd.concat([df_new_fea, ratio_common_inter1['ratio_common_interest1']], axis=1)
    df_new_fea_test = pd.concat([df_new_fea_test, ratio_common_inter1_test['ratio_common_interest1']], axis=1)

    print('add common interest2')

    dict_common_interest2 = get_common_interest(train_data, 'interest2', 0.25)
    df_common_interest2 = pd.DataFrame(dict_common_interest2)
    train_data = pd.merge(train_data, df_common_interest2, on=['aid'], how='left')
    test_data = pd.merge(test_data, df_common_interest2, on=['aid'], how='left')

    ratio_common_inter2 = pd.DataFrame()
    ratio_common_inter2_test = pd.DataFrame()
    ratio_common_inter2['ratio_common_interest2'] = [(len(set(i.split(' ')).intersection(set(j))) + 0.0001) / (len(j) + 0.0001) for i, j in
                                                   train_data[['interest2', 'common_interest2']].values]
    ratio_common_inter2_test['ratio_common_interest2'] = [(len(set(i.split(' ')).intersection(set(j))) + 0.0001) / (len(j) + 0.0001) for i, j in
                                                   test_data[['interest2', 'common_interest2']].values]

    # ratio_common_inter2['ratio_common_interest2_lisan'] = pd.cut(ratio_common_inter2['ratio_common_interest2'], bins=10,
    #                                                            labels=False)
    # ratio_common_inter2_test['ratio_common_interest2_lisan'] = pd.cut(ratio_common_inter2_test['ratio_common_interest2'], bins=10,
    #                                                              labels=False)

    df_new_fea = pd.concat([df_new_fea, ratio_common_inter2['ratio_common_interest2']], axis=1)

    df_new_fea_test = pd.concat([df_new_fea_test, ratio_common_inter2_test['ratio_common_interest2']], axis=1)

    print('add common interest5')

    dict_common_interest5 = get_common_interest(train_data, 'interest5', 0.25)
    df_common_interest5 = pd.DataFrame(dict_common_interest5)
    train_data = pd.merge(train_data, df_common_interest5, on=['aid'], how='left')
    test_data = pd.merge(test_data, df_common_interest5, on=['aid'], how='left')

    ratio_common_inter5 = pd.DataFrame()
    ratio_common_inter5_test = pd.DataFrame()
    ratio_common_inter5['ratio_common_interest5'] = [(len(set(i.split(' ')).intersection(set(j))) + 0.0001) / (len(j) + 0.0001) for i, j in
                                                     train_data[['interest5', 'common_interest5']].values]
    ratio_common_inter5_test['ratio_common_interest5'] = [(len(set(i.split(' ')).intersection(set(j))) + 0.0001) / (len(j) + 0.0001) for i, j in
                                                     test_data[['interest5', 'common_interest5']].values]

    # ratio_common_inter5['ratio_common_interest5_lisan'] = pd.cut(ratio_common_inter5['ratio_common_interest5'], bins=10,
    #                                                              labels=False)
    # ratio_common_inter5_test['ratio_common_interest5_lisan'] = pd.cut(ratio_common_inter5_test['ratio_common_interest5'], bins=10,
    #                                                              labels=False)

    df_new_fea = pd.concat([df_new_fea, ratio_common_inter5['ratio_common_interest5']], axis=1)
    df_new_fea_test = pd.concat([df_new_fea_test, ratio_common_inter5_test['ratio_common_interest5']], axis=1)

    print('add common topic1')

    dict_common_topic1 = get_common_interest(train_data, 'topic1', 0.25)
    df_common_topic1 = pd.DataFrame(dict_common_topic1)
    train_data = pd.merge(train_data, df_common_topic1, on=['aid'], how='left')
    test_data = pd.merge(test_data, df_common_topic1, on=['aid'], how='left')

    ratio_common_topic1 = pd.DataFrame()
    ratio_common_topic1_test = pd.DataFrame()


    ratio_common_topic1['ratio_common_topic1'] = [(len(set(i.split(' ')).intersection(set(j))) + 0.0001) / (len(j) + 0.0001)for i, j in
                                                     train_data[['topic1', 'common_topic1']].values]

    ratio_common_topic1_test['ratio_common_topic1'] = [(len(set(i.split(' ')).intersection(set(j))) + 0.0001) / (len(j) + 0.0001)for i, j in
                                                     test_data[['topic1', 'common_topic1']].values]

    # ratio_common_topic1['ratio_common_topic1_lisan'] = pd.cut(ratio_common_topic1['ratio_common_topic1'], bins=10,
    #                                                              labels=range(1, 11))
    # ratio_common_topic1_test['ratio_common_topic1_lisan'] = pd.cut(ratio_common_topic1_test['ratio_common_topic1'], bins=10,
    #                                                              labels=range(1, 11))

    df_new_fea = pd.concat([df_new_fea, ratio_common_topic1['ratio_common_topic1']], axis=1)
    df_new_fea_test = pd.concat([df_new_fea_test, ratio_common_topic1_test['ratio_common_topic1']], axis=1)
    print('read raw data cost: {} s'.format(time() - time1))
    return df_new_fea, df_new_fea_test

    # X_test = pd.merge(X_test, ratio_clicked, on=['aid'], how='left')


def get_cross_fea(train_data, test_data, adFeat_list, userFeat_list):

    """
     获取交叉特征
    """
    time1 = time()
    df_cross_fea = None
    df_cross_fea_test = None
    for afeat in adFeat_list:
        for ufeat in userFeat_list:
            print('process :', afeat, ufeat)
            concat_feat = afeat + '_' + ufeat
            # 训练集
            # df_cv = train_data[[afeat, ufeat,  'label']].copy()
            # df_cv[concat_feat] = df_cv[afeat].astype('str') + '_' + df_cv[ufeat].astype('str')
            # df_cv = df_cv[[concat_feat,  'label']]
            # df_cv[concat_feat] = _remove_lowcase(df_cv[concat_feat])
            train_data[concat_feat] = train_data[afeat].astype('str') + '_' + train_data[ufeat].astype('str')
            train_data[concat_feat] = _remove_lowcase(train_data[concat_feat])
            df_cv = train_data[[concat_feat,  'label']]
            # 测试集
            test_data[concat_feat] = test_data[afeat].astype('str') + '_' + test_data[ufeat].astype('str')
            df_cv_test = test_data[[concat_feat]]
            # df_cv_test[concat_feat] = df_cv_test[afeat].astype('str') + '_' + df_cv_test[ufeat].astype('str')

            # 统计交叉特征次数
            counts = pd.DataFrame(df_cv[concat_feat].value_counts()).reset_index()
            counts.columns = [concat_feat, concat_feat+'_number']
            # counts[concat_feat+'_number_lisan'] = pd.cut(counts[concat_feat+'_number'], bins=10, labels=range(1, 11)).astype('str')
            df_cv = pd.merge(df_cv, counts, on=[concat_feat], how='left')
            # df_cv[concat_feat + '_number_lisan'] = pd.cut(df_cv[concat_feat+'_number'], bins=10, labels=False).astype('str')

            df_cv_test = pd.merge(df_cv_test, counts, on=[concat_feat], how='left')
            # df_cv_test[concat_feat + '_number_lisan'] = pd.cut(df_cv_test[concat_feat + '_number'], bins=10, labels=False).astype(
            #     'str')

            # df_cross_fea = pd.concat([df_cross_fea, df_cv[[concat_feat + '_number_lisan']]], axis=1)
            # training = df_cv[df_cv.label != -1]
            # training = training.reset_index(drop=True)
            # predict = df_cv[df_cv.label == -1]
            # del df_cv
            # gc.collect()
            df_stas_feat = None
            kf = KFold(n_splits=5, random_state=2018, shuffle=False)
            for train_index, val_index in kf.split(df_cv):
                X_train = df_cv.loc[train_index, :]
                X_val = df_cv.loc[val_index, :]
                X_val = _statis_feat(X_train, X_val, concat_feat)
                df_stas_feat = pd.concat([df_stas_feat, X_val], axis=0)
            # df_stas_feat[concat_feat+'_ratio_lisan'] = pd.cut(df_stas_feat[concat_feat + '_ratio'], 10, labels=False).astype('str')
            df_stas_feat = df_stas_feat.reset_index(drop=True)
            df_stas_feat_test = _statis_feat(train_data, df_cv_test, concat_feat)
            df_stas_feat_test = df_stas_feat_test.reset_index(drop=True)
            # df_stas_feat = pd.concat([df_stas_feat, X_pred], axis=0)
            # del df_stas_feat[ label]
            # del df_stas_feat[concat_feat]
            # del training
            # del predict
            # gc.collect()
            # df = pd.merge(df, df_stas_feat, how='left', on='index')
            # print(len(df_stas_feat))
            df_cross_fea = pd.concat([df_cross_fea, df_stas_feat[[concat_feat+'_number', concat_feat+'_ratio']]], axis=1)
            df_cross_fea_test = pd.concat([df_cross_fea_test, df_stas_feat_test[[concat_feat+'_number', concat_feat+'_ratio']]], axis=1)

            print(afeat, ufeat, 'done!')
    # del df['index']
    print('process cross fea cost: {} s'.format(time() - time1))
    return df_cross_fea, df_cross_fea_test


def _remove_lowcase(se):
    count = dict(se.value_counts())
    se = se.map(lambda x: -1 if count[x] < 20 else x)
    return se


def _statis_feat(df, df_val, feature):

    df['label'] = df['label'].replace(-1, 0)
    df = df.groupby(feature)['label'].agg(['sum', 'count']).reset_index()

    new_feat_name = feature + '_ratio'
    df.loc[:, new_feat_name] = (df['sum'] + 0.0001) / (df['count'] + 0.0001)
    df.loc[:, new_feat_name] = np.round(df.loc[:, new_feat_name].values, 4)
    df_stas = df[[feature, new_feat_name]]
    df_val = pd.merge(df_val, df_stas, how='left', on=feature)

    return df_val


def get_common_interest(raw_data, type_name, ratio):
    data_clicked = raw_data[raw_data['label'] == 1]
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


ad_fea = ['aid', 'productId', 'productType', 'creatieId']
user_fea = ['age', 'gender', 'consumptionAbility', 'education', 'LBS']

train_path = '../../datasets/merge/full_train.csv'
test_path = '../../datasets/merge/full_test2.csv'
train_data, test_data = get_raw_data(train_path, test_path)
new_features_train = None
new_features_test = None
# 获取统计特征和交叉特征
static_fea, static_fea_test = get_statistic_feature(train_data, test_data)
cross_fea, cross_fea_test = get_cross_fea(train_data, test_data,  ad_fea, user_fea)

new_features_train = pd.concat([new_features_train, static_fea], axis=1)
new_features_train = pd.concat([new_features_train, cross_fea], axis=1)

new_features_test = pd.concat([new_features_test, static_fea_test], axis=1)
new_features_test = pd.concat([new_features_test, cross_fea_test], axis=1)

new_features_train.to_csv('../../datasets/new_feature_ljh/ljh_new_features_train.csv', index=False)
new_features_test.to_csv('../../datasets/new_feature_ljh/ljh_new_features_test.csv', index=False)

# new_fea = pd.read_csv('./datasets/new_feature/new_fea_cross.csv')

print(new_features_train.head())
