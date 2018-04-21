# -*- coding: utf-8 -*-

"""
分割数据(类别不平衡), 异常值处理(缺失值, 异常值), 特征选择(人工 or WHAT ?), 数值化.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from collections import defaultdict
import sys


def feature_selection(features, file_path):
    """
    :param features: list, feature sub-set
    :param file_path:
    :return: train data after dropping some features
    """
    train_raw = pd.read_csv(file_path, usecols=features)
    # train_raw = train_raw.dropna(axis=0)  # for test data, don't drop
    train_raw.to_csv('D:\Documents\pacx\ipython_notebook\\test_raw_1.csv', index=False)
    return train_raw


def print_rawdatainfo(train_raw):
    print('dtypes:', train_raw.dtypes)
    print('shape_size:', train_raw.shape, train_raw.size)


def clean_data(train_raw):
    mapping_dict = {
        "emp_length": {
            "10+ years": 10,
            "9 years": 9,
            "8 years": 8,
            "7 years": 7,
            "6 years": 6,
            "5 years": 5,
            "4 years": 4,
            "3 years": 3,
            "2 years": 2,
            "1 year": 1,
            "< 1 year": 0,
            "n/a": 0
        },
        'loan_status': {
            'Does not meet the credit policy. Status:Charged Off': 'Charged Off',
            'Does not meet the credit policy. Status:Fully Paid': 'Fully Paid'
        }
        }
    # train_raw['emp_length'] = (train_raw['emp_length'].astype(str).str[:2]).astype(int)
    train_raw = train_raw.replace(mapping_dict)
    train_raw['term'] = (train_raw['term'].astype(str).str[1:3]).astype(int)
    return train_raw


def fill_nan(train_raw):
    """fill nan with forward valid values
    :param train_raw:
    :return:
    """
    return train_raw.fillna(method='ffill')


def load_train_data(type_num, type_obj):
    train_raw = pd.read_csv('D:\Documents\pacx\ipython_notebook\\train_raw_1.csv')
    del train_raw['member_id']
    train_raw = clean_data(train_raw)
    train_raw_num = train_raw[type_num]
    train_raw_obj = train_raw[type_obj]
    return train_raw_num, train_raw_obj


def normalize_encode(train_raw):
    """归一化, 定性属性编码
    :param train_raw:
    :return: numeric data
    """
    # spilt numeric/object attribute
    mask_num = train_raw.dtypes != object
    mask_obj = train_raw.dtypes == object
    type_num = mask_num.index[mask_num].tolist()
    type_obj = mask_obj.index[mask_obj].tolist()

    fit_num, fit_obj = load_train_data(type_num, type_obj)  # for test data, use train data to fit
    d = defaultdict(LabelEncoder)
    fit_obj.apply(lambda x: d[x.name].fit(x))

    # for numeric attributes, normalize it
    train_raw_num = train_raw[type_num]
    minmax_scaler = MinMaxScaler()
    minmax_scaler.fit(fit_num)
    train_raw_num = minmax_scaler.transform(train_raw_num)
    train_raw_num = pd.DataFrame(train_raw_num, columns=type_num)

    # for non-numeric attributes, encode it
    train_raw_obj = train_raw[type_obj]
    # labelEncoder = LabelEncoder()
    # labelEncoder.fit(fit_obj)
    train_raw_obj = train_raw_obj.apply(lambda x: d[x.name].transform(x))
    print(fit_obj.shape, train_raw_obj.shape, len(d))
    nuniques, offset, indices = train_raw_obj.nunique(), 0, []
    print('nuniques', nuniques)
    for nunique in nuniques:
        indices = np.append(indices, np.arange(1, nunique) + offset)
        offset += nunique
    indices = indices.astype(int)
    # print('indices', indices)

    # data stuck !!! delete one column for erase relation !!!
    train_raw_obj = OneHotEncoder().fit_transform(train_raw_obj).toarray()
    train_raw_obj = train_raw_obj[:, indices]
    train_raw_obj = pd.DataFrame(train_raw_obj)
    train_data = pd.concat([train_raw_num, train_raw_obj], axis=1)
    train_data.to_csv('D:\Documents\pacx\ipython_notebook\\test_data.csv', index=False)
    return train_data


def split_slices(file_path):
    """分割数据使得正负例数目相当
    :param file_path:
    :return: None
    """
    train_data = pd.read_csv(file_path)
    pos = train_data[train_data['acc_now_delinq'] > 0]
    neg = train_data[train_data['acc_now_delinq'] < 1]
    del pos['acc_now_delinq']
    del neg['acc_now_delinq']
    pos_len = pos.shape[0] * 1
    neg_len = neg.shape[0]
    for i in range(10):  # int(neg_len/pos_len)
        neg_slice = pd.DataFrame(neg[i*pos_len:(i+1)*pos_len])
        neg_slice.to_csv('D:\Documents\pacx\ipython_notebook\\10_slices\\neg-slice'+str(i)+'.csv', index=False)
    pd.DataFrame(pos).to_csv('D:\Documents\pacx\ipython_notebook\\10_slices\pos.csv', index=False)


def step_1():
    # train_raw = feature_selection(features=['member_id', 'loan_amnt', 'term', 'int_rate', 'grade',
    #                                         'emp_length', 'home_ownership', 'annual_inc', 'verification_status',
    #                                         'loan_status', 'revol_bal', 'revol_util', 'total_acc', 'out_prncp',
    #                                         'application_type', 'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim'],
    #                               file_path='D:\Documents\pacx\\test.csv')  # no 'acc_now_delinq', for test data
    train_raw = pd.read_csv('D:\Documents\pacx\ipython_notebook\\test_raw_1.csv')
    del train_raw['member_id']
    train_raw = clean_data(train_raw)
    train_raw = fill_nan(train_raw)  # only for test data
    print_rawdatainfo(train_raw)
    train_data = normalize_encode(train_raw)
    print_rawdatainfo(train_data)
    # split_slices('D:\Documents\pacx\ipython_notebook\\train_data.csv')

if __name__ == '__main__':
    # step_1()
    # 执行的步骤在step_1()中
    pass
