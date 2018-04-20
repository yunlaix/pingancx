# -*- coding: utf-8 -*-

"""
主函数
"""

import numpy as np
import pandas as pd
from sklearn import cross_validation
import predictor


# 将预测结果写入文件
def write_result(result):
    pass


def k_fold_val(file_pos, file_neg, k=5):
    """k-cross validation
    :param file_pos: positive samples
    :param file_neg: negative samples
    :param k: k-fold
    :return: None
    """
    pos = pd.read_csv(file_pos)
    neg = pd.read_csv(file_neg)
    print('pos:', pos.shape, '; neg:', neg.shape)

    X_raw = np.concatenate((np.asarray(pos), np.asarray(neg)), axis=0)
    Y = np.asarray([1]*pos.shape[0] + [0]*neg.shape[0])
    X_raw[:], Y[:] = zip(*np.random.permutation(list(zip(X_raw, Y))))

    f1s, accs, recalls = [], [], []
    fold_indices = cross_validation.KFold(n=len(Y), n_folds=k, shuffle=False)
    for train_i, val_i in fold_indices:
        X_train, X_val = X_raw[train_i], X_raw[val_i]
        Y_train, Y_val = Y[train_i], Y[val_i]
        print('\ntrain:', len(Y_train), 'val:', len(Y_val))
        # training
        f1, acc, recall = predictor.lr((X_train, Y_train), (X_val, Y_val))
        print('f1:', f1, 'acc:', acc, 'recall:', recall)
        f1s.append(f1)
        accs.append(acc)
        recalls.append(recall)
    print('f1-mean:', np.mean(f1s), 'acc-mean:', np.mean(accs), 'recall-mean:', np.mean(recalls))

if __name__ == '__main__':
    # 指定好文件路径就好
    k_fold_val(k=5, file_pos='D:\Documents\pacx\ipython_notebook\\10_slices\\pos.csv',
               file_neg='D:\Documents\pacx\ipython_notebook\\10_slices\\neg-slice0.csv')
