# -*- coding: utf-8 -*-

"""
分类器
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score


def lr(train_data, val_data):
    """logistic regression
    :param train_data: X_train and Y_train
    :param val_data: X_val and Y_val
    :return: f1-score, accuracy rate and recall-score
    """
    X_train, Y_train = train_data
    X_val, Y_val = val_data
    lr = LogisticRegression(penalty='l2', random_state=137, solver='newton-cg', C=0.3)
    lr.fit(X=X_train, y=Y_train)
    Y_val_pred = lr.predict(X=X_val)
    print('val_1:', sum(Y_val == 1))
    f1 = f1_score(y_true=Y_val, y_pred=Y_val_pred)
    recall = recall_score(y_true=Y_val, y_pred=Y_val_pred)
    acc = sum(Y_val_pred == Y_val)/len(Y_val)
    return f1, acc, recall


def my_Model(train_data, val_data):
    """my Model
    :param train_data: X_train and Y_train
    :param val_data: X_val and Y_val
    :return: f1-score, accuracy rate and recall-score
    """
    X_train, Y_train = train_data
    X_val, Y_val = val_data
    print('val_1:', sum(Y_val == 1))
    f1, acc, recall = 0, 0, 0

    # my Model ......

    return f1, acc, recall


def predict(X_train, Y_train, test_data):
    """predict for testing data
    :param model:
    :param test_data:
    :return:
    """
    lr = LogisticRegression(penalty='l2', random_state=137, solver='newton-cg', C=0.3)
    lr.fit(X=X_train, y=Y_train)
    return lr.predict(X=test_data)


if __name__ == '__main__':
    a = np.asarray([1, 2, 3, 4])
    print(np.mean(a))
    pass
