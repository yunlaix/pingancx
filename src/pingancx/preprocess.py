# -*- coding: utf-8 -*-

"""
分割数据(类别不平衡), 异常值处理(缺失值, 异常值), 特征选择(人工 or WHAT ?).
补充:
"""

import  pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
pd.set_option('display.float_format', lambda x: '%.5f' % x)
import missingno as msno
train_data=pd.read_csv('./game/train.csv')
#  忽略弹出的warnings
import warnings
warnings.filterwarnings('ignore')
#输出正例的个数
#print(train_data[train_data.acc_now_delinq>0].count())
#输出describe
#train_describe=train_data.describe()
#train_describe=pd.DataFrame(train_describe)
#train_describe.to_csv('./train_describe.csv')
#print(train_describe.head())
#print(train_describe)

#第一步：剔除无关列
train_data=train_data.drop(columns=['member_id','collections_12_mths_ex_med','earliest_cr_line',
                                    'zip_code','mths_since_last_record','addr_state','earliest_cr_line',
                                    'initial_list_status','collections_12_mths_ex_med','policy_code'])

#print(train_data.shape)

#第二步
# 统计每列属性缺失值的数量，剔除缺失率>0.4的列，转成csv
#axis=0指定对每列求缺失值，axis=1指定对行求缺失值
train_isnull=train_data.isnull().sum(axis=0).sort_values(ascending=False)/float(len(train_data)) #查看缺失值比例
#print(train_isnull[train_isnull > 0.4]) # 查看缺失比例大于40%的属性。
#转置，使用T方法必须基于dataframe,
#print(train_isnull.T)
#train_isnull.T.to_csv('./train_isnullsort.csv')
#aT=pd.DataFrame(train_isnull)
#trainT_isnull=aT.T
#print(trainT_isnull)
# 设定阀值，
thresh_count=len(train_data)*0.4
#若某一列数据缺失的数量超过阀值就会被删除
train_data=train_data.dropna(thresh=thresh_count,axis=1)
#再次检查缺失值情况，发现缺失值比较多的数据列已被我们删除
train_isnull=train_data.isnull().sum(axis=0).sort_values(ascending=False)/float(len(train_data)) #查看缺失值比例
print(train_isnull)
# 将初步预处理后的数据转化为csv
train_data.to_csv('./train_data_01.csv',index=False)

#第三步：
#我们通过Pandas的nunique方法来筛选属性分类为一的变量，
# 剔除分类数量只有1的变量，(分类数量大于15的变量？？不合理)
# Pandas方法nunique()返回的是变量的分类数量（除去非空值）。
train_data=pd.read_csv('./train_data_01.csv')
train_data.dtypes.value_counts() # 分类统计数据类型
train_data=train_data.loc[:,train_data.apply(pd.Series.nunique)!=1]
#train_data=train_data.loc[:,train_data.apply(pd.Series.nunique)<15]
print(train_data.shape)
objectColumns = train_data.select_dtypes(include=["object"]).columns
obj_null=train_data[objectColumns].isnull().sum().sort_values(ascending=False)
print(obj_null)

#输出文本变量的distinct值
objectColumns = train_data.select_dtypes(include=["object"]).columns
non_num_unique=pd.DataFrame(index=[['count','unique_value'],])
print(non_num_unique)
for col in objectColumns.values:
    objectColumns_values=train_data.drop_duplicates(col)
    count=objectColumns_values[col].count()
    unique_value=objectColumns_values[col].values
    non_num_unique[col]=[count,unique_value]

non_num_unique=non_num_unique.T
print(non_num_unique)
#non_num_unique.to_csv('./non_num_unique.csv')
#文本变量缺失值处理
print(msno.matrix(train_data[objectColumns]))
print(msno.heatmap(train_data[objectColumns]))
#填充文本变量缺失值,#以分类“Unknown”填充缺失值
train_data[objectColumns]=train_data[objectColumns].fillna('UnKnown')
print(msno.bar(train_data[objectColumns]))

#第四步
#数值型
numColumns=train_data.select_dtypes(include=[np.number]).columns
print(msno.matrix(train_data[numColumns]))
#画箱线图(画不出来效果）
#'AxesSubplot' object is not subscriptable,指定p返回类型
p=train_data[numColumns].boxplot(return_type='dict')
for i in range(0,len(numColumns)):
    x=p['fliers'][i].get_xdata()
    y=p['fliers'][i].get_ydata()
    y.sort()
    for i in range(len(x)):
        if i > 0:
            plt.annotate(y[i], xy=(x[i], y[i]), xytext=(x[i] + 0.05 - 0.8 / (y[i] - y[i - 1]), y[i]))
        else:
            plt.annotate(y[i], xy=(x[i], y[i]), xytext=(x[i] + 0.08, y[i]))
plt.show()

plt.boxplot(x=train_data[numColumns].values,labels=numColumns,showmeans=True,
            flierprops = {'marker':'o','markerfacecolor':'red','color':'black'},
            meanprops = {'marker':'D','markerfacecolor':'indianred'},
            medianprops = {'linestyle':'--','color':'orange'})
plt.show()
for col in numColumns.values:
    plt.boxplot(x=train_data[col].values)
    plt.plot(title=col)
    plt.show()

if __name__ == '__main__':
    pass
