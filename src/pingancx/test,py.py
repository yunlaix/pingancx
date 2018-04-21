import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler

# train_raw_num = MinMaxScaler().fit_transform(np.asarray([[1, 2, 3], [1, 2, np.nan], [4, 4, 3]]))
# print(train_raw_num)
df = pd.DataFrame([[1, 2, 9, 0], [3, 4, np.nan, 1],
                   [np.nan, np.nan, np.nan, 5], [np.nan, 3, np.nan, 4]],columns=list('ABCD'))
print(np.asarray(df))
print(np.asarray(df.fillna(method='ffill')))