# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :principal component analysis
# @File     :data_analysis
# @Date     :2021/10/3 17:20
# @Author   :sai
# @Software :PyCharm
-------------------------------------------------
"""

from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

boston = load_boston()  # 读取数据
column_name=np.insert(boston["feature_names"][7:], 6, "MEDV")

X, y = load_boston(return_X_y=True)  # 读取数据
feature_last = X[:, 13 - 6:]
y=y.reshape(y.shape[0],1)
data = np.concatenate((feature_last, y), axis=1)  # 水平排列ndarray，列方式
data = pd.DataFrame(data)
data.columns = column_name

data1=data.iloc[:,0:6]
import seaborn as sns
sns.heatmap(data1.corr(), annot=True)