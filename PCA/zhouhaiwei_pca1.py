# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :principal component analysis 
# @File     :zhouhaiwei_pca1
# @Date     :2021/10/2 22:21
# @Author   :sai
# @Software :PyCharm 
-------------------------------------------------
调包实现PCA
"""

import numpy as np
import pandas as pd
import os
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def compute_pca(n_components=2, last=6, step=10):
    '''
    计算数据集后last个特征前2个主成分的累计贡献率占比,时间窗口和步长均为10
    :param n_components:取前主成分的个数
    :param last:数据集取最后几个特征
    :param step:时间窗口长度
    :return:返回主成分信息占比，一条线为pc1信息占比,一条为pc2+pc1信息占比
    '''
    feature_last = X[:, feature - last:]

    ratio_all = np.zeros((number, n_components))

    for i in range(number):
        feature_window = feature_last[step * i:step * (i + 1), :]
        pca = PCA(n_components=n_components)  # 可以调整主成分个数，n_components = 1
        pca.fit(feature_window)
        ratio = pca.explained_variance_ratio_  # 输出解释方差比
        ratio_all[i] = ratio
    return ratio_all


if __name__ == '__main__':
    last_number = 6
    window_step = 10
    n_component = 2
    X, y = load_boston(return_X_y=True)  # 读取数据
    data, feature = X.shape
    number = data // window_step  # 窗口的数量
    ratio_all = compute_pca(n_components=n_component, last=last_number, step=window_step)

    time_window = np.arange(0, data // window_step)
    pca_1 = ratio_all[:, 0]
    pca_total = np.sum(ratio_all, axis=1)  # axis=1对行求和

    lw = 2
    plt.plot(time_window, pca_1, "red", lw=lw, label='PCA_1')  # lw折线图的线条宽度,
    plt.legend(loc="lower right")
    plt.plot(time_window, pca_total, "darkorange", lw=lw, label='PCA_1+PCA_2')
    plt.legend(loc="lower right")
    plt.xlabel("time windows")
    plt.ylabel("ratio of principal component information")
    plt.title("PCA calculation of Boston housing price data set")
    plt.show()
