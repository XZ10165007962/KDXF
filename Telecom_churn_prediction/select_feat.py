# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         select_feat
# Description:
# Author:       xinzhuang
# Date:         2022/6/19
# Function:
# Version：
# Notice:
# -------------------------------------------------------------------------------

from sklearn.feature_selection import SequentialFeatureSelector,RFE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
# import lightgbm as lgb

data = pd.read_csv('data/train.csv')

# 训练数据/测试数据准备
features = [f for f in data.columns if f not in ['是否流失', '客户ID']]
x_train = data[features]
y_train = data['是否流失']
selector = SequentialFeatureSelector(
    LinearRegression(), n_features_to_select=55,direction="forward"
).fit_transform(x_train, y_train)
print(selector.get_feature_names_out())