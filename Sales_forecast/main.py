#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: main.py
@time: 2022/7/5 15:36
@version:
@desc: 
"""
import model
import config

import pandas as pd
import numpy as np

data = pd.read_csv(config.save_trans_data_path)
data = data[data["qty_month_count"] >= 6]
columns = data.columns
del_col = ["label_sum", "label_mean", "label_std", "label_cv", "label_month", "qty_month_count"]
label = ["label_month"]
use_col = [x for x in columns if x not in del_col]

train_data = data[data["date_block_num"] < 37]
test_data = data[data["date_block_num"] >= 37]

x_train = train_data.loc[:, use_col]
y_train = train_data.iloc[:, label]

x_test = test_data.loc[:, use_col]

test_pre = np.zeros(test_data.shape[0])
train_pre = np.zeros(train_data.shape[0])

print("*****************lgb模型训练**********************")
for i, date in enumerate([37, 38, 39]):
	print("*****************{}**********************".format(date))
	lgb_train, lgb_test = model.lgb_model(x_train, y_train, test_data, date)
	index = train_data[train_data["year_month"] == date].index
	train_pre[index] = lgb_train
	test_data["label"+str(i)] = lgb_test
print("训练结束，保存数据")
train_data["pre"] = train_pre
train_data.to_csv(config.save_train_pre_path, index=False)
test_data.to_csv(config.save_test_pre_path, index=False)