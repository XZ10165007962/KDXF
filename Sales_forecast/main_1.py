#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: main_1.py
@time: 2022/7/17 13:49
@version:
@desc: 
"""
from sklearn.metrics import mean_absolute_error

import model_1
import config
import transform_1
import data_1
from fusai import utils

import pandas as pd
import numpy as np

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)

# 读取原始数据
data = data_1.read_data()
# data = data[data["qty_month_count"] >= 3]
data["pre"] = [0]*data.shape[0]

windows = 3

for shift, date in enumerate([37, 38, 39]):
	use_data = data[data["date_block_num"] <= date]
	# 构造数据特征
	use_data = transform_1.trans_data(use_data, window=windows, shift=shift)
	columns = use_data.columns
	del_col = [
		"label_sum", "label_mean", "label_std", "label_cv", "label_month", "qty_month_count", "date",
		"date_block_num", "pre", "product_id", "mean_qty",	"scan_qty", "sale_count"
	]
	k_fold = 5
	test_data = use_data[use_data["date_block_num"] == date]
	test_pre = np.zeros((test_data.shape[0], k_fold))
	for i in range(k_fold):
		train_data = use_data[use_data["date_block_num"] < (36-i)]
		val_data = use_data[use_data["date_block_num"] == (36-i)]

		# 目标字段
		label = ["scan_qty"]
		# 特征字段
		use_col = [x for x in columns if x not in del_col]

		train_x = train_data.loc[:, use_col]
		val_x = val_data.loc[:, use_col]
		test_x = test_data.loc[:, use_col]

		train_y = train_data.loc[:, label]
		val_y = val_data.loc[:, label]
		test_y = test_data.loc[:, label]

		print(f"*****************{date}-{i+1}/{k_fold}_lgb模型训练**********************")
		lgb_train, lgb_test = model_1.lgb_model(train_x, train_y, test_x, val_x, val_y)
		test_pre[:, i] = lgb_test
	data.loc[test_x.index, ["pre"]] = np.mean(test_pre, 1)
data["pre1"] = data["pre"] * data["mean_qty"]
# data.to_csv("output/tijiao_1.csv", index=False)

df_test = data[data["date_block_num"] >= 37]
df_test = df_test.sort_values(by=['month', 'product_id'])
# sub = pd.read_csv('data/goods/提交示例.csv')
sub = pd.read_csv('data/初赛/提交示例.csv')
sub["label"] = df_test["pre1"].values

sub["label"] = sub["label"].map(lambda x: x if x >= 0 else 0)
sub["label"] = sub["label"] * 1.055
sub.loc[sub[sub["label"] == 0].index, ["label"]] = 300
pred = sub["label"].values
true = df_test["label_month"].values
print(utils.get_accuracy(true, pred))
# sub.to_csv('output/初赛/baseline0811.csv', index=False)