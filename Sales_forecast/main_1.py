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

import pandas as pd
import numpy as np

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)

# 读取原始数据
data = pd.read_csv("output/data_1.csv")
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
		"date_block_num", "pre", "product_id", "mean_qty",	"scan_qty"

	]
	train_data = use_data[use_data["date_block_num"] < 36]
	val_data = use_data[use_data["date_block_num"] == 36]
	test_data = use_data[use_data["date_block_num"] == date]

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

	print("*****************lgb模型训练**********************")
	lgb_train, lgb_test = model_1.lgb_model(train_x, train_y, test_x, val_x, val_y)
	data.loc[test_x.index, ["pre"]] = lgb_test
	# print(("score:", mean_absolute_error(test_y, lgb_test)))
	# 将预测结果拼接到原始数据
	# data.loc[test_x.index, ["label_month"]] = lgb_test

data["pre1"] = data["pre"] * data["mean_qty"]
data.to_csv("output/tijiao_1.csv", index=False)

df_test = data[data["date_block_num"] >= 37]
df_test = df_test.sort_values(by=['month', 'product_id'])
print(df_test.head())
# sub = pd.read_csv('data/goods/提交示例.csv')
sub = pd.read_csv('data/提交示例.csv')
print(sub.head())
sub["label"] = df_test["pre1"].values
sub["label"] = sub["label"].map(lambda x: x if x >= 0 else 0)

# sub.to_csv('data/goods/baseline0613.csv', index=False)
sub.to_csv('output/baseline0717.csv', index=False)