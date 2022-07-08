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
import transform

import pandas as pd
import numpy as np

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)


def func():
	data = pd.read_csv(config.save_trans_data_path)
	data = data[data["qty_month_count"] >= 6]
	columns = data.columns
	del_col = [
		"label_sum", "label_mean", "label_std", "label_cv", "label_month", "qty_month_count", "date", "date_block_num"
	]
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


if __name__ == '__main__':
	# 读取原始数据
	data = pd.read_csv(config.save_data_path)
	# data = data[data["qty_month_count"] >= 3]
	data["pre"] = [0]*data.shape[0]

	for date in [37, 38, 39]:
		use_data = data[data["date_block_num"] <= date]
		# 构造数据特征
		use_data = transform.trans_data(use_data, 3)
		columns = use_data.columns
		del_col = [
			"label_sum", "label_mean", "label_std", "label_cv", "label_month", "qty_month_count", "date",
			"date_block_num", "pre", "product_id"
		]
		train_data = use_data[use_data["date_block_num"] < date-1]
		val_data = use_data[use_data["date_block_num"] == date-1]
		test_data = use_data[use_data["date_block_num"] == date]

		# 目标字段
		label = ["label_month"]
		# 特征字段
		use_col = [x for x in columns if x not in del_col]

		train_x = train_data.loc[:, use_col]
		val_x = val_data.loc[:, use_col]
		test_x = test_data.loc[:, use_col]

		train_y = train_data.loc[:, label]
		val_y = val_data.loc[:, label]

		print("*****************lgb模型训练**********************")
		lgb_train, lgb_test = model.lgb_model(train_x, train_y, test_x, val_x, val_y)
		data.loc[val_x.index, ["pre"]] = lgb_train
		# 将预测结果拼接到原始数据
		data.loc[test_x.index, ["label_month"]] = lgb_test

	data.to_csv(config.save_test_pre_path, index=False)