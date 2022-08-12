#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: main_1.py
@time: 2022/8/11 15:40
@version:
@desc: 
"""
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm

import data
import transform
import model

warnings.filterwarnings('ignore')
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 200)

# 读取原始数据
data = data.read_data()
# data = data[data["qty_month_count"] >= 3]
data["pre"] = [0]*data.shape[0]

windows = 3

for shift, date in enumerate([40, 41, 42]):
	use_data = data[data["date_block_num"] <= date]
	# 构造数据特征
	if date == 42:
		use_data = transform.trans_data(use_data, window=windows, shift=0)
	else:
		use_data = transform.trans_data(use_data, window=windows, shift=shift)
	columns = use_data.columns
	del_col = [
		"label_sum", "label_mean", "label_std", "label_cv", "label_month", "qty_month_count", "date",
		"date_block_num", "pre", "product_id", "mean_qty",	"scan_qty", "sale_count"
	]
	k_fold = 5
	test_data = use_data[use_data["date_block_num"] == date]
	test_pre = np.zeros((test_data.shape[0], k_fold))
	for i in range(k_fold):
		if date == 42:
			train_data = use_data[use_data["date_block_num"] < (41-i)]
			val_data = use_data[use_data["date_block_num"] == (41-i)]
		else:
			train_data = use_data[use_data["date_block_num"] < (39-i)]
			val_data = use_data[use_data["date_block_num"] == (39-i)]

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
		lgb_train, lgb_test = model.lgb_model(train_x, train_y, test_x, val_x, val_y)
		test_pre[:, i] = lgb_test
	data.loc[test_x.index, ["pre"]] = np.mean(test_pre, 1)
	data.loc[test_x.index, label] = np.mean(test_pre, 1)
data["pre1"] = data["pre"] * data["mean_qty"]
data.to_csv("E:/KDXF/Sales_forecast/output/复赛/tijiao.csv", index=False)

df_test = data[data["date_block_num"] >= 40]
df_test = df_test.sort_values(by=['month', 'product_id'])
sub = pd.read_csv("E:/KDXF/Sales_forecast/output/复赛/data.csv")
sub = sub.query("year==2021 and month>=4")
sub = sub.drop_duplicates(['month', 'product_id'])
sub = sub.loc[:, ["year", 'month', 'product_id']]
sub = sub.merge(df_test.loc[:, ['month', 'product_id', "pre1"]], how="left", on=['month', 'product_id'])
sub = df_test.loc[:, ["year", 'month', 'product_id', "pre1"]].rename(columns={'pre1': 'label'})
sub["month1"] = sub["year"].astype("str")+"-"+sub["month"].astype("str")
del sub["year"]
del sub["month"]
sub = sub.rename(columns={'month1': 'month'})
sub["label"] = sub["label"].map(lambda x: x if x >= 0 else 0)
# sub["label"] = sub["label"] * 1.055
# sub.loc[sub[sub["label"] == 0].index, ["label"]] = 300
sub = sub.loc[:, ["month", "product_id", "label"]]
sub.to_csv('E:/KDXF/Sales_forecast/output/复赛/sub.csv', index=False)
