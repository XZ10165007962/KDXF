#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: transform.py
@time: 2022/8/11 11:17
@version:
@desc: 
"""
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 200)

# 用来做滑动和滞后特征的函数
def makelag(data_, values, columns, window, shift=0):
	"""
	滑窗特征跟滞后特征
	:param columns: 字段名
	:param if_type: 是否滑动type类型
	:param data_: 输入数据
	:param values: 目标字段字段
	:param window: 滑窗大小
	:param shift: 步幅
	:return:
	"""
	lags = [i + shift for i in range(1, window+1)]
	rollings = [i for i in range(2, window+4)]
	for lag in lags:
		data_[f'{columns}_lag_{lag}'] = values.shift(lag)
		# if columns == "scan_qty":
		# 	data_[f's_{columns}_lag_{lag}_wem_mean'] = values.shift(lag).ewm(alpha=0.9).mean()
	for rolling in rollings:
		data_[f's_{columns}_roll_{rolling}_min'] = values.shift(rolling).rolling(window=rolling).min()
		data_[f's_{columns}_roll_{rolling}_max'] = values.shift(rolling).rolling(window=rolling).max()
		data_[f's_{columns}_roll_{rolling}_median'] = values.shift(rolling).rolling(window=rolling).median()
		data_[f's_{columns}_roll_{rolling}_std'] = values.shift(rolling).rolling(window=rolling).std()
		data_[f's_{columns}_roll_{rolling}_mean'] = values.shift(rolling).rolling(window=rolling).mean()
		data_[f's_{columns}_roll_{rolling}_ewm_mean'] = values.shift(rolling).ewm(alpha=0.9).mean()
	# 	for rolling in rollings:
	# 		data_[f's_{columns}_roll_{lag}_{rolling}_min'] = values.shift(lag).rolling(window=rolling).min()
	# 		data_[f's_{columns}_roll_{lag}_{rolling}_max'] = values.shift(lag).rolling(window=rolling).max()
	# 		data_[f's_{columns}_roll_{lag}_{rolling}_median'] = values.shift(lag).rolling(window=rolling).median()
	# 		data_[f's_{columns}_roll_{lag}_{rolling}_std'] = values.shift(lag).rolling(window=rolling).std()
	# 		data_[f's_{columns}_roll_{lag}_{rolling}_mean'] = values.shift(lag).rolling(window=rolling).mean()

	return data_


def trans_data(data_, window=3, shift=0):
	data_ = data_.groupby(["product_id"]).apply(lambda x: makelag(x, x["scan_qty"], "scan_qty", window, shift=shift))
	# data_ = data_.groupby(["product_id"]).apply(lambda x: makelag(x, x['end_stock'], 'end_stock', window))
	data_["type"] = pd.factorize(data_["type"])[0]

	# 类别特征的encoding
	for func in ['mean', 'std']:
		data_[f'product_label_{func}'] = data_.groupby(['product_id'])["scan_qty"].transform(func)
		data_[f'type_label_{func}'] = data_.groupby(['type'])["scan_qty"].transform(func)
		# data_[f'product_label_year_{func}'] = data_.groupby(['product_id', "year"])["scan_qty"].transform(func)
		# data_[f'product_label_month_{func}'] = data_.groupby(['product_id', "month"])["scan_qty"].transform(func)
		# data_[f'type_year_{func}'] = data_.groupby(['type', "year"])["scan_qty"].transform(func)
		# data_[f'type_month_{func}'] = data_.groupby(['type', "month"])["scan_qty"].transform(func)
	# data_["order_diff"] = data_.groupby(['product_id'])["order"].diff(1)
	# 后续添加销量变动差异
	return data_


if __name__ == '__main__':
	data = pd.read_csv("E:/KDXF/Sales_forecast/output/复赛/data.csv")
	fliter_id = [
		1117, 1131, 1140, 1141, 1142, 1143, 1145, 1146, 1147, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156,
		1157, 1158, 1159, 1160, 1161, 1162
	]
	data = trans_data(data, window=3, shift=0)
	data.fillna(0, inplace=True)
	# print(data.head())
	data.to_csv("E:/KDXF/Sales_forecast/output/复赛/trans_data_0.csv", index=False)

	# path = ["output/trans_data_1.csv", "output/trans_data_2.csv", "output/trans_data_3.csv"]
	# for i in path:
	# 	data = pd.read_csv(i)
	# 	print(data.corr()["scan_qty"])