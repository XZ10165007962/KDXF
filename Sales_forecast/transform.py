#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: transform.py
@time: 2022/7/6 15:27
@version:
@desc: 
"""
import pandas as pd

import config

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)

"""
在训练集中未发生售卖事件的商品
1117 1131 1140 1141 1142 1143 1145 1146 1147 1148 1149 1150 1151 1152
1153 1154 1155 1156 1157 1158 1159 1160 1161 1162
"""


# 用来做滑动和滞后特征的函数
def makelag(data_, values, columns, window, shift=1, if_type=False):
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
	lags = [i + shift for i in range(window)]
	rollings = [i for i in range(2, window)]
	for lag in lags:
		data_[f'{columns}_lag_{lag}'] = values.shift(lag)
	for rolling in rollings:
		data_[f's_{columns}_roll_{rolling}_min'] = values.shift(shift).rolling(window=rolling).min()
		data_[f's_{columns}_roll_{rolling}_max'] = values.shift(shift).rolling(window=rolling).max()
		data_[f's_{columns}_roll_{rolling}_median'] = values.shift(shift).rolling(window=rolling).median()
		data_[f's_{columns}_roll_{rolling}_std'] = values.shift(shift).rolling(window=rolling).std()
		data_[f's_{columns}_roll_{rolling}_mean'] = values.shift(shift).rolling(window=rolling).mean()
	if if_type:
		for lag in lags:
			data_[f'type_lag_{lag}'] = data_.groupby(['type', 'date_block_num'])[f'lag_{lag}'].transform('mean')

	return data_


def trans_data(data_, window=3):
	data_ = data_.groupby(["product_id"]).apply(lambda x: makelag(x, x['label_month'], "label_month", window, True))
	data_ = data_.groupby(["product_id"]).apply(lambda x: makelag(x, x['end_stock'], 'end_stock', window))
	data_["type"] = pd.factorize(data_["type"])[0]

	# 类别特征的encoding
	for func in ['mean', 'std']:
		data_[f'product_label_{func}'] = data_.groupby(['product_id'])['label_month'].transform(func)
		data_[f'type_label_{func}'] = data_.groupby(['type'])['label_month'].transform(func)

	# 后续添加销量变动差异
	return data_


if __name__ == '__main__':
	data = pd.read_csv(config.save_data_path)
	fliter_id = [
		1117, 1131, 1140, 1141, 1142, 1143, 1145, 1146, 1147, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156,
		1157, 1158, 1159, 1160, 1161, 1162
	]
	data = trans_data(data)
	data.to_csv(config.save_trans_data_path, index=False)