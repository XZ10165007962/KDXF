#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: main_3.py
@time: 2022/8/17 10:32
@version:
@desc: holtwinters 0.27926
"""
import pandas as pd
import numpy as np
import warnings

from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from tqdm import tqdm
from pmdarima.arima import auto_arima

warnings.filterwarnings('ignore')
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 200)


def read_data():
	data_1 = pd.read_csv("data/train.csv")
	data_2 = pd.read_csv("data/test.csv")
	data_1 = data_1.loc[:, ["session_id", "rank", "pm"]]
	data = pd.concat([data_1, data_2], axis=0)
	data = data.sort_values(["session_id", "rank"], ascending=False).reset_index(drop=True)
	return data


def trans_data(data_):

	data_["diff"] = data_.groupby(["session_id"])["pm"].diff(1)
	data_ = data_.dropna()
	feat_col = ["diff"]

	return data_, feat_col


if __name__ == '__main__':
	data = read_data()
	label_col = ["label"]
	for id in tqdm(data["session_id"].unique()):
		data_i = data[data["session_id"] == id]
		for i in range(1, 13):
			data_i_ = data_i[data_i["rank"] >= (14-i)]
			train_data_x = data_i_.loc[:, ["pm"]]
			model = ExponentialSmoothing(train_data_x.values,
										 seasonal_periods=12,
										 trend="mul",
										 seasonal="mul",
										 ).fit()
			pred_data = model.forecast(1)
			# 将预测结果拼接回原数据
			sum_data_index = data_i[data_i["rank"] == (14 - i - 1)].index
			data_i.loc[sum_data_index, "pm"] = pred_data
			data.loc[sum_data_index, "pm"] = pred_data

	data.to_csv("output/holtwinters_data.csv", index=False)
	sum_data = data[data["rank"] < 13]
	sum_data = sum_data.loc[:, ["session_id", "rank", "pm"]]
	sum_data.to_csv("output/holtwinters.csv", index=False)
