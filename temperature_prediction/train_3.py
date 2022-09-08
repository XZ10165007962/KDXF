#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: train+4.py
@time: 2022/8/17 14:03
@version:
@desc:
"""
import pandas as pd
import numpy as np
import warnings

from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from pmdarima.arima import auto_arima
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing

warnings.filterwarnings('ignore')
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 200)


def read_data():
	data = pd.read_csv("data/train.csv")
	data = data.sort_values(["session_id", "rank"], ascending=False)
	data = data.loc[:, ["session_id", "rank", "pm"]]
	data["pm_true"] = data["pm"]
	return data


def trans_data(data_):

	data_["diff"] = data_.groupby(["session_id"])["pm"].diff(1)
	data_ = data_.dropna()
	feat_col = ["diff"]

	return data_, feat_col


if __name__ == '__main__':
	data = read_data()
	data["pre"] = 0
	data["pre1"] = 0
	label_col = ["label"]
	for id in data["session_id"].unique():
		data_i = data[data["session_id"] == id]
		for i in range(1, 13):
			train_data = data_i[data_i["rank"] >= (26-i)]
			train_data_x = train_data.loc[:, ["pm"]]
			model = ExponentialSmoothing(train_data_x.values,
										 seasonal_periods=12,
										 trend="mul",
										 seasonal="mul",
										 ).fit()
			pred_data = model.forecast(1)
			# 将预测结果拼接回原数据
			sum_data_index = data_i[data_i["rank"] == (26 - i - 1)].index
			data_i.loc[sum_data_index, "pm"] = pred_data.values
			data.loc[sum_data_index, "pm"] = pred_data.values
		err_data = data_i[data_i["rank"] < 25]
		err_pred = err_data["pm"]
		err_trud = err_data["pm_true"]
		print(f"{id}损失为:", mean_absolute_error(err_trud, err_pred))
	data.to_csv("output/holtwinter_train_pre.csv", index=False)
	err_data = data[data["rank"] < 25]
	err_pred = err_data["pm"]
	err_trud = err_data["pm_true"]
	print(f"总体损失为:", mean_absolute_error(err_trud, err_pred))
