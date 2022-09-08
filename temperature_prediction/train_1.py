#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: main_1.py
@time: 2022/8/14 0:14
@version:
@desc: 
"""
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.api import SimpleExpSmoothing, Holt
from sklearn.metrics import mean_absolute_error
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
	data_["mean"] = data_.groupby(["session_id"])["pm"].transform("mean")
	data_["std"] = data_.groupby(["session_id"])["pm"].transform("std")
	data_["scaler_pm"] = list(map(
		lambda x, y, z: (x-y)/z, data_["pm"], data_["mean"], data_["std"]
	))
	data_["label"] = data_.groupby(["session_id"])["scaler_pm"].shift(-1)
	feat_col = ["scaler_pm"]

	return data_, feat_col


if __name__ == '__main__':
	data = read_data()
	data["pre"] = 0
	data["pre1"] = 0
	label_col = ["label"]
	for id in data["session_id"].unique():
		data_i = data[data["session_id"] == id]
		for i in range(1, 2):
			data_i_, feat_col = trans_data(data_i)
			train_data = data_i_[data_i_["rank"] >= 25]
			train_data = train_data.dropna()
			train_data_x = train_data.loc[:, feat_col]
			train_data_y = train_data.loc[:, label_col]

			model = Holt(train_data_x, initialization_method="heuristic").fit(
				smoothing_level=0.8, smoothing_trend=0.2
			)
			pred_data = model.forecast(12)
			# 将预测结果拼接回原数据
			sum_data_index = data_i_[data_i_["rank"] < 25].index
			data_i_.loc[sum_data_index, "pre"] = list(pred_data)
			data_i_["pre1"] = list(map(
				lambda x, y, z: x*z+y, data_i_["pre"], data_i_["mean"], data_i_["std"]
			))
			data_i.loc[sum_data_index, "pm"] = data_i_.loc[sum_data_index, "pre1"]
			data.loc[sum_data_index, "pm"] = data_i_.loc[sum_data_index, "pre1"]
		err_data = data_i[data_i["rank"] < 25]
		err_pred = err_data["pm"]
		err_trud = err_data["pm_true"]
		print(f"{id}损失为:", mean_absolute_error(err_trud, err_pred))
	data.to_csv("output/holt_train_pre.csv", index=False)
	err_data = data[data["rank"] < 25]
	err_pred = err_data["pm"]
	err_trud = err_data["pm_true"]
	print(f"总体损失为:", mean_absolute_error(err_trud, err_pred))