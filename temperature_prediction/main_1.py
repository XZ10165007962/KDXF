#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: main_1.py
@time: 2022/8/14 0:14
@version:
@desc: 指数平滑 0.26210 和线性回归平均0.24244
"""
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.api import SimpleExpSmoothing, Holt
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
	label_col = ["label"]
	for id in tqdm(data["session_id"].unique()):
		data_i = data[data["session_id"] == id]
		index = data_i.index
		data_i["pre"] = 0
		data_i["pre1"] = 0
		train_data = data_i[data_i["rank"] > 12]
		train_data, feat_col = trans_data(train_data)
		data_i = data_i.merge(train_data.loc[:, ["session_id", "mean", "std"]].drop_duplicates(), how="left", on="session_id")
		data_i.index = index
		train_data_x = train_data.loc[:, feat_col]

		model = Holt(train_data_x, initialization_method="estimated").fit(smoothing_level=0.8, smoothing_trend=0.2)
		pred_data = model.forecast(12)
		# 将预测结果拼接回原数据
		sum_data_index = data_i[data_i["rank"] <= 12].index
		data_i.loc[sum_data_index, "pre"] = list(pred_data)
		data_i["pre1"] = list(map(
			lambda x, y, z: x*z+y, data_i["pre"], data_i["mean"], data_i["std"]
		))
		data_i.loc[sum_data_index, "pm"] = data_i.loc[sum_data_index, "pre1"].values
		data.loc[sum_data_index, "pm"] = data_i.loc[sum_data_index, "pre1"].values
	print(data.tail())
	data.to_csv("output/Holt_data.csv", index=False)
	sum_data = data[data["rank"] < 13]
	sum_data = sum_data.loc[:, ["session_id", "rank", "pm"]]
	sum_data.to_csv("output/Holt.csv", index=False)