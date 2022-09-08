#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: win_rolling.py
@time: 2022/8/30 21:41
@version:
@desc: 
"""
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings('ignore')
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 200)


def train():
	def read_data():
		data = pd.read_csv("data/train.csv")
		data = data.sort_values(["session_id", "rank"], ascending=False)
		data = data.loc[:,["session_id", "rank", "pm"]]
		data["pm_true"] = data["pm"]
		return data


	data = read_data()
	ids = data["session_id"].unique()
	for id in ids:
		for i in range(12):
			data_ = data[data["session_id"] == id]
			data_ = data_[data_["rank"] >= (25-i)]
			data_["pre"] = data_["pm"].rolling(6).mean()
			simple = data[data["session_id"] == id]
			sub_index = simple[simple["rank"] == (24-i)].index
			data.loc[sub_index, ["pm"]] = data_.loc[data_[data_["rank"] == (25-i)].index, ["pre"]].values
		err_data = data[data["session_id"] == id]
		err_data = err_data[err_data["rank"] < 25]
		print(f"{id}:", mean_absolute_error(err_data["pm"], err_data["pm_true"]))
	err_data = data[data["rank"] < 25]
	print(f"总体损失为:", mean_absolute_error(err_data["pm"], err_data["pm_true"]))
	data.to_csv("output/win_rolling_train.csv", index=False)


def eval():
	def read_data():
		data_1 = pd.read_csv("data/train.csv")
		data_2 = pd.read_csv("data/test.csv")
		data_1 = data_1.loc[:, ["session_id", "rank", "pm"]]
		data = pd.concat([data_1, data_2], axis=0)
		data = data.sort_values(["session_id", "rank"], ascending=False).reset_index(drop=True)
		return data

	data = read_data()
	ids = data["session_id"].unique()
	for id in ids:
		for i in range(12):
			data_ = data[data["session_id"] == id]
			data_ = data_[data_["rank"] >= (13 - i)]
			data_["pre"] = data_["pm"].rolling(3).mean()
			simple = data[data["session_id"] == id]
			sub_index = simple[simple["rank"] == (12 - i)].index
			data.loc[sub_index, ["pm"]] = data_.loc[data_[data_["rank"] == (13 - i)].index, ["pre"]].values
	sub_data = data[data["rank"] < 13]
	sub_data.to_csv("output/win_rolling.csv", index=False)


def cv():
	lgb_mae_seed2022 = pd.read_csv("output/lgb_mae_seed2022_train_pre.csv")
	win = pd.read_csv("output/win_rolling_train.csv")
	linear = pd.read_csv("output/liearn_train_pre.csv")

	lgb_mae_seed2022 = lgb_mae_seed2022.loc[:, ["session_id", "rank", "pm", "pm_true"]]

	win = win[win["rank"] < 25]
	lgb_mae_seed2022 = lgb_mae_seed2022[lgb_mae_seed2022["rank"] < 25]
	linear = linear[linear["rank"] < 25]

	ids = win["session_id"].unique()
	for id in ids:
		data1 = win[win["session_id"] == id]
		data2 = lgb_mae_seed2022[lgb_mae_seed2022["session_id"] == id]
		data3 = linear[linear["session_id"] == id]
		print(id)
		print("lgb error", mean_absolute_error(data2["pm"], data2["pm_true"]))
		print("win error", mean_absolute_error(data1["pm"], data1["pm_true"]))
		print("linear error", mean_absolute_error(data3["pm"], data3["pm_true"]))


if __name__ == '__main__':
	cv()