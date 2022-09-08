#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: train_2.py
@time: 2022/8/15 10:55
@version:
@desc: 动量计算
"""
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
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
	alph = 0.2
	data_["pm_diff"] = data_.groupby(["session_id"])["pm"].diff(1)
	data_["pm_diff_cum"] = data_.groupby(["session_id"])["pm_diff"].cumsum()
	data_["count"] = data_.shape[0]
	data_["momentum"] = list(map(
		lambda x, y, z: alph*x+(1-alph)*(y/(z)), data_["pm_diff"], data_["pm_diff_cum"], data_["count"]
	))
	data_["pre"] = list(map(
		lambda x, y: x + y, data_["pm"], data_["momentum"]
	))
	return data_


if __name__ == '__main__':
	data = read_data()
	data["pre"] = 0
	data["pre1"] = 0
	data = data[data["rank"] <= 35]
	for id in data["session_id"].unique():
		data_i = data[data["session_id"] == id]
		for i in range(1, 13):
			data_i_ = data_i[data_i["rank"] >= (26-i)]
			data_i_ = trans_data(data_i_)

			# 将预测结果拼接回原数据
			sum_data_index = data_i[data_i["rank"] == (26-i-1)].index
			pre_data_index = data_i_[data_i_["rank"] == (26-i)].index
			data_i.loc[sum_data_index, "pm"] = data_i_.loc[pre_data_index, "pre"].values
			data.loc[sum_data_index, "pm"] = data_i_.loc[pre_data_index, "pre"].values
		err_data = data_i[data_i["rank"] < 25]
		err_pred = err_data["pm"]
		err_trud = err_data["pm_true"]
		print(f"{id}损失为:", mean_absolute_error(err_trud, err_pred))
	data.to_csv("output/test_train_pre.csv", index=False)
	err_data = data[data["rank"] < 25]
	err_pred = err_data["pm"]
	err_trud = err_data["pm_true"]
	print(f"总体损失为:", mean_absolute_error(err_trud, err_pred))