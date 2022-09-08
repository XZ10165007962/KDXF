#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: linear_train.py
@time: 2022/9/8 9:50
@version:
@desc: 
"""
import pandas as pd
import warnings
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_absolute_error
warnings.filterwarnings('ignore')
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 200)


def read_data():
	data = pd.read_csv("../xfdata/oppo_breeno_round1_data/train.csv")
	data = data.sort_values(["session_id", "rank"], ascending=False)
	data["pm_true"] = data["pm"]
	return data


def trans_data(data_, lag):
	data_["mean"] = data_.groupby(["session_id"])["pm"].transform("mean")
	data_["std"] = data_.groupby(["session_id"])["pm"].transform("std")
	data_["scaler_pm"] = list(map(
		lambda x, y, z: (x - y) / z, data_["pm"], data_["mean"], data_["std"]
	))
	data_["label"] = data_.groupby(["session_id"])["scaler_pm"].shift(-1)
	feat_col = []
	for i in range(lag+1):
		data_[f"lag_{i}"] = data_.groupby(["session_id"])["scaler_pm"].shift(i)
		feat_col.append(f"lag_{i}")
	feat_col.append("rank")
	return data_, feat_col


if __name__ == '__main__':
	data = read_data()
	data["pre"] = 0
	data["pre1"] = 0
	label_col = ["label"]
	list_12 = [
		0, 2, 3, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 24, 25, 28, 29, 32, 33, 34, 35, 36, 37, 39, 40, 43,
		44, 45, 46, 48, 49, 56, 58, 59, 60, 61, 62, 64, 65, 66, 67, 69, 74, 75, 76, 77, 83, 84, 85, 89, 91, 92, 95, 98,
		100, 101, 102, 104, 105, 106, 108, 110, 112, 113, 114, 115, 116, 118, 119, 125, 126, 128, 130, 135, 136, 137,
		138, 139, 144, 148, 151, 153, 154, 155, 161, 163, 165, 166, 167, 170, 174, 175, 177, 180, 186, 187, 188, 190,
		191, 194, 199, 202, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 216, 217, 220, 221, 224, 228, 229,
		230, 233, 234, 237, 239, 242, 246, 247, 249, 250, 252, 253, 254, 256, 260, 261, 268, 269, 271, 272, 273, 274,
		275, 276, 277, 278, 281, 285, 286, 287, 289, 292, 294
	]
	list_6 = [
		1, 4, 5, 6, 7, 9, 17, 19, 26, 27, 30, 31, 38, 41, 42, 47, 50, 51, 52, 53, 54, 55, 57, 63, 68, 70, 71, 72, 73,
		78, 79, 80, 81, 82, 86, 87, 88, 90, 93, 94, 96, 97, 99, 103, 107, 109, 111, 117, 120, 121, 122, 123, 124, 127,
		129, 131, 132, 133, 134, 140, 141, 142, 143, 145, 146, 147, 149, 150, 152, 156, 157, 158, 159, 160, 162, 164,
		168, 169, 171, 172, 173, 176, 178, 179, 181, 182, 183, 184, 185, 189, 192, 193, 195, 196, 197, 198, 200, 201,
		203, 215, 218, 219, 222, 223, 225, 226, 227, 231, 232, 235, 236, 238, 240, 241, 243, 244, 245, 248, 251, 255,
		257, 258, 259, 262, 263, 264, 265, 266, 267, 270, 279, 280, 282, 283, 284, 288, 290, 291, 293
	]
	lag1 = 12
	lag2 = 6
	for id in list_12:
		data_i = data[data["session_id"] == id]
		for i in range(1, 13):
			data_i_ = data_i[data_i["rank"] >= (26-i)]
			data_i_, feat_col = trans_data(data_i_, lag1)
			train_data = data_i_[data_i_["rank"] > (26-i)]
			train_data.dropna(inplace=True)
			train_data_x = train_data.loc[:, feat_col]
			train_data_y = train_data.loc[:, label_col]

			test_data = data_i_[data_i_["rank"] == (26-i)]
			test_data_x = test_data.loc[:, feat_col]
			test_data_y = test_data.loc[:, label_col]
			model = LinearRegression().fit(train_data_x, train_data_y)
			# model = SVR().fit(train_data_x, train_data_y)
			pred_data = model.predict(test_data_x)
			# 将预测结果拼接回原数据
			data_i_.loc[test_data.index, "pre"] = pred_data
			data_i_["pre1"] = list(map(
				lambda x, y, z: x * z + y, data_i_["pre"], data_i_["mean"], data_i_["std"]
			))
			sum_data_index = data_i[data_i["rank"] == (26-i-1)].index
			data_i.loc[sum_data_index, "pm"] = data_i_.loc[test_data.index, "pre1"].values
			data.loc[sum_data_index, "pm"] = data_i_.loc[test_data.index, "pre1"].values

		err_data = data_i[data_i["rank"] < 25]
		err_pred = err_data["pm"]
		err_trud = err_data["pm_true"]
		print(f"{id}损失为:", mean_absolute_error(err_trud, err_pred))
	for id in list_6:
		data_i = data[data["session_id"] == id]
		for i in range(1, 13):
			data_i_ = data_i[data_i["rank"] >= (26-i)]
			data_i_, feat_col = trans_data(data_i_, lag2)
			train_data = data_i_[data_i_["rank"] > (26-i)]
			train_data.dropna(inplace=True)
			train_data_x = train_data.loc[:, feat_col]
			train_data_y = train_data.loc[:, label_col]

			test_data = data_i_[data_i_["rank"] == (26-i)]
			test_data_x = test_data.loc[:, feat_col]
			test_data_y = test_data.loc[:, label_col]
			model = LinearRegression().fit(train_data_x, train_data_y)
			# model = SVR().fit(train_data_x, train_data_y)
			pred_data = model.predict(test_data_x)
			# 将预测结果拼接回原数据
			data_i_.loc[test_data.index, "pre"] = pred_data
			data_i_["pre1"] = list(map(
				lambda x, y, z: x * z + y, data_i_["pre"], data_i_["mean"], data_i_["std"]
			))
			sum_data_index = data_i[data_i["rank"] == (26-i-1)].index
			data_i.loc[sum_data_index, "pm"] = data_i_.loc[test_data.index, "pre1"].values
			data.loc[sum_data_index, "pm"] = data_i_.loc[test_data.index, "pre1"].values

		err_data = data_i[data_i["rank"] < 25]
		err_pred = err_data["pm"]
		err_trud = err_data["pm_true"]
		print(f"{id}损失为:", mean_absolute_error(err_trud, err_pred))
	data.to_csv("../user_data/tmp_data/liearn_train_pre.csv", index=False)
	err_data = data[data["rank"] < 25]
	err_pred = err_data["pm"]
	err_trud = err_data["pm_true"]
	print(f"总体损失为:", mean_absolute_error(err_trud, err_pred))
