#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: data.py
@time: 2022/8/15 14:30
@version:
@desc: 
"""
import json

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


def get_result_weight():
	data2 = pd.read_csv("output/momentum.csv")
	data1 = pd.read_csv("output/Holt.csv")
	data3 = pd.read_csv("output/linear.csv")

	data1 = data1.rename(columns={"pm": "Holt_pm"})
	data2 = data2.rename(columns={"pm": "moment_pm"})
	data3 = data3.rename(columns={"pm": "linear_pm"})

	data = data1.merge(data2, on=["session_id", "rank"], how="left")
	data = data.merge(data3, on=["session_id", "rank"], how="left")
	data["holt_mom"] = list(map(
		lambda x, y: abs(x - y), data["Holt_pm"], data["moment_pm"]
	))
	data["holt_linear"] = list(map(
		lambda x, y: abs(x - y), data["Holt_pm"], data["linear_pm"]
	))
	data["mom_linear"] = list(map(
		lambda x, y: abs(x - y), data["moment_pm"], data["linear_pm"]
	))
	data["holt_mom_mean"] = data.groupby(["session_id"])["holt_mom"].transform("mean")
	data["holt_linear_mean"] = data.groupby(["session_id"])["holt_linear"].transform("mean")
	data["mom_linear_mean"] = data.groupby(["session_id"])["mom_linear"].transform("mean")

	data_x = data[data["mom_linear_mean"] > 0.2]
	data_y = data[data["mom_linear_mean"] < 0.2]
	data_y["pm"] = list(map(
		lambda x,y: (x+y)/2, data_y["Holt_pm"],data_y["linear_pm"]
	))
	data_x["pm"] = list(map(
		lambda x,y,z: (x+y+z)/3, data_x["Holt_pm"],data_x["moment_pm"],data_x["linear_pm"]
	))
	sub_data = pd.concat([data_x, data_y])
	sub_data = sub_data.loc[:, ["session_id", "rank", "pm"]]
	sub_data.to_csv("output/sub.csv", index=False)


def get_result_mean():
	data2 = pd.read_csv("output/momentum.csv")
	data1 = pd.read_csv("output/Holt.csv")
	data3 = pd.read_csv("output/linear.csv")
	data4 = pd.read_csv("output/lgb.csv")

	data1 = data1.rename(columns={"pm": "Holt_pm"})
	data2 = data2.rename(columns={"pm": "moment_pm"})
	data3 = data3.rename(columns={"pm": "linear_pm"})
	data4 = data4.rename(columns={"pm": "lgb_pm"})

	data = data1.merge(data2, on=["session_id", "rank"], how="left")
	data = data.merge(data3, on=["session_id", "rank"], how="left")
	data = data.merge(data4, on=["session_id", "rank"], how="left")

	data["pm"] = list(map(
		lambda x, y: 0.6*x+0.4*y, data["Holt_pm"], data["linear_pm"]
	))  # 0.24225
	data["pm"] = list(map(
		lambda x, y: 0.2 * x + 0.8 * y, data["pm"], data["lgb_pm"]
	))  # 0.21244

	data = data.loc[:, ["session_id", "rank", "pm"]]
	data.to_csv("output/sub.csv", index=False)


def get_best():
	data2 = pd.read_csv("output/lgb_rmse_seed1.csv")
	data1 = pd.read_csv("output/lgb_mae_seed1.csv")
	data3 = pd.read_csv("output/lgb_rmse_seed2022.csv")
	data4 = pd.read_csv("output/lgb_mae_seed2022.csv")
	data5 = pd.read_csv("output/Holt.csv")
	data6 = pd.read_csv("output/linear.csv")
	data7 = pd.read_csv("output/cat.csv")

	data1 = data1.rename(columns={"pm": "lgb_mae_seed1_pm"})
	data2 = data2.rename(columns={"pm": "lgb_rmse_seed1_pm"})
	data3 = data3.rename(columns={"pm": "lgb_mae_seed2022_pm"})
	data4 = data4.rename(columns={"pm": "lgb_rmse_seed2022_pm"})
	data5 = data5.rename(columns={"pm": "holt_pm"})
	data6 = data6.rename(columns={"pm": "linear_pm"})
	data7 = data7.rename(columns={"pm": "cat_pm"})

	data = data1.merge(data2, on=["session_id", "rank"], how="left")
	data = data.merge(data3, on=["session_id", "rank"], how="left")
	data = data.merge(data4, on=["session_id", "rank"], how="left")
	data = data.merge(data5, on=["session_id", "rank"], how="left")
	data = data.merge(data6, on=["session_id", "rank"], how="left")
	data = data.merge(data7, on=["session_id", "rank"], how="left")
	data["pm1"] = list(map(
		lambda x, y: 0.6 * x + 0.4 * y, data["holt_pm"], data["linear_pm"]
	))  # 0.24225
	data["pm2"] = list(map(
		lambda x, y, z, k: (x+y+z+k)/4, data["lgb_mae_seed2022_pm"], data["lgb_rmse_seed2022_pm"],data["lgb_mae_seed1_pm"],data["lgb_rmse_seed1_pm"]
	))  # 0.20962
	data["pm"] = list(map(
		lambda x, y: (0.785 * x + 0.215 * y), data["pm2"], data["pm1"]
	))  # 0.20953 四个lgb+(linear+holt)
	data["pm"] = list(map(
		lambda x, y, z, j, k: 0.3*x+0.2*y+0.2*k+0.15*z+0.15*j, data["lgb_mae_seed2022_pm"], data["lgb_rmse_seed2022_pm"],
		data["holt_pm"], data["linear_pm"], data["cat_pm"]
	))  # 0.21061
	data["pm"] = list(map(
		lambda x, y: 0.6 * x + 0.4 * y, data["lgb_mae_seed2022_pm"], data["cat_pm"]
	))  # 0.21061
	data["pm"] = list(map(
		lambda x, y: 0.8 * x + 0.2 * y, data["lgb_mae_seed2022_pm"], data["pm1"]
	))  # 0.21061
	# data["pm"] = list(map(
	# 	lambda x, y, z, k: 0.225*x+0.225*y+0.275*z+0.275*k, data["lgb_mae_seed2022_pm"], data["lgb_rmse_seed2022_pm"],data["lgb_mae_seed1_pm"],data["lgb_rmse_seed1_pm"]
	# ))  # 0.21438
	# data["pm"] = list(map(
	# 		lambda x, y: (0.8 * x + 0.2 * y), data["lgb_mae_seed1_pm"], data["pm1"]
	# 	))  # 0.21314
	# data["pm2"] = list(map(
	# 	lambda x, y: (x+y)/2, data["lgb_mae_seed1_pm"],data["lgb_rmse_seed1_pm"]
	# ))  # 0.21501
	# data["pm"] = list(map(
	# 	lambda x, y: (0.8 * x + 0.2 * y), data["pm2"], data["pm1"]
	# )) # 0.21277

	data = data.loc[:, ["session_id", "rank", "pm"]]
	data.to_csv("output/sub.csv", index=False)

def get_test():
	data1 = pd.read_csv("output/lgb_mae_seed2022.csv")
	data2 = pd.read_csv("output/linear.csv")
	data3 = pd.read_csv("output/Holt.csv")

	data1 = data1.rename(columns={"pm": "lgb_mae_seed2022_pm"})
	data2 = data2.rename(columns={"pm": "linear_pm"})
	data3 = data3.rename(columns={"pm": "holt_pm"})

	data = data1.merge(data2, on=["session_id", "rank"], how="left")
	data = data.merge(data3, on=["session_id", "rank"], how="left")
	data["pm1"] = list(map(
		lambda x, y: 0.6 * x + 0.4 * y, data["holt_pm"], data["linear_pm"]
	))  # 0.24225
	id = [18,19,32,33,42,45,71,79,91,95,105,117,124,147,156,160,180,183,203,204,213,241,242,251,257,279,88,87,75,182,160]
	data1 = data[data["session_id"].isin(id)]
	data2 = data[~data["session_id"].isin(id)]
	data1["pm"] = data1["linear_pm"]
	data2["pm"] = data2["lgb_mae_seed2022_pm"]

	sub = pd.concat([data1.loc[:, ["session_id", "rank", "pm"]], data2.loc[:, ["session_id", "rank", "pm"]]], axis=0)
	sub.to_csv("output/sub.csv", index=False)


def get_test1():
	data1 = pd.read_csv("output/lgb_mae_seed2022.csv")
	data2 = pd.read_csv("output/linear.csv")

	data1 = data1.rename(columns={"pm": "lgb_mae_seed2022_pm"})
	data2 = data2.rename(columns={"pm": "linear_pm"})

	data = data1.merge(data2, on=["session_id", "rank"], how="left")
	data["pm"] = 0
	with open("output/weight_dict.json") as f:
		weight = json.load(f)
	for k, v in weight.items():
		data_1 = data[data["session_id"] == int(k)]
		data_1["pm"] = v[0]*data_1["lgb_mae_seed2022_pm"] + v[1]*data_1["linear_pm"]
		data.loc[data_1.index, ["pm"]] = data_1["pm"]
	data = data.loc[:, ["session_id", "rank", "pm"]]
	data.to_csv("output/sub.csv", index=False)


if __name__ == '__main__':
	# data = pd.read_csv("data/train.csv")
	# data = data.sort_values(["session_id", "rank"], ascending=False)
	# data["pm_true"] = data["pm"]
	# data["diff"] = data.groupby(["session_id"])["pm"].diff(1)
	# data.to_csv("output/data.csv", index=False)
	get_test()