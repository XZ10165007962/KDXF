#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: select_weight_cv.py
@time: 2022/8/25 23:34
@version:
@desc: 
"""
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
import json

warnings.filterwarnings('ignore')
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 200)

lgb_rmse_seed1 = pd.read_csv("output/lgb_rmse_seed1_train_pre.csv")
lgb_rmse_seed2022 = pd.read_csv("output/lgb_rmse_seed2022_train_pre.csv")
lgb_mae_seed1 = pd.read_csv("output/lgb_mae_seed1_train_pre.csv")
lgb_mae_seed2022 = pd.read_csv("output/lgb_mae_seed2022_train_pre.csv")
linear = pd.read_csv("output/liearn_train_pre.csv")
holt = pd.read_csv("output/holt_train_pre.csv")

lgb_rmse_seed1 = lgb_rmse_seed1[lgb_rmse_seed1["rank"] < 25]
lgb_rmse_seed2022 = lgb_rmse_seed2022[lgb_rmse_seed2022["rank"] < 25]
lgb_mae_seed1 = lgb_mae_seed1[lgb_mae_seed1["rank"] < 25]
lgb_mae_seed2022 = lgb_mae_seed2022[lgb_mae_seed2022["rank"] < 25]
linear = linear[linear["rank"] < 25]
holt = holt[holt["rank"] < 25]

y_true = lgb_mae_seed2022.loc[:, ["session_id", "pm_true"]]
train_x = pd.concat(
	[lgb_rmse_seed1.loc[:, ["session_id", "pm_true", "pm"]], lgb_rmse_seed2022["pm"], lgb_mae_seed1["pm"],
	 lgb_mae_seed2022["pm"], linear["pm"], holt["pm"]
	 ], axis=1
)
train_x.columns = [
	"session_id", "pm_true", "pm_lgb_rmse_seed1", "pm_lgb_rmse_seed2022", "pm_lgb_mae_seed1",
	"pm_lgb_mae_seed2022", "pm_linear", "pm_holt"
]
train_x["pm1"] = list(map(
		lambda x, y: 0.6 * x + 0.4 * y, train_x["pm_holt"], train_x["pm_linear"]
	))  # 0.24225
w1_list = []
w2_list = []
for total in [0.8, 0.90, 1.0,  1.1]:
	for w1 in np.arange(0, total+0.001, 0.05):
		w2 = total-w1
		w1_list.append(w1)
		w2_list.append(w2)
w1_list = np.array(w1_list)
w2_list = np.array(w2_list)

id_list = train_x["session_id"].unique()
param_dict = {}
for id in id_list:
	data = train_x[train_x["session_id"] == id]
	answer = y_true[y_true["session_id"] == id]
	MAE = 10000000
	W1 = -1
	W2 = -1

	param_list = []
	for w1,w2 in zip(w1_list,w2_list):
		y_pred = w1*data["pm_lgb_mae_seed2022"] + w2*data["pm1"]
		y = answer["pm_true"]
		mae = mean_absolute_error(y, y_pred)
		if mae < MAE:
			MAE = mae
			W1 = w1
			W2 = w2
	param_list.append(W1)
	param_list.append(W2)
	param_dict[str(id)] = param_list
	print(id, ":", W1,W2)

# w1_list = []
# w2_list = []
# w3_list = []
# w4_list = []
# w5_list = []
# w6_list = []
# for total in [0.8, 0.90, 1.0,  1.1]:
# 	for w1 in np.arange(0, total+0.001, 0.1):
# 		for w2 in np.arange(0, total-w1+0.001, 0.1):
# 			for w3 in np.arange(0, total-w1-w2+0.001, 0.1):
# 				for w4 in np.arange(0, total-w1-w2-w3+0.001, 0.1):
# 					for w5 in np.arange(0, total - w1 - w2 - w3 - w4 + 0.001, 0.05):
# 						w6 = total-w1-w2-w3-w4-w5
# 						w1_list.append(w1)
# 						w2_list.append(w2)
# 						w3_list.append(w3)
# 						w4_list.append(w4)
# 						w5_list.append(w5)
# 						w6_list.append(w6)
# w1_list = np.array(w1_list)
# w2_list = np.array(w2_list)
# w3_list = np.array(w3_list)
# w4_list = np.array(w4_list)
# w5_list = np.array(w5_list)
# w6_list = np.array(w6_list)
#
# id_list = train_x["session_id"].unique()
# param_dict = {}
# for id in id_list[1:]:
# 	data = train_x[train_x["session_id"] == id]
# 	answer = y_true[y_true["session_id"] == id]
# 	MAE = 10000000
# 	W1 = -1
# 	W2 = -1
# 	W3 = -1
# 	W4 = -1
# 	W5 = -1
# 	W6 = -1
# 	param_list = []
# 	for w1,w2,w3,w4,w5,w6 in zip(w1_list,w2_list,w3_list,w4_list,w5_list,w6_list):
# 		y_pred = w1*data["pm_lgb_rmse_seed1"] + w2*data["pm_lgb_rmse_seed2022"] + w3*data["pm_lgb_mae_seed1"] +\
# 				 w4*data["pm_lgb_mae_seed2022"] + w5*data["pm_linear"] + w6*data["pm_holt"]
# 		y = answer["pm_true"]
# 		mae = mean_absolute_error(y, y_pred)
# 		if mae < MAE:
# 			MAE = mae
# 			W1 = w1
# 			W2 = w2
# 			W3 = w3
# 			W4 = w4
# 			W5 = w5
# 			W6 = w6
# 	param_list.append(W1)
# 	param_list.append(W2)
# 	param_list.append(W3)
# 	param_list.append(W4)
# 	param_list.append(W5)
# 	param_list.append(W6)
# 	param_dict[id] = param_list
# 	print(id, ":", W1,W2,W3,W4,W5,W6)

with open("output/weight_dict.json", "w") as tf:
	json.dump(param_dict, tf)