#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: train_6.py
@time: 2022/8/28 14:42
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
import lightgbm as lgb
from catboost import CatBoostRegressor

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


def trans_data(data_, feat_lag=12, label_lag=1):
	# 归一化
	# simple = data_.groupby(["session_id"], sort=False)["diff"].max().reset_index()
	# simple = pd.DataFrame(simple).rename(columns={"diff":"diff_max"})
	# data_ = data_.merge(simple, on=["session_id"], how="left")
	# simple = data_.groupby(["session_id"], sort=False)["diff"].min().reset_index()
	# simple = pd.DataFrame(simple).rename(columns={"diff":"diff_min"})
	# data_ = data_.merge(simple, on=["session_id"], how="left")
	# data_["diff"] = (data_["diff"] - data_["diff_min"]) / (data_["diff_max"] - data_["diff_min"])
	# simple = data_.groupby(["session_id"], sort=False)["diff"].mean().reset_index()
	# simple = pd.DataFrame(simple).rename(columns={"diff":"diff_mean"})
	# data_ = data_.merge(simple, on=["session_id"], how="left")
	# data_["diff"] = data_["diff"] / data_["diff_mean"]
	feat_col = []
	for i in range(feat_lag + 1):
		data_[f"diff_log{i}"] = data_.groupby(["session_id"])["pm"].shift(i)
		feat_col.append(f"diff_log{i}")
	data_["label"] = data_.groupby(["session_id"])["pm"].shift(-label_lag)
	data_["diff_0_2_sum"] = data_.loc[:, "diff_log0":"diff_log2"].sum(1)
	data_["diff_0_2_mean"] = data_.loc[:, "diff_log0":"diff_log2"].mean(1)
	data_["diff_0_2_max"] = data_.loc[:, "diff_log0":"diff_log2"].max(1)
	data_["diff_0_2_min"] = data_.loc[:, "diff_log0":"diff_log2"].min(1)
	data_["diff_0_2_std"] = data_.loc[:, "diff_log0":"diff_log2"].std(1)
	data_["diff_3_5_sum"] = data_.loc[:, "diff_log3":"diff_log5"].sum(1)
	data_["diff_3_5_mean"] = data_.loc[:, "diff_log3":"diff_log5"].mean(1)
	data_["diff_3_5_max"] = data_.loc[:, "diff_log3":"diff_log5"].max(1)
	data_["diff_3_5_min"] = data_.loc[:, "diff_log3":"diff_log5"].min(1)
	data_["diff_3_5_std"] = data_.loc[:, "diff_log3":"diff_log5"].std(1)
	data_["diff_0_5_mean"] = data_.loc[:, "diff_log0":"diff_log5"].mean(1)
	data_["diff_0_2_0_5_mean"] = data_["diff_0_2_mean"] / data_["diff_0_5_mean"]
	data_["diff_3_5_0_5_mean"] = data_["diff_3_5_mean"] / data_["diff_0_5_mean"]
	feat_col.extend([
		"diff_0_2_sum", "diff_0_2_mean", "diff_0_2_std", "diff_0_2_max", "diff_0_2_min", "diff_0_2_0_5_mean",
		"diff_3_5_sum", "diff_3_5_mean", "diff_3_5_std", "diff_3_5_max", "diff_3_5_min", "diff_3_5_0_5_mean"
	])
	data_["diff_0_2_3_5"] = data_["diff_0_2_mean"] - data_["diff_3_5_mean"]
	data_["diff_0_2_3_5_gep"] = data_["diff_0_2_3_5"] / data_["diff_0_2_mean"]
	feat_col.append("diff_0_2_3_5")
	feat_col.append("diff_0_2_3_5_gep")
	data_["k"] = data_.groupby(["session_id"])["pm"].diff(1)
	data_["cont"] = data_["pm"] + data_["k"] * data_["rank"]
	data_["pm_1"] = list(map(lambda x,y,z: (-x)*(y-1)+z, data_["k"],data_["rank"],data_["cont"]))

	feat_col.append("k")
	feat_col.append("cont")
	feat_col.append("pm_1")
	return data_, feat_col


def one_model(clf, train_x, train_y, test_x, val_x, val_y):
	train_matrix = clf.Dataset(train_x, label=train_y)
	valid_matrix = clf.Dataset(val_x, label=val_y)
	params = {
		'boosting_type': 'gbdt',
		'objective': 'regression_l1',  # 回归问题
		'metric': 'rmse',  # 评价指标
		'min_child_weight': 3,
		'num_leaves': 2 ** 3,
		'lambda_l2': 10,
		'feature_fraction': 0.9,
		'bagging_fraction': 0.9,
		'bagging_freq': 10,
		'learning_rate': 0.01,
		'seed': 2022,
		# 'max_depth': 3,
		'verbose': -1,
		'n_jobs': -1
	}
	model = clf.train(
		params, train_set=train_matrix, num_boost_round=50000, valid_sets=[train_matrix, valid_matrix],
		categorical_feature=[], verbose_eval=3000, early_stopping_rounds=3000
	)
	val_pred = model.predict(data=val_x, num_iteration=model.best_iteration)
	test_pred = model.predict(data=test_x, num_iteration=model.best_iteration)
	print(list(
		sorted(
			zip(train_x.columns, model.feature_importance('gain')),
			key=lambda x: x[1], reverse=True)
	))

	return val_pred, test_pred


def lgb_model(x_train, y_train, x_test, *args):
	lgb_train, lgb_test = one_model(lgb, x_train, y_train, x_test, args[0], args[1])
	return lgb_train, lgb_test


if __name__ == '__main__':
	data = read_data()
	data["pre"] = 0
	data["pre1"] = 0
	label_col = ["label"]
	for i in range(1, 13):
		print(f"---------------{i}---------------")
		data_i_, col = trans_data(data, feat_lag=12, label_lag=1)
		train_data = data_i_[data_i_["rank"] > (27 - i)]
		train_data_x = train_data.loc[:, col]
		train_data_y = train_data.loc[:, label_col]

		val_data = data_i_[data_i_["rank"] == (27 - i)]
		val_data_x = val_data.loc[:, col]
		val_data_y = val_data.loc[:, label_col]
		test_data = data_i_[data_i_["rank"] == (26 - i)]
		test_data_x = test_data.loc[:, col]
		lgb_train, lgb_test = lgb_model(train_data_x, train_data_y, test_data_x, val_data_x, val_data_y)
		# 反归一化
		test_data["pre"] = lgb_test
		# 将预测结果拼接回原数据
		sum_data_index = data_i_[data_i_["rank"] == (26 - i - 1)].index
		data.loc[sum_data_index, "pm"] = lgb_test
		print("score:", mean_absolute_error(data.loc[sum_data_index, "pm"].to_list(), data.loc[sum_data_index, "pm_true"].tolist()))

	data.to_csv("output/lgb_train_pre.csv", index=False)
	err_data = data[data["rank"] < 25]
	err_pred = err_data["pm"]
	err_trud = err_data["pm_true"]
	print(f"总体损失为:", mean_absolute_error(err_trud, err_pred))  # 0.2950
