#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: train_7.py
@time: 2022/8/28 20:45
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


def trans_data(data_, feat_lag=4, label_lag=1):
	data_["diff"] = data_.groupby(["session_id"])["pm"].diff(label_lag)
	feat_col = []
	for i in range(feat_lag):
		data_[f"diff_log{i}"] = data_.groupby(["session_id"])["diff"].shift(i)
		feat_col.append(f"diff_log{i}")
	data_["label"] = data_.groupby(["session_id"])["diff"].shift(-1)
	data_["diff_0_2_sum"] = data_.loc[:, "diff_log0":"diff_log2"].sum(1)
	data_["diff_0_2_mean"] = data_.loc[:, "diff_log0":"diff_log2"].mean(1)
	data_["diff_0_2_max"] = data_.loc[:, "diff_log0":"diff_log2"].max(1)
	data_["diff_0_2_min"] = data_.loc[:, "diff_log0":"diff_log2"].min(1)
	data_["diff_0_2_std"] = data_.loc[:, "diff_log0":"diff_log2"].std(1)
	# data_["diff_3_5_sum"] = data_.loc[:, "diff_log3":"diff_log5"].sum(1)
	# data_["diff_3_5_mean"] = data_.loc[:, "diff_log3":"diff_log5"].mean(1)
	# data_["diff_3_5_max"] = data_.loc[:, "diff_log3":"diff_log5"].max(1)
	# data_["diff_3_5_min"] = data_.loc[:, "diff_log3":"diff_log5"].min(1)
	# data_["diff_3_5_std"] = data_.loc[:, "diff_log3":"diff_log5"].std(1)
	# data_["diff_0_5_mean"] = data_.loc[:, "diff_log0":"diff_log5"].mean(1)
	# data_["diff_0_2_0_5_mean"] = data_["diff_0_2_mean"] / data_["diff_0_5_mean"]
	# data_["diff_3_5_0_5_mean"] = data_["diff_3_5_mean"] / data_["diff_0_5_mean"]
	feat_col.extend([
		"diff_0_2_sum", "diff_0_2_mean", "diff_0_2_std", "diff_0_2_max", "diff_0_2_min"
	])
	# data_["diff_0_2_3_5"] = data_["diff_0_2_mean"] - data_["diff_3_5_mean"]
	# data_["diff_0_2_3_5_gep"] = data_["diff_0_2_3_5"] / data_["diff_0_2_mean"]
	# feat_col.append("diff_0_2_3_5")
	# feat_col.append("diff_0_2_3_5_gep")
	# data_["fangxiang"] = list(map(lambda x: 0 if x < 0 else 1, data_["diff"]))
	# data_["fangxiang"] = data_["fangxiang"].astype("category")
	# feat_col.append("fangxiang")
	# simple = data_.groupby(["session_id", "fangxiang"])["diff"].mean().reset_index()
	# simple.columns = ["session_id", "fangxiang", "fx_diff_mean"]
	# data_ = data_.merge(simple, on=["session_id", "fangxiang"], how="left")
	# feat_col.append("fx_diff_mean")
	# simple = data_.groupby(["session_id", "fangxiang"])["diff"].count().reset_index()
	# simple.columns = ["session_id", "fangxiang", "fx_diff_count"]
	# data_ = data_.merge(simple, on=["session_id", "fangxiang"], how="left")
	# feat_col.append("fx_diff_count")
	# simple = data_.groupby(["session_id"])["diff"].sum().reset_index()
	# simple.columns = ["session_id", "diff_mean"]
	# data_ = data_.merge(simple, on=["session_id"], how="left")
	# feat_col.append("diff_mean")

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

	print("score:", mean_absolute_error(val_y["label"].to_list(), val_pred.tolist()))
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
		data_i = data[data["rank"] >= (23 + i)]
		data_i_, col = trans_data(data_i, feat_lag=12, label_lag=i)
		train_data = data_i_[data_i_["rank"] > (25 + i)]
		train_data_x = train_data.loc[:, col]
		train_data_y = train_data.loc[:, label_col]

		val_data = data_i_[data_i_["rank"] == (25 + i)]
		val_data_x = val_data.loc[:, col]
		val_data_y = val_data.loc[:, label_col]
		test_data = data_i_[data_i_["rank"] == (24 + i)]
		test_data_x = test_data.loc[:, col]
		lgb_train, lgb_test = lgb_model(train_data_x, train_data_y, test_data_x, val_data_x, val_data_y)
		# 反归一化
		test_data["pre"] = lgb_test

		# 将预测结果拼接回原数据
		sum_data_index = data[data["rank"] == (25 - i)].index
		data.loc[sum_data_index, "pm"] = list(map(lambda x,y: x+y, test_data["pm"],test_data["pre"]))
		print("score:", mean_absolute_error(data.loc[sum_data_index, "pm"].to_list(),
											data.loc[sum_data_index, "pm_true"].tolist()))

	data.to_csv("output/lgb_train_pre.csv", index=False)
	err_data = data[data["rank"] < 25]
	err_pred = err_data["pm"]
	err_trud = err_data["pm_true"]
	print(f"总体损失为:", mean_absolute_error(err_trud, err_pred))  # 0.2950
