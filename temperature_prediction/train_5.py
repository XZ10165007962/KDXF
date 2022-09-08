#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: train_5.py
@time: 2022/8/26 10:20
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
	data_["diff"] = data_.groupby(["session_id"])["pm"].diff(1)
	# 数据归一化
	data_["diff_max"] = data_.groupby(["session_id"])["diff"].max()
	data_["diff_min"] = data_.groupby(["session_id"])["diff"].min()
	data_["diff"] = (data_["diff"] - data_["diff_min"]) / (data_["diff_max"] - data_["diff_min"])
	feat_col = []
	for i in range(feat_lag + 1):
		data_[f"diff_log{i}"] = data_.groupby(["session_id"])["diff"].shift(i)
		feat_col.append(f"diff_log{i}")
	data_["label"] = data_.groupby(["session_id"])["diff"].shift(-label_lag)
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
		"diff_0_2_sum", "diff_0_2_mean", "diff_0_2_std",
		"diff_3_5_sum", "diff_3_5_mean", "diff_3_5_std"
	])
	data_["diff_0_2_3_5"] = data_["diff_0_2_mean"] - data_["diff_3_5_mean"]
	data_["diff_0_2_3_5_gep"] = data_["diff_0_2_3_5"] / data_["diff_0_2_mean"]
	feat_col.append("diff_0_2_3_5")
	feat_col.append("diff_0_2_3_5_gep")
	return data_, feat_col


def cat_model(clf, train_x, train_y, test_x, val_x, val_y):
	params = {'learning_rate': 0.01, 'depth': 10, 'l2_leaf_reg': 10, 'bootstrap_type': "Bayesian",
			  'od_type': 'Iter', 'od_wait': 50, 'random_seed': 2022, 'allow_writing_files': False,
			  'eval_metric': "MAE", "random_strength": 10, "bagging_temperature": 2}

	model = clf(iterations=20000, **params)
	model.fit(train_x, train_y, eval_set=(val_x, val_y),
			  cat_features=[], use_best_model=True, verbose=3000)

	val_pred = model.predict(val_x)
	test_pred = model.predict(test_x)
	print("score:", mean_absolute_error(val_y["label"].to_list(), val_pred.tolist()))
	return val_pred, test_pred


def lgb_model(x_train, y_train, x_test, *args):
	lgb_train, lgb_test = cat_model(CatBoostRegressor, x_train, y_train, x_test, args[0], args[1])
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
		# 数据还原
		test_data["pre"] = lgb_test
		# 将预测结果拼接回原数据
		sum_data_index = data_i_[data_i_["rank"] == (26 - i - 1)].index
		data.loc[sum_data_index, "diff"] = lgb_test
		data.loc[sum_data_index, "pm"] = list(map(lambda x,y: x+y, test_data["pm"],test_data["pre"]))

	data.to_csv("output/cat_train_pre.csv", index=False)
	err_data = data[data["rank"] < 25]
	err_pred = err_data["pm"]
	err_trud = err_data["pm_true"]
	print(f"总体损失为:", mean_absolute_error(err_trud, err_pred))  # 0.2950
