#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: stack_train.py
@time: 2022/8/24 10:20
@version:
@desc: 
"""
import pandas as pd
import numpy as np
import warnings

from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm
from pmdarima.arima import auto_arima
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoLars

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


def one_model(clf, train_x, train_y, test_x, val_x, val_y, flag):
	train_matrix = clf.Dataset(train_x, label=train_y)
	valid_matrix = clf.Dataset(val_x, label=val_y)
	if flag == 0:
		params = {
			'boosting_type': 'gbdt',
			'objective': 'regression_l1',  # 回归问题
			'metric': 'mae',  # 评价指标
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
	elif flag == 1:
		params = {
			'boosting_type': 'gbdt',
			'objective': 'regression_l1',  # 回归问题
			'metric': 'mae',  # 评价指标
			'min_child_weight': 3,
			'num_leaves': 2 ** 3,
			'lambda_l2': 10,
			'feature_fraction': 0.9,
			'bagging_fraction': 0.9,
			'bagging_freq': 10,
			'learning_rate': 0.01,
			'seed': 1,
			# 'max_depth': 3,
			'verbose': -1,
			'n_jobs': -1
		}
	elif flag == 2:
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
	elif flag == 3:
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
			'seed': 1,
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
	lgb_train, lgb_test = one_model(lgb, x_train, y_train, x_test, args[0], args[1], args[2])
	return lgb_train, lgb_test


def get_result():
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
		lgb_train, lgb_test = lgb_model(train_data_x, train_data_y, test_data_x, val_data_x, val_data_y, 0)
		# 数据还原
		test_data["pre"] = lgb_test
		# 将预测结果拼接回原数据
		sum_data_index = data_i_[data_i_["rank"] == (26 - i - 1)].index
		data.loc[sum_data_index, "diff"] = lgb_test
		data.loc[sum_data_index, "pm"] = list(map(lambda x, y: x + y, test_data["pm"], test_data["pre"]))

	data.to_csv("output/lgb_mae_seed2022_train_pre.csv", index=False)
	err_data = data[data["rank"] < 25]
	err_pred = err_data["pm"]
	err_trud = err_data["pm_true"]
	print(f"总体损失为:", mean_absolute_error(err_trud, err_pred))

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
		lgb_train, lgb_test = lgb_model(train_data_x, train_data_y, test_data_x, val_data_x, val_data_y, 1)
		# 数据还原
		test_data["pre"] = lgb_test
		# 将预测结果拼接回原数据
		sum_data_index = data_i_[data_i_["rank"] == (26 - i - 1)].index
		data.loc[sum_data_index, "diff"] = lgb_test
		data.loc[sum_data_index, "pm"] = list(map(lambda x, y: x + y, test_data["pm"], test_data["pre"]))

	data.to_csv("output/lgb_mae_seed1_train_pre.csv", index=False)
	err_data = data[data["rank"] < 25]
	err_pred = err_data["pm"]
	err_trud = err_data["pm_true"]
	print(f"总体损失为:", mean_absolute_error(err_trud, err_pred))

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
		lgb_train, lgb_test = lgb_model(train_data_x, train_data_y, test_data_x, val_data_x, val_data_y, 2)
		# 数据还原
		test_data["pre"] = lgb_test
		# 将预测结果拼接回原数据
		sum_data_index = data_i_[data_i_["rank"] == (26 - i - 1)].index
		data.loc[sum_data_index, "diff"] = lgb_test
		data.loc[sum_data_index, "pm"] = list(map(lambda x, y: x + y, test_data["pm"], test_data["pre"]))

	data.to_csv("output/lgb_rmse_seed2022_train_pre.csv", index=False)
	err_data = data[data["rank"] < 25]
	err_pred = err_data["pm"]
	err_trud = err_data["pm_true"]
	print(f"总体损失为:", mean_absolute_error(err_trud, err_pred))

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
		lgb_train, lgb_test = lgb_model(train_data_x, train_data_y, test_data_x, val_data_x, val_data_y, 3)
		# 数据还原
		test_data["pre"] = lgb_test
		# 将预测结果拼接回原数据
		sum_data_index = data_i_[data_i_["rank"] == (26 - i - 1)].index
		data.loc[sum_data_index, "diff"] = lgb_test
		data.loc[sum_data_index, "pm"] = list(map(lambda x, y: x + y, test_data["pm"], test_data["pre"]))

	data.to_csv("output/lgb_rmse_seed1_train_pre.csv", index=False)
	err_data = data[data["rank"] < 25]
	err_pred = err_data["pm"]
	err_trud = err_data["pm_true"]
	print(f"总体损失为:", mean_absolute_error(err_trud, err_pred))


if __name__ == '__main__':
	if_train = False
	if if_train:
		get_result()
	else:
		print("训练集测试")
		lgb_rmse_seed1 = pd.read_csv("output/lgb_rmse_seed1_train_pre.csv")
		lgb_rmse_seed2022 = pd.read_csv("output/lgb_rmse_seed2022_train_pre.csv")
		lgb_mae_seed1 = pd.read_csv("output/lgb_mae_seed1_train_pre.csv")
		lgb_mae_seed2022 = pd.read_csv("output/lgb_mae_seed2022_train_pre.csv")
		cat = pd.read_csv("output/cat_train_pre.csv")
		linear = pd.read_csv("output/liearn_train_pre.csv")
		holt = pd.read_csv("output/holt_train_pre.csv")

		lgb_rmse_seed1 = lgb_rmse_seed1[lgb_rmse_seed1["rank"] < 25]
		lgb_rmse_seed2022 = lgb_rmse_seed2022[lgb_rmse_seed2022["rank"] < 25]
		lgb_mae_seed1 = lgb_mae_seed1[lgb_mae_seed1["rank"] < 25]
		lgb_mae_seed2022 = lgb_mae_seed2022[lgb_mae_seed2022["rank"] < 25]
		cat = cat[cat["rank"] < 25]
		linear = linear[linear["rank"] < 25]
		holt = holt[holt["rank"] < 25]

		train_x = pd.concat(
			[lgb_rmse_seed1.loc[:, ["session_id", "pm"]], lgb_rmse_seed2022["pm"], lgb_mae_seed1["pm"],
			 lgb_mae_seed2022["pm"], cat["pm"],linear["pm"],holt["pm"]
			 ], axis=1
		)
		train_x.columns = ["session_id", "pm_1", "pm_2", "pm_3", "pm_4", "pm_5","pm_6","pm_7"]
		train_y = lgb_rmse_seed1.loc[:, ["session_id", "pm_true"]]

		print("未stack之前的损失为")
		print(mean_absolute_error(lgb_rmse_seed1["pm"], train_y["pm_true"]))
		print(mean_absolute_error(lgb_rmse_seed2022["pm"], train_y["pm_true"]))
		print(mean_absolute_error(lgb_mae_seed1["pm"], train_y["pm_true"]))
		print(mean_absolute_error(lgb_mae_seed2022["pm"], train_y["pm_true"]))
		print(mean_absolute_error(cat["pm"], train_y["pm_true"]))

		print("测试集")
		lgb_rmse_seed1_test = pd.read_csv("output/lgb_rmse_seed1.csv")
		lgb_rmse_seed2022_test = pd.read_csv("output/lgb_rmse_seed2022.csv")
		lgb_mae_seed1_test = pd.read_csv("output/lgb_mae_seed1.csv")
		lgb_mae_seed2022_test = pd.read_csv("output/lgb_mae_seed2022.csv")
		cat_test = pd.read_csv("output/cat.csv")
		linear_test = pd.read_csv("output/linear.csv")
		holt_test = pd.read_csv("output/Holt.csv")

		test_x = pd.concat(
			[lgb_rmse_seed1_test.loc[:, ["session_id", "pm"]], lgb_rmse_seed2022_test["pm"], lgb_mae_seed1_test["pm"],
			 lgb_mae_seed2022_test["pm"], cat_test["pm"],linear_test["pm"],holt_test["pm"]
			 ], axis=1
		)
		test_x.columns = ["session_id", "pm_1", "pm_2", "pm_3", "pm_4", "pm_5","pm_6","pm_7"]
		sub = pd.read_csv("output/lgb_mae_seed2022.csv")

		for id in train_x["session_id"].unique():
			x = train_x[train_x["session_id"] == id]
			y = train_y[train_y["session_id"] == id]
			del x["session_id"]
			del y["session_id"]

			te_x = test_x[test_x["session_id"] == id]
			del te_x["session_id"]
			model = LinearRegression(fit_intercept=False,positive=True).fit(x, y)
			if model.score(x,y) >= 0.9:
				print("更改",id)
				a = model.coef_
				print(a[0])
				pre = model.predict(te_x)
				sub_index = te_x.index
				sub.loc[sub_index, ["pm"]] = pre

		sub.to_csv("output/test_sub.csv", index=False)