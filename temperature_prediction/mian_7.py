#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: mian_7.py
@time: 2022/8/29 9:44
@version:
@desc: 
"""
#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: main_4.py
@time: 2022/8/17 21:44
@version:
@desc: lgb mae:0.20831 rmse:0.2087 seed1 rmse0.2119 mae:0.21089
"""
import pandas as pd
import numpy as np
import warnings

from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from pmdarima.arima import auto_arima
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
import lightgbm as lgb

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


def trans_data(data_, feat_lag=12, label_lag=1):
	# data_["diff"] = data_.groupby(["session_id"])["pm"].diff(1)
	# feat_col = []
	# for i in range(feat_lag+1):
	# 	data_[f"diff_log{i}"] = data_.groupby(["session_id"])["diff"].shift(i)
	# 	feat_col.append(f"diff_log{i}")
	#
	# data_["diff_0_2_sum"] = data_.loc[:, "diff_log0":"diff_log2"].sum(1)
	# data_["diff_0_2_mean"] = data_.loc[:, "diff_log0":"diff_log2"].mean(1)
	# data_["diff_0_2_max"] = data_.loc[:, "diff_log0":"diff_log2"].max(1)
	# data_["diff_0_2_min"] = data_.loc[:, "diff_log0":"diff_log2"].min(1)
	# data_["diff_3_5_sum"] = data_.loc[:, "diff_log3":"diff_log5"].sum(1)
	# data_["diff_3_5_mean"] = data_.loc[:, "diff_log3":"diff_log5"].mean(1)
	# data_["diff_3_5_max"] = data_.loc[:, "diff_log3":"diff_log5"].max(1)
	# data_["diff_3_5_min"] = data_.loc[:, "diff_log3":"diff_log5"].min(1)
	# feat_col.extend([
	# 	"diff_0_2_sum", "diff_0_2_mean", "diff_0_2_max", "diff_0_2_min",
	# 	"diff_3_5_sum", "diff_3_5_mean", "diff_3_5_max", "diff_3_5_min"
	# ])
	# data_["diff_0_2_3_5"] = data_["diff_0_2_mean"] - data_["diff_3_5_mean"]
	# data_["diff_0_2_3_5_gep"] = data_["diff_0_2_3_5"] / data_["diff_0_2_mean"]
	# feat_col.append("diff_0_2_3_5")
	# feat_col.append("diff_0_2_3_5_gep")
	# data_["label"] = data_.groupby(["session_id"])["diff"].shift(-label_lag)

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
	lgb_train, lgb_test = one_model(lgb, x_train, y_train, x_test, args[0], args[1], args[2])
	return lgb_train, lgb_test


if __name__ == '__main__':
	data = read_data()
	data["pre"] = 0
	label_col = ["label"]
	ids = [
		0, 2, 4, 5, 6, 7, 12, 13, 14, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 36, 45, 46, 48, 49,
		50, 51, 52, 53, 54, 55, 57, 59,
		60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 72, 76, 77, 80, 81, 82, 83, 84, 89, 92, 93, 94, 96, 97, 98, 99, 100,
		101, 102, 104, 106, 107,
		109, 110, 111, 112, 113, 114, 115, 116, 117, 119, 121, 123, 125, 126, 128, 129, 130, 131, 132, 133, 134, 136,
		137, 138, 139, 142,
		143, 144, 146, 148, 151, 152, 162, 167, 168, 173, 176, 179, 187, 188, 190, 193, 194, 195, 196, 199, 206, 107,
		208, 210, 214, 215, 216,
		218, 223, 224, 226, 227, 228, 229, 232, 233, 234, 243, 246, 247, 252, 253, 254, 256, 261, 264, 267, 268, 269,
		273, 274, 275, 276,
	]
	data1 = data[data["session_id"].isin(ids)]
	data2 = data[~data["session_id"].isin(ids)]
	for i in range(1, 13):
		print(f"---------------{i}---------------")
		data_i_, col = trans_data(data1, feat_lag=12, label_lag=1)
		train_data = data_i_[data_i_["rank"] > (15 - i)]
		train_data_x = train_data.loc[:, col]
		train_data_y = train_data.loc[:, label_col]

		val_data = data_i_[data_i_["rank"] == (15 - i)]
		val_data_x = val_data.loc[:, col]
		val_data_y = val_data.loc[:, label_col]

		test_data = data_i_[data_i_["rank"] == (14 - i)]
		test_data_x = test_data.loc[:, col]
		lgb_train, lgb_test = lgb_model(train_data_x, train_data_y, test_data_x, val_data_x, val_data_y, 0)

		# 数据还原
		test_data["pre"] = lgb_test
		# 将预测结果拼接回原数据
		sum_data_index = data_i_[data_i_["rank"] == (14 - i - 1)].index
		data.loc[sum_data_index, "diff"] = lgb_test
		data.loc[sum_data_index, "pm"] = list(map(lambda x,y: x+y, test_data["pm"],test_data["pre"]))
		data1.loc[sum_data_index, "diff"] = lgb_test
		data1.loc[sum_data_index, "pm"] = list(map(lambda x, y: x + y, test_data["pm"], test_data["pre"]))
	for i in range(1, 13):
		print(f"---------------{i}---------------")
		data_i_, col = trans_data(data2, feat_lag=12, label_lag=1)
		train_data = data_i_[data_i_["rank"] > (15 - i)]
		train_data_x = train_data.loc[:, col]
		train_data_y = train_data.loc[:, label_col]

		val_data = data_i_[data_i_["rank"] == (15 - i)]
		val_data_x = val_data.loc[:, col]
		val_data_y = val_data.loc[:, label_col]

		test_data = data_i_[data_i_["rank"] == (14 - i)]
		test_data_x = test_data.loc[:, col]
		lgb_train, lgb_test = lgb_model(train_data_x, train_data_y, test_data_x, val_data_x, val_data_y, 0)

		# 数据还原
		test_data["pre"] = lgb_test
		# 将预测结果拼接回原数据
		sum_data_index = data_i_[data_i_["rank"] == (14 - i - 1)].index
		data.loc[sum_data_index, "diff"] = lgb_test
		data.loc[sum_data_index, "pm"] = list(map(lambda x,y: x+y, test_data["pm"],test_data["pre"]))
		data2.loc[sum_data_index, "diff"] = lgb_test
		data2.loc[sum_data_index, "pm"] = list(map(lambda x, y: x + y, test_data["pm"], test_data["pre"]))
	data.to_csv("output/test_lgb_data.csv", index=False)
	sum_data = data[data["rank"] < 13]
	sum_data = sum_data.loc[:, ["session_id", "rank", "pm"]]
	sum_data.to_csv("output/test_lgb.csv", index=False)
