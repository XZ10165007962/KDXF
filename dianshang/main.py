#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: main.py
@time: 2022/8/1 22:24
@version:
@desc: 
"""
import warnings
import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import copy

import transform

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 200)


def duild_model(train_flag, clf_name, clf, train_x, train_y, cate_feat, **kwargs):
	if train_flag:
		if clf_name == 'lgb':
			val_x = kwargs.get("val_x")
			val_y = kwargs.get("val_y")
			train_matrix = clf.Dataset(train_x, label=train_y)
			valid_matrix = clf.Dataset(val_x, label=val_y)
			params = {
				'boosting_type': 'gbdt',
				'objective': 'regression_l1',  # 回归问题
				'metric': 'rmse',  # 评价指标
				'min_child_weight': 10,
				'num_leaves': 2 ** 5,
				'lambda_l2': 10,
				'feature_fraction': 0.6,
				'bagging_fraction': 0.8,
				'bagging_freq': 10,
				'learning_rate': 0.05,
				'seed': 2022,
				# 'max_depth': 3,
				'verbose': -1,
				'n_jobs': -1
			}
			model = clf.train(
				params, train_set=train_matrix, num_boost_round=50000, valid_sets=[train_matrix, valid_matrix],
				categorical_feature=cate_feat, verbose_eval=3000, early_stopping_rounds=3000
			)
			val_pred = model.predict(data=val_x, num_iteration=model.best_iteration)
			# test_pred = model.predict(data=test_x, num_iteration=model.best_iteration)
			print(list(
				sorted(
					zip(train_x.columns, model.feature_importance('gain')),
					key=lambda x: x[1], reverse=True)
			)[:20])
			# print("%s_score:" % clf_name, mean_absolute_error(val_y, val_pred))
			# print("%s_score:" % clf_name, mean_squared_error(val_y, val_pred))
		return val_pred
	else:
		if clf_name == 'lgb':
			test_x = kwargs.get("test_x")
			train_matrix = clf.Dataset(train_x, label=train_y)
			params = {
				'boosting_type': 'gbdt',
				'objective': 'regression_l1',  # 回归问题
				'metric': 'rmse',  # 评价指标
				'min_child_weight': 10,
				'num_leaves': 2 ** 5,
				'lambda_l2': 10,
				'feature_fraction': 0.6,
				'bagging_fraction': 0.8,
				'bagging_freq': 10,
				'learning_rate': 0.05,
				'seed': 2022,
				'max_depth': 6,
				'verbose': -1,
				'n_jobs': -1
			}
			model = clf.train(
				params, train_set=train_matrix, num_boost_round=50000, categorical_feature=cate_feat, verbose_eval=3000
			)
			test_pred = model.predict(data=test_x, num_iteration=model.best_iteration)
			print(list(
				sorted(
					zip(train_x.columns, model.feature_importance('gain')),
					key=lambda x: x[1], reverse=True)
			)[:20])

		return test_pred


def lgb_model(train_flag, x_train, y_train, cate_feat, **kwargs):
	if train_flag:
		val_x = kwargs.get("val_x")
		val_y = kwargs.get("val_y")
		lgb_train = duild_model(train_flag, 'lgb', lgb, x_train, y_train, cate_feat, val_x=val_x, val_y=val_y)
		return lgb_train
	else:
		test_x = kwargs.get("test_x")
		lgb_test = duild_model(train_flag, 'lgb', lgb, x_train, y_train, cate_feat, test_x=test_x)
		return lgb_test


if __name__ == '__main__':

	# rec为1则递归预测，rec为0则直接预测
	rec = 0
	train_flag = False
	print("读取数据")
	data_ = pd.read_csv("output/trans_data.csv")
	data_["pre"] = np.zeros(data_.shape[0])
	if train_flag:
		val_list = [355, 356, 357, 358, 359, 360, 361]
		if rec == 0:
			for ind, val_idx in enumerate(val_list):
				columns = ["scaler_qty"]
				data_ = transform.lag_feat(data_, columns, lags=15, shift=ind)

				# 删除还未售卖的数据
				# data_index = data_[data_["qty_shichang"] > 30].index
				# use_id = set(data_["商品id"].values)
				# data_ = data_[data_["商品id"].isin(use_id)]

				data = copy.deepcopy(data_)
				del data["label"]
				# del data["Q1"]
				# del data["Q3"]
				del data["mean"]
				del data["pre"]
				del data["直播销量"]
				del data["视频销量"]
				del data["浏览量"]
				del data["抖音转化率"]
				del data["视频个数"]
				del data["视频达人"]
				del data["直播达人"]
				del data["直播个数"]
				del data["商品id"]
				label_col = "scaler_qty"
				cate_feat = []
				# for c in cate_feat:
				# 	data[c] = data[c].astype('category')
				print("数据切分")
				train_x = data[data["dayofyear"] > val_idx]
				del train_x[label_col]
				train_y = data[data["dayofyear"] > val_idx][label_col]

				val_x = data[data["dayofyear"] == val_idx]
				del val_x[label_col]
				val_y = data[data["dayofyear"] == val_idx][label_col]

				print("构建模型")
				print(f"****************{val_idx}******************")
				val_pred = lgb_model(train_flag, train_x, train_y, cate_feat, val_x=val_x, val_y=val_y)
				data_.loc[val_x.index, ["pre"]] = val_pred
			data_.to_csv("output/pre_data.csv", encoding="utf_8_sig", index=False)

	else:
		test_list = [362, 363, 364, 365, 1, 2, 3]
		if rec == 0:
			for ind, test_idx in enumerate(test_list):
				columns = ["scaler_qty"]
				data_ = transform.lag_feat(data_, columns, lags=15, shift=ind)

				# 删除还未售卖的数据
				# data_index = data_[data_["qty_shichang"] > 30].index
				# use_id = set(data_["商品id"].values)
				# data_ = data_[data_["商品id"].isin(use_id)]

				data = copy.deepcopy(data_)
				del data["label"]
				del data["mean"]
				del data["pre"]
				del data["直播销量"]
				del data["视频销量"]
				del data["浏览量"]
				del data["抖音转化率"]
				del data["视频个数"]
				del data["视频达人"]
				del data["直播达人"]
				del data["直播个数"]
				del data["商品id"]
				label_col = "scaler_qty"
				cate_feat = []
				# for c in cate_feat:
				# 	data[c] = data[c].astype('category')

				print("数据切分")
				train_x = data[data["dayofyear"] >= 361]
				del train_x[label_col]
				train_y = data[data["dayofyear"] >= 361][label_col]

				test_x = data[data["dayofyear"] == test_idx]
				del test_x[label_col]

				print("构建模型")
				print(f"****************{test_idx}******************")
				test_pred = lgb_model(train_flag, train_x, train_y, cate_feat, test_x=test_x)
				data_.loc[test_x.index, ["pre"]] = test_pred
			data_.to_csv("output/pre_data.csv", encoding="utf_8_sig", index=False)
		data_["pre"] = data_["pre"] * data_["mean"]
		data_['pre'] = list(map(lambda x: 0 if x <= 0 else (2 ** (x)) - 1, data_['pre']))
		sub_m = data_[data_["dayofyear"].isin([362, 363, 364, 365, 1, 2, 3])]
		sub_m = sub_m.groupby(['商品id']).agg({'pre': 'mean'}).reset_index().rename(
			columns={'pre': '未来一周天均销量'})
		submit = pd.read_csv('data/电商销量预测挑战赛公开数据/提交示例.csv')
		submit.drop(['未来一周天均销量'], axis=1, inplace=True)
		submit = pd.merge(submit, sub_m, on='商品id')
		submit.to_csv('output/submit.csv', index=False)
