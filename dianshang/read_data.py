#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: read_data.py
@time: 2022/7/24 23:14
@version:
@desc: 
"""
import copy
import json
import math

import numpy as np
import pandas as pd
import os
from tslearn.clustering import TimeSeriesKMeans
from tqdm import tqdm
import datetime

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)


def alldata():
	file_dir = "data/电商销量预测挑战赛公开数据/数据集/"
	all_file_name = os.listdir(file_dir)
	data = None
	d_7 = np.zeros((7, 11))
	d_7 = pd.DataFrame(
		d_7,
		columns=["商品id", "总销量", "时间", "浏览量", "抖音转化率", "视频个数", "直播个数", "直播销量", "视频销量",
				 "视频达人", "直播达人"]
	)
	d_7["时间"] = ["2021/12/28  16:00:00", "2021/12/29  16:00:00", "2021/12/30  16:00:00", "2021/12/31  16:00:00",
				 "2021/1/1  16:00:00", "2021/1/2  16:00:00", "2021/1/3  16:00:00"]
	for _, file in enumerate(all_file_name):
		if _ == 0:
			data = pd.read_csv(os.path.join(file_dir, file))
			d_7["商品id"] = data.loc[:7, ["商品id"]]
		else:
			data_ = pd.read_csv(os.path.join(file_dir, file))
			data = pd.concat([data, data_], ignore_index=True)
			d_7["商品id"] = data_.loc[:7, ["商品id"]]
		data = pd.concat([data, d_7], axis=0)
	data.to_csv("output/all_data.csv", encoding="utf_8_sig", index=False)
	return data


def trans_data(data_, scaler=None):

	# 时间基础特征
	data_["时间"] = pd.to_datetime(data_["时间"])
	data_["month"] = data_["时间"].dt.month
	data_["week"] = data_["时间"].dt.isocalendar().week
	data_["quarter"] = data_["时间"].dt.quarter
	data_["day"] = data_["时间"].dt.day
	data_["dayofweek"] = data_["时间"].dt.dayofweek
	data_["dayofyear"] = data_["时间"].dt.dayofyear

	# 获取每个商品第一次出现销量的时间
	simple = data_[data_["总销量"] > 0]
	simple["first_day"] = simple.groupby(["商品id"])["dayofyear"].transform("first")
	data_ = data_.merge(simple.loc[:, ["商品id", "first_day"]].drop_duplicates(), on=["商品id"], how="left")
	data_.loc[data_[data_["first_day"].isna()].index, ["first_day"]] = 361
	data_["qty_shichang"] = data_["dayofyear"].astype("int") - data_["first_day"] + 1
	data_["qty_shichang"] = list(map(lambda x: x if x > 0 else 0, data_["qty_shichang"]))
	simple = data_[data_["总销量"] > 1]
	if scaler == 0:
		# 分位数标准化
		simple["Q3"] = simple.groupby(["商品id"])["总销量"].transform(lambda x: np.percentile(x, 75))
		simple["Q1"] = simple.groupby(["商品id"])["总销量"].transform(lambda x: np.percentile(x, 25))
		data_ = data_.merge(simple.loc[:, ["商品id", "Q3", "Q1"]].drop_duplicates(), how="left", on=["商品id"])
		data_["scaler_qty"] = (data_["总销量"] - data_["Q1"]) / (data_["Q3"] - data_["Q1"])
	elif scaler == 1:
		# 去均值标准化
		simple["mean"] = simple.groupby(["商品id"])["总销量"].transform("mean")
		data_ = data_.merge(simple.loc[:, ["商品id", "mean"]].drop_duplicates(), how="left", on=["商品id"])
		data_["scaler_qty"] = data_["总销量"] / data_["mean"]
	elif scaler == 2:
		# log平滑
		data_['scaler_qty'] = list(map(lambda x: math.log(x + 1, 2), data_['总销量']))
	elif scaler == 3:
		data_['scaler_qty'] = list(map(lambda x: math.log(x + 1, 2), data_['总销量']))
		simple = data_[data_["总销量"] > 1]
		# simple["Q3"] = simple.groupby(["商品id"])["总销量"].transform(lambda x: np.percentile(x, 75))
		# simple["Q1"] = simple.groupby(["商品id"])["总销量"].transform(lambda x: np.percentile(x, 25))
		# data_ = data_.merge(simple.loc[:, ["商品id", "Q3", "Q1"]].drop_duplicates(), how="left", on=["商品id"])
		# data_["scaler_qty"] = (data_["总销量"] - data_["Q1"]) / (data_["Q3"] - data_["Q1"])
		simple["mean"] = simple.groupby(["商品id"])["scaler_qty"].transform("mean")
		data_ = data_.merge(simple.loc[:, ["商品id", "mean"]].drop_duplicates(), how="left", on=["商品id"])
		data_["scaler_qty"] = data_["scaler_qty"] / data_["mean"]
		data_.loc[data_[data_["总销量"] == 0].index, ["scaler_qty"]] = 0
		data_.fillna(0, inplace=True)
	elif scaler is None:
		data_['scaler_qty'] = data_['总销量']

	del data_["总销量"]
	del data_["first_day"]
	del data_["时间"]

	# label拼接
	data_["label"] = data_.groupby(["商品id"])["scaler_qty"].shift(-1)
	data_.to_csv("output/trans_data.csv", encoding="utf_8_sig", index=False)
	return data_


def corr_(data):
	ids = data["商品id"].unique()
	df = []
	for id in ids:
		df_ = data[data["商品id"] == id]
		df.append(df_["总销量"].values.tolist())
	df = pd.DataFrame(df)
	df = df.T
	df.columns = ids
	corr_dict = {}
	for id in tqdm(ids):
		corr_df = df.corrwith(df[id])
		corr_list = corr_df[abs(corr_df) > 0.6].index.tolist()
		corr_dict[id] = corr_list
	tf = open("output/corr_dict.json", "w")
	json.dump(corr_dict, tf)
	tf.close()


if __name__ == '__main__':
	if os.path.exists("output/all_data.csv"):
		data = pd.read_csv("output/all_data.csv")
	else:
		data = alldata()
	data = trans_data(data, scaler=3)
	# sample = data[data["总销量"] > 0 ]
	# sample["未来一周天均销量"] = sample.groupby(["商品id"])["总销量"].transform("mean")
	# sub = sample.loc[:, ["商品id", "未来一周天均销量"]].drop_duplicates()
	# data = data.merge(sample, on=["商品id"], how="left")
	# data.fillna(0, inplace=True)
	# data = data.loc[:, ["商品id", "未来一周天均销量"]].drop_duplicates()
	# data.to_csv('output/submit.csv', index=False)
	# corr_(data)
