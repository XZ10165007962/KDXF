#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: transform_week.py
@time: 2022/8/5 17:46
@version:
@desc: 
"""
import pandas as pd
import numpy as np
from tqdm import tqdm


def lag_feat(data, columns, lags=30, rollings=7, shift=0):
	print("lag_feat函数")
	for column in columns:
		print(f"{column}滞后特征生成")
		for lag in tqdm(range(1, lags+1)):
			data[f"{column}_lag_{lag}"] = data.groupby(["商品id"])[column].shift(lag + shift)

		# 强相关特征
		# 2日均值，3日均值，7日均值，14日均值，30日均值
		funcs = ["mean", "std", "sum", "min", "max", "median"]
		for fucn in funcs:
			print(fucn)
			data[f"{column}_1_4_{fucn}"] = data.loc[:, f"{column}_lag_1":f"{column}_lag_4"].agg(fucn, axis=1)
			data[f"{column}_1_2_{fucn}"] = data.loc[:, f"{column}_lag_1":f"{column}_lag_2"].agg(fucn, axis=1)
			data[f"{column}_1_3_{fucn}"] = data.loc[:, f"{column}_lag_1":f"{column}_lag_3"].agg(fucn, axis=1)
			data[f"{column}_1_4_{fucn}"] = data.loc[:, f"{column}_lag_1":f"{column}_lag_4"].agg(fucn, axis=1)
			data[f"{column}_2_3_{fucn}"] = data.loc[:, f"{column}_lag_2":f"{column}_lag_3"].agg(fucn, axis=1)
			data[f"{column}_2_4_{fucn}"] = data.loc[:, f"{column}_lag_2":f"{column}_lag_4"].agg(fucn, axis=1)
			data[f"{column}_3_4_{fucn}"] = data.loc[:, f"{column}_lag_3":f"{column}_lag_4"].agg(fucn, axis=1)

			if fucn == "mean":
				# 趋势特征
				data[f"{column}_1_2_diff"] = data[f"{column}_lag_1"] - data[f"{column}_lag_2"]
				data[f"{column}_1_3_diff"] = data[f"{column}_lag_1"] - data[f"{column}_lag_3"]
				data[f"{column}_1_4_diff"] = data[f"{column}_lag_1"] - data[f"{column}_lag_4"]

				data[f"{column}_1_2_2_3diff"] = data[f"{column}_1_2_{fucn}"] - data[f"{column}_2_3_{fucn}"]
				data[f"{column}_2_3_3_4diff"] = data[f"{column}_2_3_{fucn}"] - data[f"{column}_3_4_{fucn}"]
	for column in ["直播销量_week", "视频销量_week", "直播个数_week", "视频个数_week", "浏览量_week", "视频达人_week", "直播达人_week"]:
		for lag in tqdm(range(1, 2)):
			data[f"{column}_lag_{lag}"] = data.groupby(["商品id"])[column].shift(lag + shift)

	return data


if __name__ == '__main__':
	data = pd.read_csv("output/trans_data_week.csv")
	columns = ["scaler_qty"]
	data = lag_feat(data, columns, lags=4, shift=1)
	data.to_csv("output/feat_data_week.csv", encoding="utf_8_sig", index=False)
