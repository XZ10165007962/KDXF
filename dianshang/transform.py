#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: transform.py
@time: 2022/8/1 14:47
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
			data[f"{column}_4_7_{fucn}"] = data.loc[:, f"{column}_lag_4":f"{column}_lag_7"].agg(fucn, axis=1)
			data[f"{column}_1_7_{fucn}"] = data.loc[:, f"{column}_lag_1":f"{column}_lag_7"].agg(fucn, axis=1)
			data[f"{column}_8_14_{fucn}"] = data.loc[:, f"{column}_lag_8":f"{column}_lag_14"].agg(fucn, axis=1)
			# data[f"{column}_15_21_{fucn}"] = data.loc[:, f"{column}_lag_15":f"{column}_lag_21"].agg(fucn, axis=1)
			# data[f"{column}_22_28_{fucn}"] = data.loc[:, f"{column}_lag_22":f"{column}_lag_28"].agg(fucn, axis=1)
			data[f"{column}_1_14_{fucn}"] = data.loc[:, f"{column}_lag_1":f"{column}_lag_14"].agg(fucn, axis=1)
			# data[f"{column}_1_21_{fucn}"] = data.loc[:, f"{column}_lag_1":f"{column}_lag_21"].agg(fucn, axis=1)
			# data[f"{column}_1_28_{fucn}"] = data.loc[:, f"{column}_lag_1":f"{column}_lag_28"].agg(fucn, axis=1)

			if fucn == "mean":
				# 趋势特征
				data[f"{column}_1_2_diff"] = data[f"{column}_lag_1"] - data[f"{column}_lag_2"]
				data[f"{column}_1_3_diff"] = data[f"{column}_lag_1"] - data[f"{column}_lag_3"]
				data[f"{column}_1_8_diff"] = data[f"{column}_lag_1"] - data[f"{column}_lag_8"]
				data[f"{column}_1_15_diff"] = data[f"{column}_lag_1"] - data[f"{column}_lag_15"]
				# 比值特征
				# data[f"{column}_1_2_1_3"] = data[f"{column}_1_2_diff"] / data[f"{column}_1_3_diff"]
				# data[f"{column}_1_2_1_8"] = data[f"{column}_1_2_diff"] / data[f"{column}_1_8_diff"]
				#
				# data[f"{column}_1_4_4_7_{fucn}"] = data[f"{column}_1_4_{fucn}"] / data[f"{column}_4_7_{fucn}"]
				# data[f"{column}_1_7_8_14_{fucn}"] = data[f"{column}_1_7_{fucn}"] / data[f"{column}_8_14_{fucn}"]
	for column in ["直播销量", "视频销量", "直播个数", "视频个数", "浏览量", "抖音转化率", "视频达人", "直播达人"]:
		for lag in tqdm(range(1, 2)):
			data[f"{column}_lag_{lag}"] = data.groupby(["商品id"])[column].shift(lag + shift)

	return data


if __name__ == '__main__':
	data = pd.read_csv("output/trans_data.csv")
	columns = ["scaler_qty"]
	data = lag_feat(data, columns, lags=30, rollings=7)
	data.to_csv("output/feat_data.csv", encoding="utf_8_sig", index=False)
