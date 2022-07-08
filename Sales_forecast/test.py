#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: test.py
@time: 2022/7/5 23:32
@version:
@desc: 
"""
import numpy as np
import pandas as pd
import config

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)

def func():
	# 数据后处理
	pre_data = pd.read_csv(config.save_test_pre_path)
	simple_data = pd.read_csv(config.simple_data_path)
	pre_data["month"] = list(
		map(lambda x, y: str(x)+"-"+str(y), pre_data["year"],pre_data["month"] )
	)
	print(simple_data.head())
	del simple_data["label"]
	# 数据合并
	print(pre_data.loc[:, ['product_id', 'month', 'label_month']].tail())
	simple_data = simple_data.merge(pre_data.loc[:, ['product_id', 'month', 'label_month']], on=['product_id', 'month'], how="left")

	# 将在训练集中销量为0的商品销量继续预测为0
	zero_id = [
			1117, 1131, 1140, 1141, 1142, 1143, 1145, 1146, 1147, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156,
			1157, 1158, 1159, 1160, 1161, 1162
		]
	simple_data.loc[simple_data["product_id"].isin(zero_id), ["label_month"]] = 0
	# simple_data.to_csv(config.tijiao_path, index=False)
	# 将销售日期较少的数据用滑动平均进行预测
	small_data = pd.read_csv(config.save_data_path)
	small_data = small_data[small_data["qty_month_count"] < 6]
	small_data.to_csv("output/small.csv", index=False)


if __name__ == '__main__':
	data = pd.read_csv("output/提交示例.csv")
	print(data.head())

	data1 = pd.read_csv("output/tijiao.csv")
	print(data1.head())

	data1["month"] = data["month"]
	print(data1.head())
	data1.to_csv("output/tijiao_1.csv", index=False)