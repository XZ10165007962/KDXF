#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: sub_data.py
@time: 2022/9/8 10:30
@version:
@desc: 
"""
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 200)


def get_test():
	data1 = pd.read_csv("../user_data/tmp_data/lgb_mae_seed2022.csv")
	data2 = pd.read_csv("../user_data/tmp_data/linear.csv")

	data1 = data1.rename(columns={"pm": "lgb_mae_seed2022_pm"})
	data2 = data2.rename(columns={"pm": "linear_pm"})

	data = data1.merge(data2, on=["session_id", "rank"], how="left")

	id = [18,19,32,33,42,45,71,79,91,95,105,117,124,147,156,160,180,183,203,204,213,241,242,251,257,279,88,87,75,182,160]
	data1 = data[data["session_id"].isin(id)]
	data2 = data[~data["session_id"].isin(id)]
	data1["pm"] = data1["linear_pm"]
	data2["pm"] = data2["lgb_mae_seed2022_pm"]

	sub = pd.concat([data1.loc[:, ["session_id", "rank", "pm"]], data2.loc[:, ["session_id", "rank", "pm"]]], axis=0)
	sub.to_csv("../prediction_result/sub.csv", index=False)


if __name__ == '__main__':
	get_test()