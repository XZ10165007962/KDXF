#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: main_week.py
@time: 2022/8/6 20:33
@version:
@desc:
"""
import json
import warnings
import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import copy

import transform_week

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 200)

data = pd.read_csv("output/all_data.csv")
data = data[data["时间"] < "2021/12/28  16:00:00"]
data["时间"] = pd.to_datetime(data["时间"])
data["week"] = data["时间"].dt.isocalendar().week
data["qty_all"] = data.groupby(["商品id"])["总销量"].transform("sum")
data["qty_week"] = data.groupby(["商品id", "week"])["总销量"].transform("mean")
data.loc[data[data["qty_all"] == 0].index, ["qty_week"]] = 0
sub = data[data["week"] == 51]
sub = sub.loc[:, ["商品id", "qty_week"]].drop_duplicates()
sub = sub.rename(columns={'qty_week': f'未来一周天均销量'})
sub.to_csv('output/submit.csv', index=False, encoding="utf_8_sig")
for week in [50, 51]:
	simple = data[data["week"] == week]
	simple = simple.loc[:, ["商品id", "qty_week"]].drop_duplicates()
	simple = simple.rename(columns={'qty_week': f'qty_week_{week}'})
	data = data.merge(simple, on=["商品id"], how="left")
# 过滤没有发生售卖情况的数据
data_1 = data[data["qty_all"] > 0]
# 只取在50周发生售卖情况的商品
simple = data_1[data_1["week"] == 50]
simple = simple[simple["qty_week"] > 0]
col = set(simple["商品id"].values)
data_1 = data_1[data_1["商品id"].isin(col)]
data_1["qty_week_diff"] = data_1["qty_week_51"] / data_1["qty_week_50"]
data_1.to_csv('output/test.csv', index=False, encoding="utf_8_sig")