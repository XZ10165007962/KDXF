#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: tttt.py
@time: 2022/8/7 16:36
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
from statsmodels.tsa.holtwinters import Holt, SimpleExpSmoothing
from tqdm import tqdm

import transform_week
warnings.filterwarnings('ignore')
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 200)

data = pd.read_csv("output/all_data_week.csv")
data = data[data["时间"] > "2020/7/11  16:00:00"]
data["时间"] = pd.to_datetime(data["时间"])
data["week"] = data["时间"].dt.isocalendar().week
data = data[data["week"] < 52]
data["qty_week"] = data.groupby(["商品id", "week"])["总销量"].transform("mean")
data = data.drop_duplicates(["商品id", "week"])
data = data.loc[:, ["商品id", "qty_week"]]
data["pre"] = 0
ids = set(data["商品id"].values)

for id in tqdm(ids):
	data_ = data[data["商品id"] == id]
	model = SimpleExpSmoothing(data_["qty_week"], initialization_method="estimated").fit()
	pre = model.forecast(1)
	data.loc[data_.index, ["pre"]] = [pre] * len(data_)
data['pre'] = list(map(lambda x: 0 if x <= 0 else x, data['pre']))
data = data.loc[:, ["商品id", "pre"]].drop_duplicates()

data = data.rename(columns={'pre':'未来一周天均销量'})
submit = pd.read_csv('data/电商销量预测挑战赛公开数据/提交示例.csv')
submit.drop(['未来一周天均销量'],axis=1,inplace=True)
submit = pd.merge(submit,data,on='商品id')
submit.to_csv('output/test_1.csv', index=False, encoding="utf_8_sig")



# data["qty_all"] = data.groupby(["商品id"])["总销量"].transform("sum")
# data["qty_week"] = data.groupby(["商品id", "week"])["总销量"].transform("mean")
# data.loc[data[data["qty_all"] == 0].index, ["qty_week"]] = 0
# sub = data[data["week"] == 51]
# sub = sub.loc[:, ["商品id", "qty_week"]].drop_duplicates()
# sub = sub.rename(columns={'qty_week': f'未来一周天均销量'})
# sub.to_csv('output/submit.csv', index=False, encoding="utf_8_sig")
# for week in [50, 51]:
# 	simple = data[data["week"] == week]
# 	simple = simple.loc[:, ["商品id", "qty_week"]].drop_duplicates()
# 	simple = simple.rename(columns={'qty_week': f'qty_week_{week}'})
# 	data = data.merge(simple, on=["商品id"], how="left")
# # 过滤没有发生售卖情况的数据
# data_1 = data[data["qty_all"] > 0]
# # 只取在50周发生售卖情况的商品
# simple = data_1[data_1["week"] == 50]
# simple = simple[simple["qty_week"] > 0]
# col = set(simple["商品id"].values)
# data_1 = data_1[data_1["商品id"].isin(col)]
# data_1["qty_week_diff"] = data_1["qty_week_51"] / data_1["qty_week_50"]
# data_1.to_csv('output/test.csv', index=False, encoding="utf_8_sig")
