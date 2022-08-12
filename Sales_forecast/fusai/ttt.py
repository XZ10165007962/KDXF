#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: ttt.py
@time: 2022/8/11 12:55
@version:
@desc: 
"""
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 200)
# sub = pd.read_csv("E:/KDXF/Sales_forecast/output/复赛/data.csv")
# sub = sub.query("year==2021 & month>=4")
# sub = sub.drop_duplicates(['month', 'product_id'])
# sub = sub.loc[:, ["date", 'product_id']]
# data = pd.read_csv('E:/KDXF/Sales_forecast/output/复赛/baseline0811.csv')
# data = data.rename(columns={'month': 'date'})
# sub = sub.merge(data, on=['date', 'product_id'], how="left")
# sub = sub.rename(columns={'date': 'month'})
# sub["month"] = pd.to_datetime(sub["month"])
# sub.to_csv('E:/KDXF/Sales_forecast/output/复赛/baseline0811.csv', index=False)
da = pd.read_csv('E:/KDXF/Sales_forecast/output/初赛/baseline68791.csv')
print(da.head())