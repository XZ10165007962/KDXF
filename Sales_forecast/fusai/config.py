#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: config.py
@time: 2022/8/11 10:43
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

simple_data_path = "E:/KDXF/data/提交示例.csv"

xuqiu_train_path = "E:/KDXF/Sales_forecast/data/商品销量智能预测复赛公开数据/商品需求训练集.csv"
dingdan_train_path = "E:/KDXF/Sales_forecast/data/商品销量智能预测复赛公开数据/商品月订单训练集.csv"

xuqiu_test_path = "E:/KDXF/Sales_forecast/data/商品销量智能预测复赛公开数据/商品需求测试集.csv"
dingdan_test_path = "E:/KDXF/Sales_forecast/data/商品销量智能预测复赛公开数据/商品月订单测试集.csv"

save_data_path = "E:/KDXF/Sales_forecast/output/复赛/data.csv"
save_trans_data_path = "E:/KDXF/Sales_forecast/output/复赛/trans_data.csv"

save_test_pre_path = "E:/KDXF/Sales_forecast/output/复赛/test_pre.csv"
save_train_pre_path = "E:/KDXF/Sales_forecast/output/复赛/train_pre.csv"

tijiao_path = "E:/KDXF/Sales_forecast/output/复赛/tijiao.csv"

if_del = False