#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: conf.py
@time: 2022/9/29 10:16
@version:
@desc: 
"""
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
import torch

warnings.filterwarnings('ignore')
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 200)

# 文件路径
train_data_path = "../xfdata/金融场景下用户信用分预测挑战赛公开数据/data_train.csv"
test_data_path = "../xfdata/金融场景下用户信用分预测挑战赛公开数据/data_test.csv"
predict_data_path = "../prediction_result/"
tmp_data_paht = "../user_data/tmp_data/"

# 超参数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EPOCH = 1000
LR = 0.0001
SEED = 2022