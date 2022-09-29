#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: model.py
@time: 2022/9/29 14:48
@version:
@desc: 
"""
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
from torch.nn import Module, Sequential
import torch.nn as nn

warnings.filterwarnings('ignore')
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 200)


class MLPModel(Module):
	def __init__(self, input_dim):
		super(MLPModel, self).__init__()
		self.net = Sequential(
			nn.Linear(input_dim, 1024),
			nn.ReLU(),
			nn.Linear(1024, 512),
			nn.ReLU(),
			nn.Linear(512, 256),
			nn.ReLU(),
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 1)
		)
		self.critertion = nn.MSELoss(reduction="mean")

	def forward(self, input):
		return self.net(input).squeeze(1)

	def cal_loss(self, pred, target):
		return self.critertion(pred, target)