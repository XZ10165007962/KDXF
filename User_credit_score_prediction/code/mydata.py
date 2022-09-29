#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: mydata.py
@time: 2022/9/29 10:14
@version:
@desc: 
"""
import pandas as pd
import numpy as np
import warnings

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, random_split
import re

import conf

warnings.filterwarnings('ignore')
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 200)


def get_lisan_cols(data_):
	columns = data_.columns
	lisan_list = []
	for column in columns:
		if "b" in column:
			lisan_list.append(column)
	return lisan_list


class MyDataSet(Dataset):
	def __init__(self, model="train"):
		super(MyDataSet, self).__init__()
		self.model = model
		if model == "train" or model == "dev":
			data = pd.read_csv(conf.train_data_path).fillna(0)
			del data["id"]
			len_data = len(data)
			train_index = [i for i in range(len_data) if i % 10 != 0]
			dev_index = [i for i in range(len_data) if i % 10 == 0]
			if model == "train":
				self.data = torch.FloatTensor(data.iloc[train_index, :-1].values)
				self.target = torch.FloatTensor(data.iloc[train_index, -1].values)
			elif model == "dev":
				self.data = torch.FloatTensor(data.iloc[dev_index, :-1].values)
				self.target = torch.FloatTensor(data.iloc[dev_index, -1].values)
		if model == "test":
			data = pd.read_csv(conf.test_data_path).fillna(0)
			del data["id"]
			self.data = torch.FloatTensor(data.values)
		self.data[:, 40:] = (self.data[:, 40:] - self.data[:, 40:].mean(dim=0)) / self.data[:, 40:].std(dim=0)
		self.dim = self.data.shape[1]

	def __getitem__(self, item):
		if self.model == "test":
			return self.data[item]
		else:
			return self.data[item], self.target[item]

	def __len__(self):
		return len(self.data)


class MyDataLoader():
	def __init__(self, model="train"):
		self.dataset = MyDataSet(model)
		self.model = model

	def get_dataloader(self):
		dataloader = DataLoader(
			self.dataset,
			batch_size=conf.BATCH_SIZE,
			shuffle=(self.model == "train")
		)
		return dataloader


if __name__ == '__main__':
	data = MyDataLoader(model="dev")
	print(data)