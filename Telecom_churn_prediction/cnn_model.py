#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: cnn_model.py
@time: 2022/6/23 14:59
@version:
@desc: 
"""

import torch
from torch.utils.data import DataLoader,Dataset
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# 超参数
num_epochs = 15
batch_size = 128
learning_rate = 1e-3


class trainData(Dataset):
	def __init__(self, train_path):
		self.train_data = pd.read_csv(train_path)
		# self.train_data = np.loadtxt(train_path)
		self.target = self.train_data["是否流失"].values
		fliter_fea = ['未应答数据呼叫的平均次数','平均呼叫转移呼叫数','平均占线数据调用次数','平均丢弃数据呼叫数',"是否流失"]
		use_fea = [i for i in self.train_data.columns if i not in fliter_fea]
		self.data = self.train_data[use_fea].values

	def __len__(self):
		return len(self.train_data)

	def __getitem__(self, index):
		return self.data[index], self.target[index]


class testData(Dataset):
	def __init__(self, test_path):
		self.test_data = pd.read_csv(test_path)
		fliter_fea = ['未应答数据呼叫的平均次数', '平均呼叫转移呼叫数', '平均占线数据调用次数', '平均丢弃数据呼叫数']
		use_fea = [i for i in self.test_data.columns if i not in fliter_fea]
		self.data = self.test_data[use_fea].values

	def __len__(self):
		return len(self.test_data)

	def __getitem__(self, index):
		return self.data[index]


train_set = trainData('data/train.csv')
test_set = testData('data/test.csv')

train_loader = DataLoader(
	dataset=train_set,
	batch_size=batch_size,
	shuffle=True
)
test_loader = DataLoader(
	dataset=test_set,
	batch_size=batch_size,
	shuffle=False
)


class CnnNet(nn.Module):
	def __init__(self):
		super(CnnNet, self).__init__()
		self.conv1 = nn.Conv1d(in_channels=128, out_channels=10, kernel_size=5, stride=2)
		# self.max_pool1 = nn.MaxPool1d(kernel_size=5)
		self.conv2 = nn.Conv1d(10, 20, 5, 2)
		self.conv3 = nn.Conv1d(20, 40, 3)

		self.liner1 = nn.Linear(440, 120)
		self.liner2 = nn.Linear(120, 84)
		self.liner3 = nn.Linear(84, 128)

	def forward(self, x):
		x = self.conv1(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = F.relu(x)
		x = self.conv3(x)

		x = x.view(-1, 440)
		x = self.liner1(x)
		x = self.liner2(x)
		x = self.liner3(x)

		return x

# 定义损失与优化器
model = CnnNet()
criterion = nn.L1Loss()
optimize = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
	print("*************epcoh_{}*************".format(epoch+1))
	for i, (x, y) in enumerate(train_loader):
		optimize.zero_grad()
		x = torch.tensor(x, dtype=torch.float)
		y = torch.tensor(y, dtype=torch.float).view(-1,128)
		pre = model(x)
		loss = criterion(pre, y)
		loss.backward()
		optimize.step()

		if (i + 1) % 100 == 0:
			print(
				"Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}"
					.format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.item())
			)