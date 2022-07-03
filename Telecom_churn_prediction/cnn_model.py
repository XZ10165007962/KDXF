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
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

print("Loading data...")
data_root = "data/nn_data.csv"
train_ = np.loadtxt(data_root, encoding="UTF-8", skiprows=1,delimiter=',')

print("Split data...")
VAL_RATIO = 0.3
percent = int(train_.shape[0] * (1-VAL_RATIO))
train, val = train_[:percent], train_[:percent]
train_x, train_y = train[:, :-1], train[:, -1:]
val_x, val_y = val[:, :-1], val[:, -1:]


class myData(Dataset):
	def __init__(self, x, y=None):
		self.data = torch.from_numpy(x).float()
		if y is not None:
			self.label = torch.from_numpy(y).float()
		else:
			self.label = None

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		if self.label is not None:
			return self.data[index], self.label[index]
		else:
			return self.data[index]


train_set = myData(train_x, train_y)
val_set = myData(val_x, val_y)

# 超参数
num_epochs = 15
batch_size = 64
learning_rate = 1e-3
channels = train_x.shape[1]

train_loader = DataLoader(
	dataset=train_set,
	batch_size=batch_size,
	shuffle=True
)
test_loader = DataLoader(
	dataset=val_set,
	batch_size=batch_size,
	shuffle=False
)


class CnnNet(nn.Module):
	def __init__(self):
		super(CnnNet, self).__init__()
		self.conv1 = nn.Conv1d(in_channels=1, out_channels=10, kernel_size=4, stride=2)
		# self.max_pool1 = nn.MaxPool1d(kernel_size=5)
		self.conv2 = nn.Conv1d(10, 20, 5, 2)
		self.conv3 = nn.Conv1d(20, 40, 3)
		self.flatten = nn.Flatten()
		self.liner1 = nn.Linear(480, 120)
		self.liner2 = nn.Linear(120, 84)
		self.liner3 = nn.Linear(84, 1)

	def forward(self, x):
		x = self.conv1(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = F.relu(x)
		x = self.conv3(x)

		# x = x.view(x.shape[0], -1)
		x = self.flatten(x)
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
		x = x.view(x.shape[0], -1, 64)
		pre = model(x)
		loss = criterion(pre, y)
		loss.backward()
		optimize.step()

		if (i + 1) % 100 == 0:
			print(
				"Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}"
				.format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.item())
			)