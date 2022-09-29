#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: main.py
@time: 2022/9/29 14:53
@version:
@desc: 
"""
import pandas as pd
import numpy as np
import warnings
import torch
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error

import conf
from mydata import MyDataLoader
import model

warnings.filterwarnings('ignore')
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 200)


def get_device():
	return "cuda" if torch.cuda.is_available() else "cpu"


def train(model, train_data, dev_data):
	max_epoch = conf.EPOCH
	epoch = 1
	lr = conf.LR
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	train_loss = []
	dev_loss = []
	break_flag = 0
	min_mse = 1000
	while epoch < max_epoch:
		model.train()
		loss_num = 0
		for input, label in train_data:
			input = input.to(device)
			label = label.to(device)
			optimizer.zero_grad()
			output = model(input)
			loss = model.cal_loss(output, label)
			loss_num += loss.detach()
			train_loss.append(loss.detach())
			loss.backward()
			optimizer.step()
		if epoch % 10 == 0:
			print('[{:03d}/{:03d}] Train loss: {:3.6f}'.format(
				epoch, max_epoch, loss_num / len(input)
			))
			dev_mse = dev(model, dev_data)
			if dev_mse < min_mse:
				min_mse = dev_mse
				print('Save model epoch = {:4d}, loss = {:.4f}'.format(epoch + 1, min_mse))
				torch.save(model.state_dict(), conf.model_paht+f"mymodel_{epoch}.pth")
				break_flag = 0
			else:
				break_flag += 1
			dev_loss.append(dev_mse.detach())
			if break_flag > 200:
				break
		epoch += 1
	torch.save(model.state_dict(), conf.model_paht+"mymodel.pth")

	return train_loss, dev_loss


def dev(model, dev_data):
	model.eval()
	total_loss = []
	for input, label in dev_data:
		input = input.to(device)
		label = label.to(device)
		output = model(input)
		total_loss.append(model.cal_loss(output, label))
	return sum(total_loss) / len(total_loss)

def test(model, test_data):
	model.eval()
	output = []
	for input in test_data:
		input = input.to(device)
		pred = model(input)
		output.append(pred.cpu().detach())
	output = torch.cat(output, dim=0).numpy()
	return output


if __name__ == '__main__':
	device = get_device()
	train_data = MyDataLoader(model="train").get_dataloader()
	dev_data = MyDataLoader(model="dev").get_dataloader()
	test_data = MyDataLoader(model="test").get_dataloader()
	torch.manual_seed(conf.SEED)
	mymodel = model.MLPModel(243).to(device)
	train_loss, dev_loss = train(mymodel, train_data, dev_data)
	# model = torch.load(conf.model_paht+"mymodel_40.pth", map_location="cuda")
	# mymodel.load_state_dict(model)
	# pres = test(mymodel, test_data)
	# res = pd.read_csv(conf.sample_paht)
	# res["target"] = pres
	# res.to_csv(conf.predict_data_path+"sub.csv", index=False)