#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: baseline.py
@time: 2022/7/26 23:10
@version:
@desc: 
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
import math
import glob
import datetime
from tqdm import tqdm
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns',None)


# 数据预处理
def prepare(df):
	# 标签平滑
	df['salesVolume'] = list(map(lambda x: math.log(x + 1, 2), df['总销量']))

	df['date'] = pd.to_datetime(df['时间'])

	data = df[['商品id', 'date', 'salesVolume']]
	data['time_index'] = np.arange(1, data.shape[0] + 1)

	# 增加7行
	d_7 = pd.DataFrame({'商品id': [data['商品id'][0]] * 7,
						'date': [data['date'][data.shape[0] - 1] + datetime.timedelta(days=x) for x in range(1, 8)],
						'salesVolume': np.zeros(7),
						'time_index': np.arange(174, 181)})
	data = pd.concat([data, d_7], axis=0)

	return data


def get_feature(df, day_idx, rec):
	data = df.copy()

	# 倒数n日的销量
	for n in range(1, 15):
		if rec == 1:  # 递归预测
			data[f'last_{n}_sale'] = data['salesVolume'].shift(n)
		else:  # 直接预测
			data[f'last_{n}_sale'] = data['salesVolume'].shift(n + (day_idx - 173 - 1))

	# 强相关特征
	# 2日均值，3日均值，7日均值，14日均值，30日均值
	data['1_2_mean'] = data.loc[:, 'last_1_sale':'last_2_sale'].mean(1)
	data['1_3_mean'] = data.loc[:, 'last_1_sale':'last_3_sale'].mean(1)
	data['1_7_mean'] = data.loc[:, 'last_1_sale':'last_7_sale'].mean(1)
	data['1_14_mean'] = data.loc[:, 'last_1_sale':'last_14_sale'].mean(1)
	data['4_6_mean'] = data.loc[:, 'last_4_sale':'last_6_sale'].mean(1)
	data['8_14_mean'] = data.loc[:, 'last_8_sale':'last_14_sale'].mean(1)
	# 其他特征，自行编写

	# 趋势特征
	data['1_2_diff'] = data['last_1_sale'] - data['last_2_sale']
	data['1_3_diff'] = data['last_1_sale'] - data['last_3_sale']
	data['1_4_diff'] = data['last_1_sale'] - data['last_4_sale']
	data['1_8_diff'] = data['last_1_sale'] - data['last_8_sale']

	# 前1天至前3天的均值相对于前4天至前6天的均值
	data['1_3_to_4_6_diff'] = data['1_3_mean'] - data['4_6_mean']
	data['1_7_to_8_14_diff'] = data['1_7_mean'] - data['8_14_mean']

	# 周期特征
	# 每月的日期、星期
	data['day'] = df['date'].apply(lambda x: x.day)
	data['weekday'] = df['date'].apply(lambda x: x.weekday())
	# data['month'] = df['date'].apply(lambda x : x.month)#月

	# 节假日、重要的促销活动日（比如双十一、618）
	data['holiday'] = np.zeros(data.shape[0])
	# 双11
	data['holiday'][126] = 1

	return data


def get_model():
	model = lgb.LGBMRegressor(
		objective='mse',
		n_estimators=2000,
		learning_rate=0.05,
		max_depth=2,
		num_leaves=2 ** 2 - 1,
		subsample=0.8,
		colsample_bytree=0.8,
		# min_child_samples=1,
		# reg_alpha=0.25,
		# reg_lambda=0.25,
		random_state=2022,
		verbose=1,
	)
	return model


def get_train_model(df_, m, features, cat_feat, rec):
	df = df_.copy()
	# 数据集划分
	if rec == 1:  # 递归预测
		train_idx = df['time_index'].between(1, m - 1)
		test_idx = df['time_index'].between(m, m)
	else:  # 直接预测
		train_idx = df['time_index'].between(1, 173)
		test_idx = df['time_index'].between(m, m)

	model = get_model()
	model.fit(df[train_idx][features], df[train_idx]['salesVolume'], categorical_feature=cat_feat)

	# 预测（这里同时预测了训练集和测试集）
	df['forecastVolume'] = model.predict(df[features])

	# 特征重要性
	# df_importance = pd.DataFrame({'column': features, 'importance': model.feature_importances_})
	# df_importance = df_importance.sort_values(by='importance', ascending=False).reset_index(drop=True)
	# df_importance.to_csv('/importance.csv', index=False)

	# 按测试集id提取数据
	sub = df[test_idx][['商品id']]
	sub['forecastVolume'] = df[test_idx]['forecastVolume']

	return sub


data_path = glob.glob('data/电商销量预测挑战赛公开数据/数据集/*')

# rec为1则递归预测，rec为0则直接预测
rec = 1

train_data = pd.DataFrame()

# 存储每轮预测的预测值
sub = pd.DataFrame()
# 存储7天预测值
sub_7 = pd.DataFrame()

for day_index in [174, 175, 176, 177, 178, 179, 180]:

	# 按商品循环计算特征并合并所有数据
	print("合并数据。。。。")
	for dp in tqdm(range(0, len(data_path))):
		if dp == 0:
			train_data = pd.read_csv(data_path[0])
			train_data = prepare(train_data)

			# 把新的预测值作为历史数据
			if day_index >= 175:
				train_data['salesVolume'][day_index - 1] = sub['forecastVolume'][0]

			train_data = get_feature(train_data, day_index, rec)
		else:
			dt = pd.read_csv(data_path[dp])
			dt = prepare(dt)

			if day_index >= 175:
				dt['salesVolume'][day_index - 1] = sub['forecastVolume'][dp]

			dt = get_feature(dt, day_index, rec)

			train_data = pd.concat([train_data, dt], axis=0)

	cat_feat = ['商品id']
	for i in cat_feat:
		train_data[i] = train_data[i].astype('category')

	features = [col for col in train_data.columns if col not in ['date', 'time_id', 'salesVolume']]
	print("模型训练.....")
	sub = get_train_model(train_data, day_index, features, cat_feat, rec)
	sub.reset_index(drop=True, inplace=True)

	if day_index == 174:
		sub_7 = sub
	else:
		sub_7 = pd.concat([sub_7, sub], axis=0)

	print('day_index:', day_index)

sub_7['forecastVolume'] = list(map(lambda x: 0 if x <= 0 else (2 ** (x)) - 1, sub_7['forecastVolume']))
sub_m = sub_7.groupby(['商品id']).agg({'forecastVolume':'mean'}).reset_index().rename(columns={'forecastVolume':'未来一周天均销量'})
submit = pd.read_csv('data/电商销量预测挑战赛公开数据/提交示例.csv')
submit.drop(['未来一周天均销量'],axis=1,inplace=True)
submit = pd.merge(submit,sub_m,on='商品id')
submit.to_csv('output/submit.csv', index=False)