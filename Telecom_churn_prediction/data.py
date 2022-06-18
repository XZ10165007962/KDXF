# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         data
# Description:
# Author:       xinzhuang
# Date:         2022/6/12
# Function:
# Version：
# Notice:
# -------------------------------------------------------------------------------
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
matplotlib.rcParams['font.sans-serif'] = ['Heiti TC']
matplotlib.rcParams['font.family'] = ['sans-serif']

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)

train_data = pd.read_csv("/Users/xinzhuang/Documents/2022科大讯飞/电信客户流失预测挑战赛数据集/train.csv")
print(train_data.info())
print(train_data.describe())
print(train_data.head())

train_data_0 = train_data[train_data["是否流失"] == 0]
train_data_1 = train_data[train_data["是否流失"] == 1]


for _, i in enumerate(train_data.columns):
    if _ == 0:
        continue
    else:
        plt.figure(figsize=(8, 4), dpi=100)
        sns.countplot(data=train_data, x=i, hue='是否流失', palette="muted")
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.ylabel('Counts', fontsize=12)
        plt.xlabel('Phone Service', fontsize=12)
        plt.title(i)
        plt.show()