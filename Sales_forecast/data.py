# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         data
# Description:
# Author:       xinzhuang
# Date:         2022/6/9
# Function:
# Version：
# Notice:
# -------------------------------------------------------------------------------
import pandas as pd

import config

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)

print("read data ...")
train_dingdan_data = pd.read_csv(config.dingdan_train_path, encoding="gb2312")
train_xuqiu_data = pd.read_csv(config.xuqiu_train_path, encoding="gb2312")

test_dingdan_data = pd.read_csv(config.dingdan_test_path)
test_xuqiu_data = pd.read_csv(config.xuqiu_test_path)

print("transform data")
# 时间类型
train_xuqiu_data["date"] = pd.to_datetime(train_xuqiu_data["date"])
train_xuqiu_data["year_month"] = train_xuqiu_data["date"].dt.strftime("%Y%m")
train_xuqiu_data["year"] = train_xuqiu_data["date"].dt.year
train_xuqiu_data["month"] = train_xuqiu_data["date"].dt.month

test_xuqiu_data["date"] = pd.to_datetime(test_xuqiu_data["date"])
test_xuqiu_data["year"] = test_xuqiu_data["date"].dt.year
test_xuqiu_data["month"] = test_xuqiu_data["date"].dt.month

train_data = train_xuqiu_data.merge(train_dingdan_data, how="left", on=["product_id", "year", "month"])
test_data = test_xuqiu_data.merge(test_dingdan_data, how="left", on=["product_id", "year", "month"])

# 月汇集
train_data["label_month"] = train_data.groupby(["product_id", "year", "month"])["label"].transform("sum")

del train_data["date"]
del train_data["label"]
del test_data["date"]

train_data = train_data.drop_duplicates(["product_id", "year", "month"])
test_data = test_data.drop_duplicates(["product_id", "year", "month"])

# 将离散数据进行编码
lisan_col = ["product_id", "type", "year"]
for col in lisan_col:
	train_data[col] = pd.factorize(train_data[col])[0]

for col in lisan_col:
	test_data[col] = pd.factorize(test_data[col])[0]

print("sucess transform data")
print(train_data.shape)
print(test_data.shape)
del train_data["year"]
del test_data["year"]
# 保存数据
train_data.to_csv(config.save_train_path, index=False)
test_data.to_csv(config.save_test_path, index=False)
print("save data done")