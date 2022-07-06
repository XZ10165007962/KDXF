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
test_xuqiu_data["year_month"] = test_xuqiu_data["date"].dt.strftime("%Y%m")
test_xuqiu_data["year"] = test_xuqiu_data["date"].dt.year
test_xuqiu_data["month"] = test_xuqiu_data["date"].dt.month

train_data = train_xuqiu_data.merge(train_dingdan_data, how="left", on=["product_id", "year", "month"])
test_data = test_xuqiu_data.merge(test_dingdan_data, how="left", on=["product_id", "year", "month"])

data = pd.concat([train_data, test_data]).reset_index(drop=True)
data = data.sort_values(['product_id', 'year_month'])

# 获取数据最早的售卖月份
simple = data[data["label"] > 0]
simple["qty_first_month"] = simple.groupby(["product_id"])["year_month"].transform("first")
simple = simple.loc[:, ["product_id", "qty_first_month"]].drop_duplicates()
data = data.merge(simple, on=["product_id"], how="left")
# 过滤还未售卖的时间
data = data[data["qty_first_month"] <= data["year_month"]].reset_index(drop=True)

# 数据分类,想通过变异系数进行稳定数据与不稳定数据划分
simple = data[~data["label"].isnull()]
simple["label_sum"] = simple.groupby(["product_id"])["label"].transform("sum")
simple["label_mean"] = simple.groupby(["product_id"])["label"].transform("mean")
simple["label_std"] = simple.groupby(["product_id"])["label"].transform("std")
simple["label_cv"] = simple["label_std"] / simple["label_mean"]
simple["sale_count"] = simple.groupby(["product_id", "year_month"])["is_sale_day"].transform("sum")
simple = simple.loc[:, ["product_id", "label_sum", "label_mean", "label_std", "label_cv", "sale_count"]].drop_duplicates()
data = data.merge(simple, on=["product_id"], how="left")

"""
在训练集中未发生售卖事件的商品
1117 1131 1140 1141 1142 1143 1145 1146 1147 1148 1149 1150 1151 1152
1153 1154 1155 1156 1157 1158 1159 1160 1161 1162
"""
# simple = simple.loc[:, ["product_id", "label_cv"]].drop_duplicates()
# print(simple)

# 月汇集
data["label_month"] = data.groupby(["product_id", "year", "month"])["label"].transform("sum")
data = data.drop_duplicates(["product_id", "year", "month"]).reset_index(drop=True)

# 获取销售的月份数量，进行新品主品数据的判断
simple = data[~data["label"].isnull()]
simple["qty_month_count"] = simple.groupby(["product_id"])["year_month"].transform("count")
simple = simple.loc[:, ["product_id", "qty_month_count"]].drop_duplicates()
data = data.merge(simple, on=["product_id"], how="left")

print("sucess transform data")
# del data["date"]
del data["label"]
del data["is_sale_day"]
del data["year_month"]
del data["qty_first_month"]
data["date_block_num"] = (data["year"] - 2018) * 12 + data["month"]
# 保存数据
data.to_csv(config.save_data_path, index=False)
print("save data done")