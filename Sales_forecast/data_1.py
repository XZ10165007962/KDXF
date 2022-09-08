#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: data_1.py
@time: 2022/7/17 13:08
@version:
@desc: 
"""
import pandas as pd
from scipy import stats

import config

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)


def read_data():
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
	# train_xuqiu_data["quarter"] = train_xuqiu_data["date"].dt.quarter

	test_xuqiu_data["date"] = pd.to_datetime(test_xuqiu_data["date"])
	test_xuqiu_data["year_month"] = test_xuqiu_data["date"].dt.strftime("%Y%m")
	test_xuqiu_data["year"] = test_xuqiu_data["date"].dt.year
	test_xuqiu_data["month"] = test_xuqiu_data["date"].dt.month
	# test_xuqiu_data["quarter"] = test_xuqiu_data["date"].dt.quarter

	train_data = train_xuqiu_data.merge(train_dingdan_data, how="left", on=["product_id", "year", "month"])
	test_data = test_xuqiu_data.merge(test_dingdan_data, how="left", on=["product_id", "year", "month"])

	data = pd.concat([train_data, test_data]).reset_index(drop=True)
	data = data.sort_values(['product_id', 'year_month'])

	simple = data[~data["label"].isnull()]
	simple["sale_count"] = simple.groupby(["product_id", "year_month"])["is_sale_day"].transform("sum")
	simple = simple.loc[:, ["product_id", "year_month", "sale_count"]].drop_duplicates(["product_id", "year_month"])
	data = data.merge(simple, on=["product_id", "year_month"], how="left")

	# 月汇集
	data["label_month"] = data.groupby(["product_id", "year", "month"])["label"].transform("sum")
	data["sale_count"] = data.groupby(["product_id", "year_month"])["is_sale_day"].transform("sum")
	data = data.drop_duplicates(["product_id", "year", "month"]).reset_index(drop=True)

	# 统计售卖月份数
	# 获取第一次售卖的月份
	simple = data[data["label_month"] > 0]
	simple["qty_first_month"] = simple.groupby(["product_id"])["year_month"].transform("first")
	simple = simple.loc[:, ["product_id", "qty_first_month"]].drop_duplicates()
	data = data.merge(simple, on=["product_id"], how="left")
	# 过滤还未售卖的时间
	data_first = data[data["qty_first_month"] <= data["year_month"]].reset_index(drop=True)

	simple = data_first[~data_first["label"].isnull()]
	simple["qty_month_count"] = simple.groupby(["product_id"])["year_month"].transform("count")
	simple = simple.loc[:, ["product_id", "qty_month_count"]].drop_duplicates()
	data = data.merge(simple, on=["product_id"], how="left")

	# 数据标准化
	data["all_qty"] = data.groupby(["product_id"])["label_month"].transform("sum")
	data["mean_qty"] = data["all_qty"] / data["qty_month_count"]
	data["scan_qty"] = data["label_month"] / data["mean_qty"]
	data["date_block_num"] = (data["year"] - 2018) * 12 + data["month"]
	data["scan_qty"].fillna(0, inplace=True)

	simple = data[data["year_month"] == data["qty_first_month"]]
	simple["qty_first"] = simple["date_block_num"]
	simple = simple.loc[:, ["product_id", "qty_first"]]
	data = data.merge(simple, on=["product_id"], how="left")
	data["qty_num"] = data["date_block_num"] - data["qty_first"] + 1
	data["qty_num"] = data["qty_num"].map(lambda x: x if x > 0 else 0)

	del data["qty_first"]
	del data["label"]
	del data["is_sale_day"]
	del data["year_month"]
	del data["all_qty"]
	del data["qty_first_month"]
	del data["qty_month_count"]
	# del data["qty_num"]
	return data


if __name__ == '__main__':
	data = read_data()
	data.to_csv("output/初赛/data_1.csv", index=False)