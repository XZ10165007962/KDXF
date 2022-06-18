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

train_data = pd.read_csv("/Users/xinzhuang/Documents/2022科大讯飞/中国信息协会区块链分会-糖尿病遗传风险预测挑战赛公开数据/train.csv",
                   encoding="gb2312")
print(train_data.info())

test_data = pd.read_csv("/Users/xinzhuang/Documents/2022科大讯飞/中国信息协会区块链分会-糖尿病遗传风险预测挑战赛公开数据/test.csv",
                   encoding="gb2312")
print(test_data.info())