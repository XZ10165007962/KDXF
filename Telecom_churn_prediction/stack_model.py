# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         stack_model
# Description:
# Author:       xinzhuang
# Date:         2022/6/21
# Function:
# Version：
# Notice:
# -------------------------------------------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, auc, roc_auc_score
from sklearn.svm import SVC

xgb_data = pd.read_csv("xgbtrain.csv", header=None)
lgb_data = pd.read_csv("lgbtrain.csv", header=None)
cat_data = pd.read_csv("cattrain.csv", header=None)
data = pd.concat([xgb_data, lgb_data, cat_data], axis=1)
data.columns = ["xgb", "lgb", "cat"]
y_ = pd.read_csv('data/train.csv')
y = y_["是否流失"]

test_data_ = pd.read_csv("sub_all.csv")
test_data = test_data_.iloc[:, -3:]

x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=2022)
model = LinearRegression()
model.fit(x_train, y_train)
pre = model.predict(x_test)
for i,j in enumerate(pre):
    if j <= 0:
        pre[i] = 0
    elif j >= 1:
        pre[i] = 1
print(pre)
print(mean_squared_error(pre,y_test))
#print(auc(pre,y_test))
print(mean_absolute_error(pre,y_test))

pre_test_data = model.predict(test_data)
print(pre_test_data)
test_data_["是否流失_stack"] = pre_test_data
test_data_[['客户ID', '是否流失_stack']].to_csv('sub_sample_stack.csv', index=False)
