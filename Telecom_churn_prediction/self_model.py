from sklearn import metrics
import pandas as pd
import os,gc,math,time,warnings
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler
from gensim.models import Word2Vec
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


# 构建模型
def cv_model(clf,train_x,train_y,test_x,clf_name):
    folds = 5
    seed = 2022
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    train = np.zeros(train_x.shape[0])
    test = np.zeros(test_x.shape[0])

    cv_scores = []
    for i, (train_index,valid_index) in enumerate(kf.split(train_x,train_y)):
        print('**************** {} *****************'.format(str(i+1)))
        trn_x,trn_y,val_x,val_y = train_x.iloc[train_index],train_y[train_index],train_x.iloc[valid_index],train_y[valid_index]

        if clf_name == 'lgb':
            train_matrix = clf.Dataset(trn_x,label=trn_y)
            valid_matrix = clf.Dataset(val_x,label=val_y)

            params = {
                'boosting_type':'gbdt',
                'objective':'binary', # 二分类问题
                'metric':'auc', # auc评价指标
                'min_child_weight':3,
                'num_leaves':2**5,
                'lambda_l2':10,
                'feature_fraction':0.75,
                'bagging_fraction':0.75,
                'bagging_freq':10,
                'learning_rate':0.15,
                'seed':2022,
                # 'max_depth': 10,
                'verbose': -1,
                'n_jobs':-1
            }

            model = clf.train(params,train_set = train_matrix,num_boost_round = 50000,valid_sets = [train_matrix,valid_matrix],
            categorical_feature = [], verbose_eval = 3000, early_stopping_rounds = 200)
            val_pred = model.predict(data=val_x,num_iteration=model.best_iteration)
            test_pred = model.predict(data=test_x,num_iteration=model.best_iteration)
            print(list(
                sorted(
                    zip(features,model.feature_importance('gain')),
                key=lambda x: x[1], reverse=True)
            ))

        elif clf_name == 'xgb':
            train_matrix = clf.DMatrix(trn_x, label=trn_y)
            valid_matrix = clf.DMatrix(val_x, label=val_y)
            test_matrix = clf.DMatrix(test_x)

            params = {'booster': 'gbtree',
                      'objective': 'binary:logistic',
                      'eval_metric': 'auc',
                      'gamma': 1,
                      'min_child_weight': 1.5,
                      'max_depth': 10,
                      'lambda': 9,
                      'subsample': 0.8,
                      'colsample_bytree': 0.8,
                      'eta': 0.1,
                      'tree_method': 'exact',
                      'seed': 2022,
                      'nthread': 36,
                      "silent": True,
                      }

            watchlist = [(train_matrix, 'train'), (valid_matrix, 'eval')]

            model = clf.train(params, train_matrix, num_boost_round=50000, evals=watchlist, verbose_eval=3000,
                              early_stopping_rounds=200)
            val_pred = model.predict(valid_matrix, ntree_limit=model.best_ntree_limit)
            test_pred = model.predict(test_matrix, ntree_limit=model.best_ntree_limit)

        elif clf_name == 'cat':
            params = {'learning_rate': 0.15, 'depth': 7, 'l2_leaf_reg': 9, 'bootstrap_type': 'Bernoulli',
                      'od_type': 'Iter', 'od_wait': 50, 'random_seed': 2022, 'allow_writing_files': False,
                      'eval_metric': "AUC"}

            model = clf(iterations=20000, **params)
            model.fit(trn_x, trn_y, eval_set=(val_x, val_y),
                      cat_features=[], use_best_model=True, verbose=3000)

            val_pred = model.predict(val_x)
            test_pred = model.predict(test_x)

        train[valid_index] = val_pred
        test += test_pred / kf.n_splits
        cv_scores.append(roc_auc_score(val_y, val_pred))

    print("保存训练集结果")
    np.savetxt(clf_name+"train.csv", train, delimiter=",")
    print("%s_scotrainre_list:" % clf_name, cv_scores)
    print("%s_score_mean:" % clf_name, np.mean(cv_scores))
    print("%s_score_std:" % clf_name, np.std(cv_scores))
    return train, test


def lgb_model(x_train,y_train,x_test):
    lgb_train,lgb_test = cv_model(lgb,x_train,y_train,x_test,'lgb')
    return lgb_train,lgb_test


def xgb_model(x_train,y_train,x_test):
    xgb_train, xgb_test = cv_model(xgb, x_train, y_train, x_test, 'xgb')
    return xgb_train, xgb_test


def cat_model(x_train,y_train,x_test):
    cat_train, cat_test = cv_model(CatBoostRegressor, x_train, y_train, x_test, 'cat')
    return cat_train, cat_test


if __name__=="__main__":
    # 数据预处理
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    data = pd.concat([train, test], axis=0, ignore_index=True)

    # 训练数据/测试数据准备
    features = [f for f in data.columns if f not in ['是否流失', '客户ID']]
    features = [
        '当前设备使用天数','当月使用分钟数与前三个月平均值的百分比变化','每月平均使用分钟数' ,'客户生命周期内的平均每月使用分钟数'
        ,'客户整个生命周期内的平均每月通话次数','已完成语音通话的平均使用分钟数','在职总月数','客户生命周期内的总使用分钟数'
        ,'计费调整后的总分钟数','客户生命周期内的总费用','过去三个月的平均每月通话次数','计费调整后的总费用','使用高峰语音通话的平均不完整分钟数'
        ,'当前手机价格','客户生命周期内的总通话次数','计费调整后的呼叫总数','过去六个月的平均每月使用分钟数','过去六个月的平均每月通话次数'
        ,'过去三个月的平均每月使用分钟数' ,'客户生命周期内平均月费用','平均非高峰语音呼叫数' ,'当月费用与前三个月平均值的百分比变化'
        ,'平均呼入和呼出高峰语音呼叫数','平均月费用','平均未接语音呼叫数','过去六个月的平均月费用','平均接听语音电话数'
        ,'过去三个月的平均月费用','地理区域','一分钟内的平均呼入电话数','信用等级代码','平均超额使用分钟数','平均完成的语音呼叫数'
        ,'尝试拨打的平均语音呼叫次数','平均掉线或占线呼叫数','预计收入','平均掉线语音呼叫数','平均已完成呼叫数','平均尝试调用次数'
        ,'使用客户服务电话的平均分钟数','平均占线语音呼叫数','家庭成人人数','平均超额费用','婚姻状况','平均语音费用','平均客户服务电话次数'
        ,'平均漫游呼叫数','家庭中唯一订阅者的数量','平均呼叫等待呼叫数','新手机用户','手机网络功能','是否双频','家庭活跃用户数','账户消费限额'
        ,'是否翻新机','平均三通电话数','尝试数据调用的平均数','信用卡指示器','数据超载的平均费用','完成数据调用的平均数','平均峰值数据调用次数'
        ,'非高峰数据呼叫的平均数量','信息库匹配'
        # ,'未应答数据呼叫的平均次数','平均呼叫转移呼叫数','平均占线数据调用次数','平均丢弃数据呼叫数'
    ]
    train = data[data['是否流失'].notnull()].reset_index(drop=True)
    test = data[data['是否流失'].isnull()].reset_index(drop=True)
    x_train = train[features]
    x_test = test[features]
    y_train = train['是否流失']

    # 调用模型
    print("*****************lgb模型训练**********************")
    lgb_train, lgb_test = lgb_model(x_train, y_train, x_test)
    test['是否流失_lgb'] = lgb_test
    print("*****************xgb模型训练**********************")
    xgb_train, xgb_test = xgb_model(x_train, y_train, x_test)
    test['是否流失_xgb'] = xgb_test
    print("*****************catboost模型训练**********************")
    cat_train, cat_test = cat_model(x_train, y_train, x_test)
    test['是否流失_cat'] = cat_test

    test['是否流失'] = (test['是否流失_lgb'] + test['是否流失_xgb'] + test['是否流失_cat'])/3
    test.to_csv('sub_all.csv', index=False)
    test[['客户ID', '是否流失']].to_csv('sub_sample.csv', index=False)