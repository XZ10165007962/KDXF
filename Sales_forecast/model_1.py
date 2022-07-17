#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: model.py
@time: 2022/7/5 16:10
@version:
@desc:
"""

import warnings
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_absolute_error
import copy

warnings.filterwarnings('ignore')


# 构建模型
def cv_model(clf, train_x, train_y, test_x, clf_name):
    folds = 5
    seed = 2022
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    train = np.zeros(train_x.shape[0])
    test = np.zeros(test_x.shape[0])

    cv_scores = []
    for i, (train_index,valid_index) in enumerate(kf.split(train_x, train_y)):
        print('**************** {} *****************'.format(str(i+1)))
        trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], train_y[valid_index]

        if clf_name == 'lgb':
            train_matrix = clf.Dataset(trn_x, label=trn_y)
            valid_matrix = clf.Dataset(val_x, label=val_y)

            params = {
                'boosting_type': 'gbdt',
                'objective': 'tweedie',  # 回归问题
                'metric': 'L2 loss',  # 评价指标
                'min_child_weight': 3,
                'num_leaves': 2**5,
                'lambda_l2': 10,
                'feature_fraction': 0.75,
                'bagging_fraction': 0.75,
                'bagging_freq': 10,
                'learning_rate': 0.15,
                'seed': 2022,
                # 'max_depth': 10,
                'verbose': -1,
                'n_jobs': -1
            }

            model = clf.train(
                params, train_set=train_matrix, num_boost_round=50000, valid_sets=[train_matrix, valid_matrix],
                categorical_feature=[], verbose_eval=3000, early_stopping_rounds=200
            )
            val_pred = model.predict(data=val_x, num_iteration=model.best_iteration)
            test_pred = model.predict(data=test_x, num_iteration=model.best_iteration)
            print(list(
                sorted(
                    zip(train_x.columns, model.feature_importance('gain')),
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
        cv_scores.append(mean_absolute_error(val_y, val_pred))

    print("%s_train_score_list:" % clf_name, cv_scores)
    print("%s_score_mean:" % clf_name, np.mean(cv_scores))
    print("%s_score_std:" % clf_name, np.std(cv_scores))
    return train, test


def one_model(clf, train_x, train_y, test_x, clf_name, val_x, val_y):

    if clf_name == 'lgb':
        train_matrix = clf.Dataset(train_x, label=train_y)
        valid_matrix = clf.Dataset(val_x, label=val_y)

        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression_l1',  # 回归问题
            'metric': 'mse',  # 评价指标
            'min_child_weight': 3,
            'num_leaves': 2 ** 3,
            'lambda_l2': 10,
            'feature_fraction': 0.5,
            'bagging_fraction': 0.5,
            'bagging_freq': 10,
            'learning_rate': 0.1,
            'seed': 2022,
            'max_depth': 3,
            'verbose': -1,
            'n_jobs': -1
        }
        model = clf.train(
            params, train_set=train_matrix, num_boost_round=50000, valid_sets=[train_matrix, valid_matrix],
            categorical_feature=[], verbose_eval=3000, early_stopping_rounds=3000
        )
        val_pred = model.predict(data=val_x, num_iteration=model.best_iteration)
        test_pred = model.predict(data=test_x, num_iteration=model.best_iteration)
        print(list(
            sorted(
                zip(train_x.columns, model.feature_importance('gain')),
                key=lambda x: x[1], reverse=True)
        ))
        print("%s_score:" % clf_name, mean_absolute_error(val_y, val_pred))

        return val_pred, test_pred


def lgb_model(x_train, y_train, x_test, *args):
    if args is not None:
        lgb_train, lgb_test = one_model(lgb, x_train, y_train, x_test, 'lgb', args[0], args[1])
    else:
        lgb_train, lgb_test = cv_model(lgb, x_train, y_train, x_test, 'lgb')
    return lgb_train, lgb_test


def xgb_model(x_train, y_train, x_test):
    xgb_train, xgb_test = cv_model(xgb, x_train, y_train, x_test, 'xgb')
    return xgb_train, xgb_test


def cat_model(x_train, y_train, x_test):
    cat_train, cat_test = cv_model(CatBoostRegressor, x_train, y_train, x_test, 'cat')
    return cat_train, cat_test