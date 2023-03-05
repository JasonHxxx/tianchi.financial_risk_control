#!/usr/bin/env python
# coding: utf-8

# In[1]:


#特征处理
import time

import pandas as pd
from bayes_opt import BayesianOptimization
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
from sklearn.feature_selection import chi2
from sklearn.model_selection import cross_val_score
import json
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
import datetime

# 调节cgb模型参数
def adjust_param_xgb():
    # 获得收益最大的参数和得分
    def get_max():
        # 读取已经特征处理的数据
        train = pd.read_csv('middle-product/train_featured_xgb.csv')
        print('数据加载完成！')
        print(f'训练数据基本信息：{train.info()}')
        x_train = train.drop(columns='isDefault')
        y_train = train['isDefault']
        def xgb_cv(min_child_weight,
                   max_depth, gamma,
                   subsample,
                   colsample_bytree,
                   reg_alpha,
                   reg_lambda,
                   learning_rate,
                   scale_pos_weight):
            # 建立模型
            model = xgb.XGBClassifier(
                objective='binary:logistic',  # 目标函数
                scale_pos_weight=scale_pos_weight,  # 解决样本个数不平衡的问题
                random_state=2023,
                n_estimators=100,
                min_child_weight=min_child_weight,
                max_depth=int(max_depth),
                gamma=gamma,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                learning_rate=learning_rate
            )

            val = cross_val_score(model, x_train, y_train, cv=5, scoring='roc_auc').mean()
            return val

        """定义优化参数"""
        bayes = BayesianOptimization(
            xgb_cv,
            {
                'min_child_weight': (0.001, 1),
                'max_depth': (5, 10),
                'gamma': (0.1, 1.0),
                'subsample': (0.6, 0.95),
                'colsample_bytree': (0.6, 0.95),
                'reg_alpha': (1e-2, 100),
                'reg_lambda': (0, 1000),
                'learning_rate': (0.01, 0.15),
                'scale_pos_weight':(0.5,2)
            }
        )

        """开始优化"""
        bayes.maximize(init_points=5,  # init_points：初始点，int，可选（默认值=5），探索开始探索之前的迭代次数
                       n_iter=25)  # 最大迭代次数，int，可选（默认值=25），方法试图找到最大值的迭代次数
        print("最大参数情况：",bayes.max)
        return bayes.max

    # 获得迭代次数
    def get_iters(max_params: dict):
        # 读取已经特征处理的数据
        train = pd.read_csv('middle-product/train_featured_xgb.csv')
        print('数据加载完成！')
        print(f'训练数据基本信息：{train.info()}')
        x_train = train.drop(columns='isDefault')
        y_train = train['isDefault']

        # 数据集分割
        # x_train: 训练集的特征
        # x_test:  测试集的特征
        # y_train: 训练集的标签
        # t_test:  测试集的标签，
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=2023)
        ### fit model for train data
        model = xgb.XGBClassifier(
            learning_rate=max_params['learning_rate'],  # 学习率
            n_estimators=8000,  # 树的个数--1000棵树建立xgboost，也就是迭代次数
            max_depth=int(max_params['max_depth']),  # 树的深度
            min_child_weight=max_params['min_child_weight'],  # 叶子节点最小权重
            gamma=max_params['gamma'],  # 惩罚项中叶子结点个数前的参数
            subsample=max_params['subsample'],  # 随机选择80%样本建立决策树
            colsample_bytree=max_params['colsample_bytree'],  # 随机选择80%特征建立决策树
            reg_alpha=max_params['reg_alpha'],
            reg_lambda=max_params['reg_lambda'],
            objective='binary:logistic',  # 目标函数
            scale_pos_weight=max_params['scale_pos_weight'],  # 0:640390 1:159610 解决样本个数不平衡的问题 sum(negative instances) / sum(positive instances)
            random_state=2023,  # 随机数
        )

        model.fit(x_train,
                  y_train,
                  eval_set=[(x_test, y_test)],
                  eval_metric="auc",
                  verbose=20,
                  early_stopping_rounds=200)
        print('训练完成')

    max = get_max()
    max_params = max['params']
    get_iters(max_params)
    print("执行完成")
    return max
max = adjust_param_xgb()


# In[5]:


def get_iterations():
    # 读取已经特征处理的数据
    train = pd.read_csv('middle-product/train_featured_xgb.csv')
    print('数据加载完成！')
    print(f'训练数据基本信息：{train.info()}')
    x_train = train.drop(columns='isDefault')
    y_train = train['isDefault']

    # 数据集分割
    # x_train: 训练集的特征
    # x_test:  测试集的特征
    # y_train: 训练集的标签
    # t_test:  测试集的标签，
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=2023)
    ### fit model for train data
    model = xgb.XGBClassifier(
        learning_rate=0.08,  # 学习率
        n_estimators=8000,  # 树的个数--1000棵树建立xgboost，也就是迭代次数
        max_depth=8,  # 树的深度
        min_child_weight=0.3,  # 叶子节点最小权重
        gamma=0.2,  # 惩罚项中叶子结点个数前的参数
        subsample=0.85,  # 随机选择80%样本建立决策树
        colsample_bytree=0.8,  # 随机选择80%特征建立决策树
        reg_alpha=32.0,
        reg_lambda=200,
        objective='binary:logistic',  # 目标函数
        scale_pos_weight=1.6,  # 0:640390 1:159610 解决样本个数不平衡的问题 sum(negative instances) / sum(positive instances)
        random_state=2023,  # 随机数
    )

    model.fit(x_train,
              y_train,
              eval_set=[(x_test, y_test)],
              eval_metric="auc",
              verbose=20,
              early_stopping_rounds=200)
    print('训练完成')
get_iterations()


# In[12]:


def get_iterations2():
    # 读取已经特征处理的数据
    train = pd.read_csv('middle-product/train_featured_xgb.csv')
    print('数据加载完成！')
    print(f'训练数据基本信息：{train.info()}')
    x_train = train.drop(columns='isDefault')
    y_train = train['isDefault']

    # 数据集分割
    # x_train: 训练集的特征
    # x_test:  测试集的特征
    # y_train: 训练集的标签
    # t_test:  测试集的标签，
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=2023)
    ### fit model for train data
    model = xgb.XGBClassifier(
        learning_rate=0.12,  # 学习率
        n_estimators=8000,  # 树的个数--1000棵树建立xgboost，也就是迭代次数
        max_depth=9,  # 树的深度
        min_child_weight=0.5,  # 叶子节点最小权重
        gamma=0.99,  # 惩罚项中叶子结点个数前的参数
        subsample=0.8,  # 随机选择80%样本建立决策树
        colsample_bytree=0.9,  # 随机选择80%特征建立决策树
        reg_alpha=12.0,
        reg_lambda=100,
        objective='binary:logistic',  # 目标函数
        scale_pos_weight=1.2,  # 0:640390 1:159610 解决样本个数不平衡的问题 sum(negative instances) / sum(positive instances)
        random_state=2023,  # 随机数
    )

    model.fit(x_train,
              y_train,
              eval_set=[(x_test, y_test)],
              eval_metric="auc",
              verbose=50,
              early_stopping_rounds=200)
    print('训练完成')
get_iterations2()


# In[ ]:




