#!/usr/bin/env python
# coding: utf-8

# In[3]:


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

def adjust_param_catb():
    with open('middle-product/feature_param_catb.json', 'r') as file:
        dic = json.load(file)
        cat_features = dic['cat_features']
        file.close()
        print('cat_features加载完成！')

    def get_max():
        # 读取已经特征处理的数据
        # 读取已经特征处理的数据
        train = pd.read_csv('middle-product/train_featured_catb.csv')
        print('数据加载完成！')
        # 根据模型要求修改其类型（加载时的类型识别为数值）

        for i in train.columns:
            if i in cat_features:
                train[i] = train[i].astype('str')
        print('数据已经完成类别转换')

        print('*' * 10)
        x_train = train.drop(columns='isDefault')
        y_train = train['isDefault']

        def catb_cv(depth, learning_rate, subsample, rsm,scale_pos_weight):
            # 建立模型
            model = CatBoostClassifier(
                scale_pos_weight=scale_pos_weight,
                loss_function="Logloss",
                eval_metric="AUC",
                task_type="CPU",
                learning_rate=learning_rate,
                iterations=200,
                random_seed=2023,
                od_type="Iter",
                depth=int(depth),
                subsample=subsample,
                rsm=rsm) # 防止输出溢出
            val = cross_val_score(model, x_train, y_train, cv=5, scoring='roc_auc',
                                  fit_params={'cat_features': cat_features}).mean()
            return val

        # """定义优化参数"""
        bayes = BayesianOptimization(
            catb_cv,
            {
                'depth': (5, 10),
                'learning_rate': (0.05, 0.3),
                'subsample': (0.6, 0.95),
                'rsm': (0.6, 1.0),  ## 列采样比率，别名colsample_bylevel 取值（0，1],默认值1
                'scale_pos_weight':(0.5,2)
            }
        )

        """开始优化"""
        bayes.maximize(init_points=5,  # init_points：初始点，int，可选（默认值=5），探索开始探索之前的迭代次数
                       n_iter=25)  # 最大迭代次数，int，可选（默认值=25），方法试图找到最大值的迭代次数

        print(bayes.max)
        return bayes.max



    def get_iters(max_params: dict):
        # 读取已经特征处理的数据
        # 读取已经特征处理的数据
        train = pd.read_csv('./middle-product/train_featured_catb.csv')
        print('数据加载完成！')
        for i in train.columns:
            if i in cat_features:
                train[i] = train[i].astype('str')

        print('*' * 10)
        x_train = train.drop(columns='isDefault')
        y_train = train['isDefault']

        # 数据集分割
        # x_train: 训练集的特征
        # x_test:  测试集的特征
        # y_train: 训练集的标签
        # t_test:  测试集的标签，
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=2023)
        ### fit model for train data
        model = CatBoostClassifier(
            scale_pos_weight=max_params['scale_pos_weight'],
            loss_function="Logloss",
            eval_metric="AUC",
            task_type="CPU",
            learning_rate=max_params['learning_rate'],
            iterations=8000,
            random_seed=2023,
            od_type="Iter",
            depth=int(max_params['depth']),
            subsample=max_params['subsample'],
            rsm=max_params['rsm'])

        model.fit(x_train,
                  y_train,
                  eval_set=[(x_test, y_test)],
                  verbose=20,
                  early_stopping_rounds=200,
                  cat_features=cat_features)
        print('训练完成')

    max = get_max()
    max_params = max['params']
#     get_iters(max_params)
    print("执行完成")
    return max

max= adjust_param_catb()


# In[6]:


def get_iterations():
    with open('middle-product/feature_param_catb.json', 'r') as file:
        dic = json.load(file)
        cat_features = dic['cat_features']
        file.close()
        print('cat_features加载完成！')
    # 读取已经特征处理的数据
    # 读取已经特征处理的数据
    train = pd.read_csv('./middle-product/train_featured_catb.csv')
    print('数据加载完成！')
    for i in train.columns:
        if i in cat_features:
            train[i] = train[i].astype('str')

    print('*' * 10)
    x_train = train.drop(columns='isDefault')
    y_train = train['isDefault']

    # 数据集分割
    # x_train: 训练集的特征
    # x_test:  测试集的特征
    # y_train: 训练集的标签
    # t_test:  测试集的标签，
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=2023)
    ### fit model for train data
    model = CatBoostClassifier(
        scale_pos_weight=1.5,
        loss_function="Logloss",
        eval_metric="AUC",
        task_type="CPU",
        learning_rate=0.06,
        iterations=8000,
        random_seed=2023,
        od_type="Iter",
        depth=7,
        subsample=0.88,
        rsm=0.8)

    model.fit(x_train,
              y_train,
              eval_set=[(x_test, y_test)],
              verbose=20,
              early_stopping_rounds=200,
              cat_features=cat_features)
    print('训练完成')
get_iterations()


# In[9]:


def get_iterations1():
    with open('middle-product/feature_param_catb.json', 'r') as file:
        dic = json.load(file)
        cat_features = dic['cat_features']
        file.close()
        print('cat_features加载完成！')
    # 读取已经特征处理的数据
    # 读取已经特征处理的数据
    train = pd.read_csv('./middle-product/train_featured_catb.csv')
    print('数据加载完成！')
    for i in train.columns:
        if i in cat_features:
            train[i] = train[i].astype('str')

    print('*' * 10)
    x_train = train.drop(columns='isDefault')
    y_train = train['isDefault']

    # 数据集分割
    # x_train: 训练集的特征
    # x_test:  测试集的特征
    # y_train: 训练集的标签
    # t_test:  测试集的标签，
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=2023)
    ### fit model for train data
    model = CatBoostClassifier(
        scale_pos_weight=1,
        loss_function="Logloss",
        eval_metric="AUC",
        task_type="CPU",
        learning_rate=0.1,
        iterations=8000,
        random_seed=2023,
        od_type="Iter",
        depth=7,
        subsample=0.85,
        rsm=0.8)

    model.fit(x_train,
              y_train,
              eval_set=[(x_test, y_test)],
              verbose=20,
              early_stopping_rounds=200,
              cat_features=cat_features)
    print('训练完成')
get_iterations1()


# In[16]:


def get_iterations2():
    with open('middle-product/feature_param_catb.json', 'r') as file:
        dic = json.load(file)
        cat_features = dic['cat_features']
        file.close()
        print('cat_features加载完成！')
    # 读取已经特征处理的数据
    # 读取已经特征处理的数据
    train = pd.read_csv('./middle-product/train_featured_catb.csv')
    print('数据加载完成！')
    for i in train.columns:
        if i in cat_features:
            train[i] = train[i].astype('str')

    print('*' * 10)
    x_train = train.drop(columns='isDefault')
    y_train = train['isDefault']

    # 数据集分割
    # x_train: 训练集的特征
    # x_test:  测试集的特征
    # y_train: 训练集的标签
    # t_test:  测试集的标签，
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=2023)
    ### fit model for train data
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        task_type="CPU",
        learning_rate=0.12,
        iterations=8000,
        random_seed=2023,
        od_type="Iter",
        depth=7,
        subsample=0.9)

    model.fit(x_train,
              y_train,
              eval_set=[(x_test, y_test)],
              verbose=50,
              early_stopping_rounds=200,
              cat_features=cat_features)
    print('训练完成')
get_iterations2()


# In[ ]:




