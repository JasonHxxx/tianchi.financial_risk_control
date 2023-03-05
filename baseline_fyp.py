#!/usr/bin/env python
# coding: utf-8

# In[11]:


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

chosen_numerical_columns = ['loanAmnt', 'term', 'interestRate', 'installment', 'grade', 'employmentTitle', 'employmentLength', 'homeOwnership', 'annualIncome', 'verificationStatus', 'issueDate', 'purpose', 'postCode', 'dti', 'delinquency_2years', 'ficoRangeLow', 'ficoRangeHigh', 'openAcc', 'pubRec', 'pubRecBankruptcies', 'revolBal', 'revolUtil', 'totalAcc', 'initialListStatus', 'applicationType', 'earliesCreditLine', 'title', 'n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n13', 'n14']




def feature_selection(data: pd.DataFrame):
    # 填写所有为空的值
    def fill_na(data: pd.DataFrame):
        # 按照中位数填写数值类型
        numerical_feats = list(data.select_dtypes(exclude=['object']).columns)
        category_feats = list(data.select_dtypes(include=['object']).columns)
        for numerical_feat in numerical_feats:
            data[numerical_feat] = data[numerical_feat].fillna(data[numerical_feat].median())

        for category_feat in category_feats:
            data[category_feat] = data[category_feat].fillna(data[category_feat].mode()[0])  # 取出第一行的值填入即可

    # 日期转换为int类型
    def transform_datetime(df: pd.DataFrame):
        # 转化成时间格式
        df['issueDate'] = pd.to_datetime(df['issueDate'], format='%Y-%m-%d')
        start_date = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
        # 构造时间特征
        df['issueDate'] = df['issueDate'].apply(lambda x: x - start_date).dt.days

    # 时间为序列数据，转化为连续时间|就业年限
    def transform_employment_length(data: pd.DataFrame):
        data['employmentLength'].replace(to_replace='10+ years', value='10 years', inplace=True)
        data['employmentLength'].replace('< 1 year', '0 years', inplace=True)
        data['employmentLength'] = data['employmentLength'].apply(lambda x: int(x.split()[0]))

    # 时间为序列数据，转化为连续时间|开通时间
    def transform_earliest_credit_line(data: pd.DataFrame):
        import calendar
        def month_year_to_month_count(s):
            month = list(calendar.month_abbr).index(s[0:3])
            year = int(s[-4:])
            return 12 * (year - 1944) + month

        data['earliesCreditLine'] = data['earliesCreditLine'].apply(lambda s: month_year_to_month_count(s))

    # 等级为序列数据，转化为连续数值|等级
    def transform_grade(data: pd.DataFrame):
        data['grade'] = data['grade'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7})

    def bin(data: pd.DataFrame):
        data['loanAmnt'] = pd.qcut(data['loanAmnt'], 10, labels=False)
        data['interestRate'] = pd.qcut(data['interestRate'], 10, labels=False)
        data['ficoRangeLow'] = pd.qcut(data['ficoRangeLow'], 10, labels=False)
        data['ficoRangeHigh'] = pd.qcut(data['ficoRangeHigh'], 10, labels=False)
        data['totalAcc'] = pd.qcut(data['totalAcc'], 10, labels=False)



    print('填充空缺值...')
    fill_na(data)
    print('等级数值化...')
    transform_grade(data)
    print('时间数值化...')
    transform_datetime(data)
    print('就业年限数值化...')
    transform_employment_length(data)
    print('开通时间数值化...')
    transform_earliest_credit_line(data)
    bin(data)
    data.drop(columns=['id'], inplace=True)
    numerical_feats = list(data.select_dtypes(exclude=['object']).columns)
    #判断当前特征处理数据集是否是训练集
    is_train_data = False

    # 训练数据
    if('isDefault' in numerical_feats):
        is_train_data=True #含有标签，是训练数据
        y_train = data['isDefault']
        numerical_feats.remove('isDefault')
        x_train = data.drop(columns='isDefault')
        x_train = x_train[numerical_feats]
        select_k_best = SelectKBest(k=35).fit(x_train, y_train)
        column_indexes = select_k_best.get_support(indices=True)
        column_list = list(x_train.columns)
        chosen_columns = []
        for column_index in column_indexes:
            chosen_columns.append(column_list[column_index])
        print('chosen_columns：',len(chosen_columns),chosen_columns)
    # 选中的数值型标签
    category_columns = list(data.select_dtypes(include=['object']).columns)
    label = 'isDefault'

    # 保留列
    reserved_columns = chosen_numerical_columns+category_columns
    if(is_train_data):
        reserved_columns.append(label)
    print('保留列：',len(reserved_columns),reserved_columns)

    # 取反操作，去掉不在保留列的所有值。
    to_drop_columns = []
    column_list = list(data.columns)
    for column in column_list:
        if column not in reserved_columns:
            to_drop_columns.append(column)

    return data.drop(columns=to_drop_columns)






def feature_xgb(data: pd.DataFrame):
    
    is_train_data = ('isDefault' in list(data.columns))
    data=feature_selection(data)
    # 对非序列化特征进行onehot编码
    category_features = list(data.select_dtypes(include=['object']).columns)
    return pd.get_dummies(data, columns=category_features, drop_first=True)

def feature_catb(data: pd.DataFrame):
    is_train_data = ('isDefault' in list(data.columns))
    data = feature_selection(data)

    #对于待预测数据，需要将特征字符串化
    if not is_train_data:
        with open('middle-product/feature_param_catb.json', 'r') as file:
            dic = json.load(file)
            cat_features = dic['cat_features']
            file.close()
            print('cat_features加载完成！')
        for i in data.columns:
            if i in cat_features:
                data[i] = data[i].astype('str')

    return data



def save_train_feature_xgb():
    train = pd.read_csv('train.csv')
    train_featured = feature_xgb(train)
    train_featured.to_csv('middle-product/train_featured_xgb.csv',index=False)
    print('xgb训练集特征已存储')

raw_cat_features = [
 'term',
 'subGrade',
 'homeOwnership',
 'verificationStatus',
 'purpose',
 'initialListStatus',
 'applicationType',
 'policyCode',
 'n11',
 'n12']

def save_train_feature_catb():
    train = pd.read_csv('train.csv')
    train_featured = feature_catb(train)
    cat_features = []
    for cat_feature in raw_cat_features:
        if cat_feature in train_featured.columns:
            cat_features.append(cat_feature)
    train_featured.to_csv('middle-product/train_featured_catb.csv',index=False)
    print('catb训练集特征已存储')

    feature_param = {}
    feature_param['cat_features'] = cat_features
    with open('middle-product/feature_param_catb.json','w') as file:
        json.dump(feature_param,file)
    print('cat特征参数已经存储')


save_train_feature_xgb()
save_train_feature_catb()


# In[12]:


def final_train_xgb():
    # 读取已经特征处理的数据
    # 读取已经特征处理的数据
    train = pd.read_csv('middle-product/train_featured_xgb.csv')
    print('数据加载完成！')

    print('*' * 10)
    x_train = train.drop(columns='isDefault')
    y_train = train['isDefault']

    print('xgb x_train:')
    x_train.info()

    ### fit model for train data
    model = xgb.XGBClassifier(
        learning_rate=0.12,  # 学习率
        n_estimators=800,  # 树的个数--1000棵树建立xgboost，也就是迭代次数
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
              y_train)
    print('训练完成')
    model.save_model('middle-product/model_xgb.json')
    print("最终训练")











def final_train_catb():

    with open('middle-product/feature_param_catb.json', 'r') as file:
        dic = json.load(file)
        cat_features = dic['cat_features']
        file.close()
        print('cat_features加载完成！')

    # 读取已经特征处理的数据
    # 读取已经特征处理的数据
    train = pd.read_csv('middle-product/train_featured_catb.csv')
    print('数据加载完成！')
    # print(train.info())
    for i in train.columns:
        if i in cat_features:
            train[i] = train[i].astype('str')

    print('*' * 10)
    x_train = train.drop(columns='isDefault')
    y_train = train['isDefault']

    print('catb x_train:')
    x_train.info()
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        task_type="CPU",
        learning_rate=0.12,
        iterations=800,
        random_seed=2023,
        od_type="Iter",
        depth=7,
        subsample=0.9)

    model.fit(x_train,
              y_train,cat_features=cat_features)
    print('训练完成')
    model.save_model('middle-product/model_catb.json')
    print("已经将模型持久化")



final_train_xgb()
final_train_catb()


# In[14]:


class VotingModel:
    #总权重
    _weight_sum=0
    #模型列表
    _model_list=[]

    def _load_model(self,model_type, model_file,model_weight):
        print('开始加载',model_file)
        if(model_type == 'xgb'):
            model = xgb.XGBClassifier()
        if(model_type == 'catb'):
            model = CatBoostClassifier()

        model.load_model(model_file)
        model_item = {}
        model_item['model_type']=model_type
        model_item['model'] = model
        model_item['model_weight'] = model_weight
        self._model_list.append(model_item)
        print('加载完成', model_file)

    def __init__(self,model_config):
        for item in model_config:
            model_weight = item['model_weight']
            model_type = item['model_type']
            model_file = item['model_file']
            self._weight_sum += model_weight
            self._load_model(model_type,model_file,model_weight)

        print('模型加载完成，权重和为：',self._weight_sum)
        print('模型配置为：',self._model_list)

    def _get_predict(self,model_type,test:pd.DataFrame,model):
        test_predict = None
        if (model_type == 'xgb'):
            featured_test =  feature_xgb(test)
            print("xgb预测数据的特征:")
            featured_test.info()
            test_predict = model.predict_proba(featured_test)[:, -1]
        if(model_type == 'catb'):
            featured_test =  feature_catb(test)
            print("catb预测数据的特征:")
            featured_test.info()
            test_predict = model.predict(featured_test, prediction_type='Probability')[:, -1]

        return test_predict

    # 读取模型列表，获得不同的特征数据后，进行按权重地预测
    def fit(self,test:pd.DataFrame):
        final_predict = None
        for model_item in self._model_list:
            test_copy = test.copy()
            model_type = model_item['model_type']
            model = model_item['model']
            model_weight = model_item['model_weight']
            test_predict = self._get_predict(model_type,test_copy,model)
            print(model_item,'预测结果：',test_predict)
            if final_predict is None:
                final_predict = test_predict*(model_weight/self._weight_sum)
            else:
                final_predict = final_predict + test_predict*(model_weight/self._weight_sum)

        return final_predict


def predict(xgb_weight=1,catb_weight=2):

    test = pd.read_csv('testA.csv')
    # print(type(test[['id']]))#<class 'pandas.core.frame.DataFrame'>
    # print(type(test['id']))#<class 'pandas.core.series.Series'>

    model_config = [
        {
            'model_type':'xgb',
            'model_file':'middle-product/model_xgb.json',
            'model_weight': xgb_weight
        },
        {
            'model_type': 'catb',
            'model_file': 'middle-product/model_catb.json',
            'model_weight': catb_weight
        },
    ]
    voting_model = VotingModel(model_config)
    out = test[['id']].copy()
    out['isDefault'] = voting_model.fit(test)

    print('预测输出')
    out.info()
    file_name = 'predict/predict' + '___xgb_weight-' +str(xgb_weight)+'___cat_weight-'+str(catb_weight) +'___'+ str(time.time())+'.csv'
    out.to_csv(file_name,index=False)
    print("已经完成预测")
predict(1,1)


# In[ ]:




