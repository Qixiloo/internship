from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from metrics import *
from sampler import *
from sklearn import tree
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from  xgboost import XGBClassifier
import statsmodels.api as sm
from sklearn.tree import DecisionTreeClassifier


from sklearn.model_selection import StratifiedKFold  # 数据切分、分层五折验证包
import lightgbm as lgb  # lgb模型 ,安装的方法是在anaconda promote里，直接pip install lightgbm 即可，做第一层1模型
import xgboost as xgb  # xgb模型，安装的方法是在anaconda promote里，直接pip install xgboost 即可，和lightgbm一样，做第一层2模型



    
    # X_train, y_train = X_res, y_res
    #X_res,y_res=randomSample(X,y)
    # X_res,y_res=SMOTEENNSample
def Smoteover(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    X_res,y_res=over_smote_Sample(X_train,y_train)
    return X_res,y_res,X_test,y_test

def Randomunder(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    X_res,y_res=random_under_Sample(X_train,y_train)
    return X_res,y_res,X_test,y_test

def Smoteenn(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    X_res,y_res=SMOTEENN_Sample(X_train,y_train)
    return X_res,y_res,X_test,y_test

def withoutSample(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    return X_train,y_train,X_test,y_test
    


def RandomForest(X_train, y_train,X_test,y_test):
    rf = RandomForestClassifier(n_estimators=100,class_weight='balanced')
    rf.fit(X_train, y_train)
    print('RandomForestClassifier')
    y_train_predict_prob = rf.predict_proba(X_train)[:, 1]
    print(y_train_predict_prob)
    y_test_predict_prob = rf.predict_proba(X_test)[:, 1]
    metrices_opt(y_train, y_test, y_train_predict_prob, y_test_predict_prob)
    metrics_plot(y_train, y_test, y_train_predict_prob, y_test_predict_prob)

def DecisionTree(X_train, y_train,X_test,y_test):
    tree=DecisionTreeClassifier(class_weight='balanced')
    tree.fit(X_train, y_train)
    print('DecisionTreeClassifier')
    y_train_predict_prob = tree.predict_proba(X_train)[:, 1]
    y_test_predict_prob = tree.predict_proba(X_test)[:, 1]
    metrices_opt(y_train, y_test, y_train_predict_prob, y_test_predict_prob)
    metrics_plot(y_train, y_test, y_train_predict_prob, y_test_predict_prob)


def DecisionTree_cut(X_train, y_train,X_test,y_test):
    tree=DecisionTreeClassifier(class_weight='balanced',max_features=24)
    tree.fit(X_train, y_train)
    print('DecisionTreeClassifier')
    y_train_predict_prob = tree.predict_proba(X_train)[:, 1]
    y_test_predict_prob = tree.predict_proba(X_test)[:, 1]
    metrices_opt(y_train, y_test, y_train_predict_prob, y_test_predict_prob)
    metrics_plot(y_train, y_test, y_train_predict_prob, y_test_predict_prob)


def find_best_tree(X_train, y_train,X_test,y_test):
    test = []
    for i in range(24):
        clf = tree.DecisionTreeClassifier(max_depth=i+1
                                        ,criterion="entropy"
                                        ,random_state=30
                                        ,splitter="random"
                                        )
        clf = clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        test.append(score)
    plt.plot(range(1,25),test,color="red",label="max_depth")
    plt.legend()
    plt.show()

def Logistic(X_train, y_train,X_test,y_test):
    lr = LogisticRegression(C=0.01, penalty='l1', solver='liblinear', max_iter=10000, n_jobs=-1, class_weight='balanced')
    lr.fit(X_train, y_train)
    print('LogisticRegression')
    y_train_predict_prob = lr.predict_proba(X_train)[:, 1]
    y_test_predict_prob = lr.predict_proba(X_test)[:, 1]
    print(lr.coef_, lr.intercept_)
    metrices_opt(y_train, y_test, y_train_predict_prob, y_test_predict_prob)
    metrics_plot(y_train, y_test, y_train_predict_prob, y_test_predict_prob)
   

def Logistic_coe(X,y):
    logit_model=sm.Logit(y,X)
    result=logit_model.fit()
    print(result.summary2())



def SVM(X_train, y_train,X_test,y_test):
    sc=StandardScaler()
    sc.fit_transform(X_train)
    X_train_std=sc.transform(X_train)
    X_test_std=sc.transform(X_test)
    svm = SVC(kernel='linear',probability=True)
    svm.fit(X_train_std, y_train)
    y_train_predict_prob = svm.predict_proba(X_train_std)[:, 1]
    y_test_predict_prob = svm.predict_proba(X_test_std)[:, 1]
    metrices_opt(y_train, y_test, y_train_predict_prob, y_test_predict_prob)
    metrics_plot(y_train, y_test, y_train_predict_prob, y_test_predict_prob)



def xgb(X_train, y_train,X_test,y_test):
    x = XGBClassifier(
        n_estimators=203,
        learning_rate=0.06,  
        eval_metric = 'auc'
                            )                      # n_estimators=205,learning_rate=0.06
                                                # x_res, y_res  x_train,y_train
    x.fit(X_train, y_train,eval_metric = 'auc')
    y_train_predict_prob = x.predict_proba(X_train)[:, 1]
    y_test_predict_prob = x.predict_proba(X_test)[:, 1]
    metrices_opt(y_train, y_test, y_train_predict_prob, y_test_predict_prob)
    metrics_plot(y_train, y_test, y_train_predict_prob, y_test_predict_prob)


"""
X= pd.read_csv('data/data.csv')
y= pd.read_csv('data/labels.csv')

X=X.iloc[:,1:].fillna(0).drop(['Indcd'],axis=1)
y=y.iloc[:,1:]

X_train, y_train,X_test,y_test=Smoteenn(X,y)

SVM(X_train, y_train,X_test,y_test)
"""

"""
def get_leaderboard_score(test_df,prediction):
    
    定义评分函数
    test_df: 测试集
    prediction: 预测结果
    reture: 输出结果分数
    
    label = test_df['label'].values  # 拿出真实样本
    assert len(prediction) == len(label)  # 断言其长度相等
    print('stacking auc score: ', roc_auc_score(label, prediction))  # 计算评分



def fusion(X, y,X_test,y_test):
    feats = X.columns  # 拿出特征
    data_seed = 2020
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=data_seed)
    # lgb和xgb的参数
    lgb_params = {
        'boosting': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.06,
        'num_leaves': 31,
        'max_depth': -1,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'feature_fraction': 0.8,
        'feature_fraction_seed': 300, ### 特征抽样的随机种子
        'bagging_seed': 3, ### 数据抽样的随机种子,取10个不同的，然后对结果求平均,todo:求出10个结果，然后求平均
        #'is_unbalance': True   #### 第一种方法：设置is_unbalance为True，表明传入的数据集是类别不平衡的
        #'scale_pos_weight': 98145/1855###负样本数量/正样本数量 -> scale_pos_weight * 正样本 == 负样本
    }
    xgb_params = {
        'booster': 'gbtree',  ##提升类型
        'objective': 'binary:logistic',  ###目标函数
        'eval_metric': 'auc',  ##评价函数
        'eta': 0.1,  ### 学习率 ，一般0.0几
        'max_depth': 6,  ###树最大深度
        'min_child_weight': 1,  ###最小样本二阶梯度权重, 取值是整数
        'subsample': 1.0,  ###训练数据采样 ,取值0.0~1.0之间
        'colsample_bytree': 1.0,  ###训练特征采样，取值0.0~1.0之间
        'lambda': 1,  ## l2正则，取值是整数
        'alpha': 0,   ### l1正则，取值整数
        'silent': 1   ### 取值1控制xgboost训练信息不输出
    }

    blend_train = pd.DataFrame()  # 定义df数据接收验证集结果，作为特征
    blend_test = pd.DataFrame()  # 定义df数据接收测试集结果，作为特征
    # 训练lgb，用作第一层模型中的其中一个
    test_pred_lgb = 0  # 预测结果存放对象
    cv_score_lgb = []  # 存放每次auc的对象
    train_feats = np.zeros(X.shape[0])  # 整体训练的样本数量
    for idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print('training fold: ', idx + 1)  # 遍历的第几次
        train_x, valid_x = X[train_idx], X[test_idx]  # 拆分成训练集和验证集
        train_y, valid_y = y[train_idx], y[test_idx]  # 拆分成训练集和验证集
        dtrain = lgb.Dataset(train_x, train_y, feature_name=feats)  # 组成训练集
        dvalid = lgb.Dataset(valid_x, valid_y, feature_name=feats)  # 组成验证集
        model = lgb.train(lgb_params, dtrain, num_boost_round=2000, valid_sets=dvalid, early_stopping_rounds=50, verbose_eval=50)  # 定义lgb模型

        valid_pred = model.predict(valid_x, num_iteration=model.best_iteration)  # 当前模型最佳参数并预测，num_iteration：选择最优的lgb
        train_feats[test_idx] = valid_pred  # 每次把验证集的结果填入，做训练的结果集，由于是5折，所以每次都是1/5的数据，把它们当作lgb训练集特征
        auc_score = roc_auc_score(valid_y, valid_pred)  # 计算auc
        print('auc score: ', auc_score)
        cv_score_lgb.append(auc_score)  # 存放验证集auc值
        test_pred_lgb += model.predict(X_test, num_iteration=model.best_iteration)  # 预测结果并累加，做预测的结果集，把它们当作lgb测试集当作特征
        
    print("训练的结果:",train_feats)
    test_pred_lgb /= 5
    print("测试的结果", test_pred_lgb)  # 由于测试的结果是5折每次的累加，所以需要除于5
    # 将训练结果和预测结果加入到blend数据集
    blend_train['lgb_feat'] = train_feats
    blend_test['lgb_feat'] = test_pred_lgb

    # 训练xgb，用作第一层模型中的其中一个
    test_pred_xgb = 0 # 预测结果存放对象
    cv_score_xgb = []  # 存放每次auc的对象
    train_feats_xgb = np.zeros(X.shape[0])  # 整体训练的样本数量
    for idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print('training fold: ', idx + 1) # 遍历的第几次
        train_x, valid_x = X[train_idx], X[test_idx]  # 拆分成训练集和验证集
        train_y, valid_y = y[train_idx], y[test_idx]  # 拆分成训练集和验证集
        dtrain = xgb.DMatrix(train_x, train_y, feature_names=feats)  # 组成训练集
        dvalid = xgb.DMatrix(valid_x, valid_y, feature_names=feats)  # 组成验证集
        watchlist = [(dvalid, 'eval')]
        model = xgb.train(xgb_params, dtrain, num_boost_round=2000, evals=watchlist, early_stopping_rounds=50, verbose_eval=50)  # 定义xgb模型

        valid_pred = model.predict(dvalid, ntree_limit=model.best_iteration)  # 当前模型最佳参数并预测，ntree_limit：选择最优的xgb
        train_feats_xgb[test_idx] = valid_pred  # 每次把验证集的结果填入，做训练的结果集，由于是5折，所以每次都是1/5的数据
        auc_score = roc_auc_score(valid_y, valid_pred)  # 计算auc
        print('auc score: ', auc_score)
        cv_score_xgb.append(auc_score)  # 存放验证集auc值
        dtest = xgb.DMatrix(X_test,feature_names=feats)  ##同时指定特征名字
        test_pred_xgb += model.predict(dtest, ntree_limit=model.best_iteration)  # 预测结果并累加，做预测的结果集
        
    print("训练的结果:", train_feats_xgb)
    train_feats_xgb /= 5  # 由于测试的结果是5折每次的累加，所以需要除于5
    print("测试的结果:", train_feats_xgb)
    # 将训练结果和预测结果加入到blend数据集
    blend_train['xgb_feat'] = train_feats_xgb
    blend_test['xgb_feat'] = test_pred_xgb

    print(blend_train.head(5))  # 查看训练集作为特征的情况
    print(blend_test.head(5))  #  查看测试集作为特征的情况
    # 第二层模型lr训练
    lr_model = LogisticRegression()  # 实例化
    lr_model.fit(blend_train.values, y)  # 训练
    print("特征权重:", lr_model.coef_)
    test_pred_lr = lr_model.predict_proba(blend_test.values)[:,1]  # 第二层模型预测结果

    # 各个模型的评分情况
    #get_leaderboard_score(test,test_pred_lr)  # stacking模型的分数
    """