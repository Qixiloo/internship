import lightgbm as lgb
from matplotlib import pyplot as plt  
import pandas as pd  
import numpy as np  
from sklearn import metrics  
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix  # 准确率、roc计算、auc计算、混淆矩阵
import itertools  
import gc 
import warnings  
from sklearn.feature_selection import RFE, RFECV  
from imblearn.under_sampling import RandomUnderSampler  
from imblearn.over_sampling import SMOTE
from collections import Counter
from imblearn.combine import SMOTEENN 
from sklearn.linear_model import LogisticRegression


def over_smote_Sample(X, y):

    smote = SMOTE(random_state=42) # radom_state为随机值种子，1:ss[1]+表示label为1的数据增加多少个
    # adasyn = ADASYN(sampling_strategy={0:ss[0],1:ss[1]+800},random_state=2019) # 改变正样本数量参数
    X_resampled, y_resampled = smote.fit_resample(X, y)

    check_num_X = X_resampled.shape[0] - X.shape[0]
    check_num_y = y_resampled.shape[0] - y.shape[0]
    num=X_resampled.shape[0]
    print("*********************************")
    print("SMOTE过采样个数为：", num)
    if (check_num_X == check_num_y):
        print("SMOTE过采样校验：成功")
        print(y_resampled.value_counts())
        return X_resampled, y_resampled
    else:
        print("SMOTE过采样校验：失败")

    
# RandomUnderSampler: 随机选取数据的子集.
def random_under_Sample(X,y):
    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    num=X_resampled.shape[0]
    print("*********************************")
    print("随机欠采样个数为：", num)
    check_num_X = X_resampled.shape[0] - X.shape[0]
    check_num_y = y_resampled.shape[0] - y.shape[0]
    if (check_num_X == check_num_y):
        print("随机采样校验：成功")
        print(y_resampled.value_counts())
        return X_resampled, y_resampled
    else:
        print("随机采样校验：失败")





def SMOTEENN_Sample(X,y):
    sme = SMOTEENN(random_state=42)
    X_resampled, y_resampled =  sme.fit_resample(X, y)
    print("*********************************")
    print("SMOTEENN采样个数为：", X_resampled.shape[0])
    if(X_resampled.shape[0]==y_resampled.shape[0]):
        print("SMOTEENN采样校验：成功")
        print(y_resampled.value_counts())
        return X_resampled, y_resampled
    else:
        print("SMOTEENN采样校验：失败")

"""'''

def underBalanceSample(X,y):
    bc = BalanceCascade(random_state=0,
                    estimator=LogisticRegression(random_state=0),
                    n_max_subset=10)
    bc.fit(X, y)
    X_resampled, y_resampled = bc.sample(X, y)
    print("BalanceCascade采样个数为：", X_resampled.shape[0])
    if(X_resampled.shape[0]==y_resampled.shape[0]):
        print("BalanceCascade采样校验：成功")
        print(sorted(Counter(y_resampled[0]).items()))
        return X_resampled, y_resampled
    else:
        print("BalanceCascade采样校验：失败")

"""

"""'''
X= pd.read_csv('data/data.csv')
y= pd.read_csv('data/labels.csv')

X=X.iloc[:,1:].fillna(0).drop(['Indcd'],axis=1)
y=y.iloc[:,1:]

#overSample(X,y)

#randomSample(X,y)

#SMOTEENNSample(X,y)



"""