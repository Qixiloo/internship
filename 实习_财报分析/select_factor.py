import pandas as pd
import numpy as np

import warnings


from imblearn.over_sampling import SMOTE



import json

warnings.filterwarnings('ignore')
import sklearn as sk
from sklearn.svm import SVR
from feature_selector import FeatureSelector
from sklearn.feature_selection import RFE
from pylab import mpl
    # 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["Arial Unicode MS"]

def select_1(data,labels):
    fs = FeatureSelector(data = data ,labels = labels)
    fs.identify_zero_importance(task = 'classification',
        eval_metric='auc',
        n_iterations=10,
        early_stopping=False)
    fs.identify_low_importance(0.98)
    fs.identify_single_unique()
    fs.identify_collinear(correlation_threshold = 0.98, one_hot=True)
    df_remove = fs.remove(methods =['zero_importance', 'single_unique','low_importance','collinear'],keep_one_hot=True)

    fs.plot_feature_importances(threshold = 0.98,plot_n = 30)
    fs.plot_collinear(plot_all=True)
    print("the features after selection1 are: ",df_remove.columns)
    return df_remove



def select_2(X,y):
    estimator = SVR(kernel="linear")

    selector = RFE(estimator, n_features_to_select=10, step=1)

    selector.fit(X, y)
    print(selector.support_)

    print(selector.ranking_)
    print("the features after selection2 are: ",X.columns[selector.support_])

    return X[X.columns[selector.support_]]
    
    