import numpy as np
import pandas as pd

from sklearn.feature_selection import chi2 
from sklearn.feature_selection import SelectKBest, SelectPercentile
def chi_select(X, y, k=None, percentile=None): 
    ## Univariate feature selection
    ### only for non-negative variables
    if k:
        percentile=None 
        #使用选择器的SelectKBest函数，使用chi2模型（卡方验证）保留关联最大的300个特征
        X_chi = SelectKBest(chi2, k=300).fit_transform(X, y)
    if percentile: 
        # 除了选择前K个特征，也可以设置选择百分比例SelectPercentile()
        X_chi = SelectPercentile(chi2, percentile=10).fit_transform(X, y)
    # chi2函数也可直接返回特征与目标之间的卡方值和对应的p值
    # chi, p = chi2(X_fsvar,y)
    # 根据各个特征计算出的p值，以0.05或者0.1为阈值过滤相应的特征
    return X_chi 


###bmethod 1
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif, f_regression
def UnivariateFeatureSelection(X, y, n_features=10): 
    ''' 
    Compute the ANOVA F-value for the provided sample.
    ''' 
    X_train, y_train = X, y
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    univariate = f_classif(X_train, y_train)
    ## Capture P values in a series
    univariate = pd.Series(univariate[1])
    univariate.index = X_train.columns
    univariate.sort_values(ascending=False, inplace=True)
    # ## Plot the P values
    # univariate.sort_values(ascending=False).plot.bar(figsize=(20,8))
    ## Select K best Features
    k_best_features = SelectKBest(f_classif, k=n_features).fit(X_train.fillna(0), y_train) 
    # k_best_features = SelectPercentile(f_classif, percentile=10).fit(X_train.fillna(0), y_train)
    # print( X_train.columns[k_best_features.get_support()] )
    # selected_features = X_train.columns[k_best_features.get_support()]
    # # print( X_train.shape ) 
    # X_train = k_best_features.transform(X_train.fillna(0))

    feats_score = k_best_features.scores_ 
    feats_score_sort_large_small = feats_score.argsort()[-n_features:][::-1] 

    selected_features_sort_large_small = X_train.columns[feats_score_sort_large_small]
    X_train_np = X_train[selected_features_sort_large_small].fillna(0).to_numpy()
    return X_train_np, selected_features_sort_large_small 

##bmethod 2
from sklearn.feature_selection import mutual_info_classif
def mutual_info_selection(X, y, n_features=10): 
    # Calculate Mutual Information between each feature and the target

    X_train, y_train = X, y
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    mutual_info = mutual_info_classif(X_train.fillna(0), y_train, discrete_features=False)
    ## Create Feature Target Mutual Information Series
    mi_series = pd.Series(mutual_info)
    mi_series.index = X_train.columns
    mi_series.sort_values(ascending=False) 
    # mi_series.sort_values(ascending=False).plot.bar(figsize=(20,8))
    # # Select K best features
    k_best_features = SelectKBest(mutual_info_classif, k=n_features).fit(X_train.fillna(0), y_train) 
    # k_best_features = SelectPercentile(mutual_info_classif, percentile=10).fit(X_train.fillna(0), y_train)
    # print( X_train.columns[k_best_features.get_support()] )
    # selected_features = X_train.columns[k_best_features.get_support()]
    # X_train = k_best_features.transform(X_train.fillna(0)) 

    feats_score = k_best_features.scores_ 
    feats_score_sort_large_small = feats_score.argsort()[-n_features:][::-1] 

    selected_features_sort_large_small = X_train.columns[feats_score_sort_large_small]
    X_train_np = X_train[selected_features_sort_large_small].fillna(0).to_numpy()
    return X_train_np, selected_features_sort_large_small 


###bmethod 3
from sklearn.feature_selection import VarianceThreshold
def varianceThreshold_rm(X, var_thresh=0): 
    ## method1
    selector = VarianceThreshold(threshold = var_thresh)#设置方差过滤阈值为0
    selector.fit_transform(X)
    aa_name = selector.get_support()#get_support函数返回方差＞阈值的布尔值序列
    selected_features = selector.get_feature_names_out()
    aa_select = X.iloc[:,aa_name]#根据布尔值序列取出各个特征
    # ## method2 
    # var = feature_frame.var()#求每列方差
    # var_thresh = var_thresh #设置阈值
    # var_select = var[var>var_thresh]#根据条件返回序列值，取大于阈值
    # aa_select = feature_frame[var_select.index]#取出特征
    return aa_select, selected_features 

import mrmr    ###pip install mrmr_selection
from mrmr import mrmr_classif
def mrmr_selection(X, y, n_features=10): 
    # select top 10 features using mRMR
    X_train, y_train = X, y
    selected_features, feats_score, _ = mrmr_classif(X=X_train, y=y_train, K=n_features, n_jobs=10, return_scores=True, show_progress=False) 
    # X = X[selected_features] 
    # X = X.to_numpy() 
    feats_score = feats_score.to_numpy()
    feats_score_sort_large_small = feats_score.argsort()[-n_features:][::-1] 

    selected_features_sort_large_small = X_train.columns[feats_score_sort_large_small]
    X_train_np = X_train[selected_features_sort_large_small].fillna(0).to_numpy()
    return X_train_np, selected_features_sort_large_small

from scipy import stats
def ttest_selection(X, y, n_features=10, sig=0.05 ): 
    feats_name = X.columns.tolist()
    X_train = X.to_numpy()
    y_train = np.squeeze( y )     
    X_train_0 = X_train[y_train==0,:]
    X_train_1 = X_train[y_train==1,:]
    pv_list = []
    for i in range(X_train_0.shape[1]): 
        t_stat, pv = stats.ttest_ind(X_train_0[:, i], X_train_1[:, i] ) 
        pv_list.append(pv ) 
    indices = np.argsort(pv_list).tolist()
    sorted_pv_list = [pv_list[index] for index in indices] 
    sorted_feats_name = [feats_name[index] for index in indices] 
    selected_features = sorted_feats_name[:n_features] 
    X_selected = X[selected_features]
    X_selected = X_selected.to_numpy()
    return X_selected, selected_features ## sorted_pv_list, sorted_feats_name  

from scipy import stats
##Wilcoxon rank-sum statistic
def ranksums_selection(X, y, n_features=10, sig=0.05 ): 
    feats_name = X.columns.tolist()
    X_train = X.to_numpy()
    y_train = np.squeeze( y )     
    X_train_0 = X_train[y_train==0,:]
    X_train_1 = X_train[y_train==1,:]
    pv_list = []
    for i in range(X_train_0.shape[1]): 
        stat, pv = stats.ranksums(X_train_0[:, i], X_train_1[:, i] ) 
        pv_list.append(pv ) 
    indices = np.argsort(pv_list).tolist()
    sorted_pv_list = [pv_list[index] for index in indices] 
    sorted_feats_name = [feats_name[index] for index in indices] 
    selected_features = sorted_feats_name[:n_features] 
    X_selected = X[selected_features]
    X_selected = X_selected.to_numpy()
    return X_selected, selected_features ## sorted_pv_list, sorted_feats_name  

from sklearn.feature_selection import RFE, RFECV
from sklearn.svm import SVR, SVC
from sklearn.model_selection import StratifiedKFold
def rfe_selection(X, y, n_features=50):
    # estimator = SVR(kernal='linear')  ##need to change
    estimator = SVC(kernal='linear')  ##need to change
    selector = RFE(estimator, n_features_to_select = n_features, step=1) 
    # selector = RFECV( estimator=estimator, step=1, cv=StratifiedKFold(2), scoring="accuracy", min_features_to_select=1, )
    selector = selector.fit(X, y)
    selected_features = selector.support_
    selected_ranking = selector.ranking_
    # print("Optimal number of features : %d" % selector.n_features_)  ###FOR RFECV
    X_selected = X.iloc[:,selected_features] 
    return X_selected

from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC
import matplotlib.pyplot as plt
def SelectFromModel_(X, y):
    RFC_ = RFC(n_estimators=10)
    ##### print(X_embedded.shape)
    #模型的维度明显被降低了
    #画学习曲线来找最佳阈值
    
    RFC_.fit(X,y).feature_importances_
    threshold = np.linspace(0,(RFC_.fit(X,y).feature_importances_).max(),20)
    score = []
    for i in threshold:
        X_embedded = SelectFromModel(RFC_,threshold=i).fit_transform(X,y)
        once = cross_val_score(RFC_,X_embedded,y,cv=5).mean()
        score.append(once)
    plt.plot(threshold,score)
    plt.show()
    X_embedded = SelectFromModel(RFC_,threshold=0.00067).fit_transform(X,y)
    X_embedded.shape
    print(X_embedded.shape)
    print(cross_val_score(RFC_,X_embedded,y,cv=5).mean()) 
    # from sklearn.svm import LinearSVC
    # from sklearn.datasets import load_iris
    # from sklearn.feature_selection import SelectFromModel
    # lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
    # model = SelectFromModel(lsvc, prefit=True)
    # X_new = model.transform(X)
    # X_new.shape 
