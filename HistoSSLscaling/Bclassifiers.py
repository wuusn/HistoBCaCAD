
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
def search_pipeline(X_train_data, y_train_data, model, param_grid, 
                    cv=10, scoring_fit=['neg_mean_squared_error', 'accuracy'],
                    search_mode = 'GridSearchCV', n_iterations = 0):
    fitted_model = None
    
    if(search_mode == 'GridSearchCV'):
        gs = GridSearchCV(estimator=model, 
                            param_grid=param_grid, 
                            cv=cv, 
                            n_jobs=-1, 
                            scoring=scoring_fit,
                            verbose=1, 
                            refit='acc'
                        )
        fitted_model = gs.fit(X_train_data, y_train_data)
    elif (search_mode == 'RandomizedSearchCV'):
        rs = RandomizedSearchCV(estimator=model,
                                param_distributions=param_grid, 
                                cv=cv,
                                n_iter=n_iterations,
                                n_jobs=-1, 
                                scoring=scoring_fit,
                                verbose=1
                            )
        fitted_model = rs.fit(X_train_data, y_train_data)

    if(fitted_model != None):
        # if do_probabilities:
        #     pred = fitted_model.predict_proba(X_test_data)
        # else:
        #     pred = fitted_model.predict(X_test_data)
        print('best model: ', fitted_model.best_params_ )
        return fitted_model



from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
class QDA_: 
    def __init__(self, X, y, params_dict=None):
        if params_dict is not None: 
            self.clf = QuadraticDiscriminantAnalysis(**params_dict)
        else: 
            self.clf = QuadraticDiscriminantAnalysis()
        self.clf.fit(X, y)
        self.X_train = X
        self.y_train = y

    def predict(self, X_test): 
        y_test = self.clf.predict(X_test)
        y_test_proba = self.clf.predict_proba(X_test)
        return y_test, y_test_proba 
    
    def search_params(self, X_test, params=None): 
        if params is None: 
            params = [{'reg_param': [0.1, 0.2, 0.3, 0.4, 0.5]}]
        # clf = GridSearchCV(self.clf, params, cv=4)
        # clf.fit(self.X_train, self.y_train )
        fitted_model = search_pipeline(self.X_train, self.y_train, self.clf, param_grid=params, 
                        cv=5, scoring_fit={'AUC': 'roc_auc', 'acc': 'accuracy'}, search_mode = 'GridSearchCV')   ##, 'accuracy'
        y_test = fitted_model.predict(X_test)
        y_test_proba = fitted_model.predict_proba(X_test)
        return y_test, y_test_proba, fitted_model.best_params_
    
    # def redefine(self, params_dict): 
    #     self.clf_redef = QuadraticDiscriminantAnalysis(**params_dict)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
class LDA_: 
    def __init__(self, X, y, params_dict=None):
        if params_dict is not None: 
            self.clf = LinearDiscriminantAnalysis(**params_dict)
        else: 
            self.clf = LinearDiscriminantAnalysis()
        self.clf.fit(X, y)
        self.X_train = X
        self.y_train = y
    def predict(self, X_test): 
        y_test = self.clf.predict(X_test)
        y_test_proba = self.clf.predict_proba(X_test)
        return y_test, y_test_proba 
    def search_params(self, X_test, params=None): 
        if params is None: 
            params = [{'solver': ['svd', 'lsqr', 'eigen'], 
                        'shrinkage': np.arange(0, 1, 0.01)}]
        # clf = GridSearchCV(self.clf, params, cv=4)
        # clf.fit(self.X_train, self.y_train )
        fitted_model = search_pipeline(self.X_train, self.y_train, self.clf, param_grid=params, 
                        cv=5, scoring_fit={'AUC': 'roc_auc', 'acc': 'accuracy'}, search_mode = 'GridSearchCV') 
        y_test = fitted_model.predict(X_test)
        y_test_proba = fitted_model.predict_proba(X_test)
        return y_test, y_test_proba, fitted_model.best_params_
# # define grid
# grid = dict()
# grid['solver'] = ['svd', 'lsqr', 'eigen']
# grid['shrinkage'] = arange(0, 1, 0.01)
# # define search
# search = GridSearchCV(model, grid, scoring='accuracy', cv=cv, n_jobs=-1)
# # perform the search
# results = search.fit(X, y)


class RandomForestC_: 
    def __init__(self, X, y, params_dict=None):
        if params_dict is not None: 
            self.clf = RandomForestClassifier(**params_dict)
        else: 
            self.clf = RandomForestClassifier(max_depth=5, n_estimators=10, n_jobs=10) 
        self.clf.fit(X, y) 
        self.X_train = X
        self.y_train = y
    def predict(self, X_test): 
        y_test = self.clf.predict(X_test) 
        y_test_proba = self.clf.predict_proba(X_test)
        return y_test, y_test_proba
    def search_params(self, X_test, params=None): 
        if params is None: 
            params = {
                'criterion':['gini','entropy'],
                'n_estimators':[5, 10, 15, 20, 30, 50,75,100],
                'max_features':['auto','sqrt','log2'],
                'class_weight':['balanced','balanced_subsample']
            }
        fitted_model = search_pipeline(self.X_train, self.y_train, self.clf, param_grid=params, 
                        cv=5, scoring_fit={'AUC': 'roc_auc', 'acc': 'accuracy'}, search_mode = 'GridSearchCV') 
        y_test = fitted_model.predict(X_test)
        y_test_proba = fitted_model.predict_proba(X_test)
        return y_test, y_test_proba, fitted_model.best_params_

from sklearn.linear_model import RidgeClassifier
class Ridge_: 
    def __init__(self, X, y): 
        self.clf = RidgeClassifier().fit(X, y) 
    def predict(self, X_test): 
        y_test = self.clf.predict(X_test) 
        y_test_proba = self.clf.predict_proba(X_test)
        return y_test, y_test_proba


from sklearn.linear_model import SGDClassifier 
from sklearn.calibration import CalibratedClassifierCV
class SGDClassifier_: 
    def __init__(self, X, y, params_dict=None):
        if params_dict is not None: 
            self.clf = SGDClassifier(**params_dict)
        else: 
            self.clf = SGDClassifier(max_iter=1000, tol=1e-3, n_jobs=10)
        self.clf.fit(X, y) 
        calibrator = CalibratedClassifierCV(self.clf, cv='prefit')
        self.model = calibrator.fit(X, y)    
        self.X_train = X
        self.y_train = y
    def predict(self, X_test): 
        y_test = self.clf.predict(X_test) 
        y_test_proba = self.model.predict_proba(X_test)
        return y_test, y_test_proba
    def search_params(self, X_test, params=None): 
        if params is None: 
            params = {
                "loss" : ["hinge", "log_loss", "log", "squared_hinge", "modified_huber"],
                "alpha" : [0.0001, 0.001, 0.01, 0.1],
                "penalty" : ["l2", "l1", "elasticnet"],
            }
        fitted_model = search_pipeline(self.X_train, self.y_train, self.clf, param_grid=params, 
                        cv=5, scoring_fit={'AUC': 'roc_auc', 'acc': 'accuracy'}, search_mode = 'GridSearchCV') 
        y_test = fitted_model.predict(X_test)
        calibrator = CalibratedClassifierCV(fitted_model, cv='prefit').fit(self.X_train, self.y_train)
        y_test_proba = calibrator.predict_proba(X_test)
        return y_test, y_test_proba, fitted_model.best_params_

from sklearn import tree
class DecisionTree_: 
    def __init__(self, X, y, params_dict=None):
        if params_dict is not None: 
            self.clf = tree.DecisionTreeClassifier(**params_dict)
        else: 
            self.clf = tree.DecisionTreeClassifier(max_depth=5)
        self.clf.fit(X, y) 
        self.X_train = X
        self.y_train = y
    def predict(self, X_test): 
        y_test = self.clf.predict(X_test) 
        # clf.score(X, y) 
        y_test_proba = self.clf.predict_proba(X_test)
        return y_test, y_test_proba
    def search_params(self, X_test, params=None): 
        if params is None: 
            params = {'criterion':['gini','entropy'], 
                        'max_features': ['auto', 'sqrt', 'log2'],
                        'ccp_alpha': [0.1, .01, .001],
                        'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150]}
        fitted_model = search_pipeline(self.X_train, self.y_train, self.clf, param_grid=params, 
                        cv=5, scoring_fit={'AUC': 'roc_auc', 'acc': 'accuracy'}, search_mode = 'GridSearchCV') 
        y_test = fitted_model.predict(X_test)
        y_test_proba = fitted_model.predict_proba(X_test)
        return y_test, y_test_proba, fitted_model.best_params_


from sklearn.neighbors import KNeighborsClassifier
class KNeighbors_: 
    def __init__(self, X, y, params_dict=None):
        if params_dict is not None: 
            self.clf = KNeighborsClassifier(**params_dict)
        else: 
            self.clf = KNeighborsClassifier(n_neighbors=3, n_jobs=10)
        self.clf.fit(X, y) 
        self.X_train = X
        self.y_train = y
    def predict(self, X_test): 
        y_test = self.clf.predict(X_test) 
        # clf.score(X, y) 
        y_test_proba = self.clf.predict_proba(X_test)
        return y_test, y_test_proba 
    def search_params(self, X_test, params=None, search_mode = 'GridSearchCV'): 
        if params is None: 
            params = {'n_neighbors': np.arange(1, 31 ), 
                        'weights': ['uniform', 'distance'], 
                        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 
                        'leaf_size': [5, 10, 15, 20, 25, 30 ], 
                        'p': [1, 2]}
        fitted_model = search_pipeline(self.X_train, self.y_train, self.clf, param_grid=params, 
                        cv=5, scoring_fit={'AUC': 'roc_auc', 'acc': 'accuracy'}, search_mode = search_mode) 
        y_test = fitted_model.predict(X_test)
        y_test_proba = fitted_model.predict_proba(X_test)
        # print (grid.best_score_)
        # print (grid.best_params_)
        # print (grid.best_estimator_)
        return y_test, y_test_proba, fitted_model.best_params_

# from sklearn.svm import SVC
# class SVC_linear_: 
#     def __init__(self, X, y): 
#         self.clf = SVC(kernel="linear", C=1, probability=True ).fit(X, y) 
#     def predict(self, X_test): 
#         y_test = self.clf.predict(X_test) 
#         # clf.score(X, y) 
#         y_test_proba = self.clf.predict_proba(X_test)
#         return y_test, y_test_proba  
### too slow ^1

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
class LinearSVC_: 
    def __init__(self, X, y, params_dict=None):
        if params_dict is not None: 
            self.clf = LinearSVC(**params_dict)
        else: 
            self.clf = LinearSVC( )
        self.clf.fit(X, y) 
        self.clf2 = CalibratedClassifierCV(self.clf).fit(X, y )
        self.X_train = X
        self.y_train = y

    def predict(self, X_test): 
        y_test = self.clf.predict(X_test) 
        # clf.score(X, y) 
        y_test_proba = self.clf2.predict_proba(X_test)
        return y_test, y_test_proba 
    def search_params(self, X_test, params=None, search_mode = 'GridSearchCV'): 
        if params is None: 
            params = {'penalty': ['l1', 'l2'], 
                        'loss': ['hinge', 'squared_hinge'] , 
                        'C': [0.01, 0.1, 1, 10, 100] }
        fitted_model = search_pipeline(self.X_train, self.y_train, self.clf, param_grid=params, 
                        cv=5, scoring_fit={'AUC': 'roc_auc', 'acc': 'accuracy'}, search_mode = search_mode) 
        y_test = fitted_model.predict(X_test)
        # y_test_proba = fitted_model.predict_proba(X_test)
        calibrator = CalibratedClassifierCV(fitted_model, cv='prefit').fit(self.X_train, self.y_train)
        y_test_proba = calibrator.predict_proba(X_test)
        # print (grid.best_score_)
        # print (grid.best_params_)
        # print (grid.best_estimator_)
        return y_test, y_test_proba, fitted_model.best_params_


from sklearn.svm import SVC 
from sklearn.model_selection import GridSearchCV
class SVC_rbf_(): 
    def __init__(self, X, y, params_dict=None):
        if params_dict is not None: 
            self.clf = SVC(probability=True, **params_dict)
        else: 
            self.clf = SVC(kernel='rbf', probability=True)
        self.clf.fit(X, y) 
        self.X_train = X
        self.y_train = y
    def predict(self, X_test): 
        y_test = self.clf.predict(X_test) 
        # clf.score(X, y) 
        y_test_proba = self.clf.predict_proba(X_test)
        return y_test, y_test_proba 

    def search_params(self, X_test, params=None, search_mode = 'GridSearchCV'): 
        if params is None: 
            params = {'C': [1, 10, 100], 
                        # "kernel": ['poly', 'rbf','sigmoid'], 
                        "kernel": ['rbf'], 
                        'gamma': [1,0.1,0.01,0.001]
                        } 
        fitted_model = search_pipeline(self.X_train, self.y_train, self.clf, param_grid=params, 
                        cv=5, scoring_fit={'AUC': 'roc_auc', 'acc': 'accuracy'}, search_mode = search_mode) 
        y_test = fitted_model.predict(X_test)
        y_test_proba = fitted_model.predict_proba(X_test)
        # print (grid.best_score_) 
        # print (grid.best_params_) 
        # print (grid.best_estimator_) 
        return y_test, y_test_proba, fitted_model.best_params_



from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern, RationalQuadratic, WhiteKernel
class GaussianProcess_: 
    def __init__(self, X, y, params_dict=None):
        if params_dict is not None: 
            self.clf = GaussianProcessClassifier(**params_dict)
        else: 
            kernel = 1.0 * RBF(1.0)
            self.clf = GaussianProcessClassifier(kernel=kernel, n_jobs=10)
        self.clf.fit(X, y)
        self.X_train = X
        self.y_train = y

    def predict(self, X_test): 
        y_test = self.clf.predict(X_test) 
        # clf.score(X, y) 
        y_test_proba = self.clf.predict_proba(X_test)
        return y_test, y_test_proba 

    def search_params(self, X_test, params=None, search_mode = 'GridSearchCV'): 
        if params is None: 
            params = {
                        "kernel": [1*RBF(), 1*DotProduct(), 1*Matern(), 1*WhiteKernel()], 
                      } 
        fitted_model = search_pipeline(self.X_train, self.y_train, self.clf, param_grid=params, 
                        cv=5, scoring_fit={'AUC': 'roc_auc', 'acc': 'accuracy'}, search_mode = search_mode) 
        y_test = fitted_model.predict(X_test)
        y_test_proba = fitted_model.predict_proba(X_test)
        # print (grid.best_score_) 
        # print (grid.best_params_) 
        # print (grid.best_estimator_) 
        return y_test, y_test_proba, fitted_model.best_params_


from sklearn.neural_network import MLPClassifier
class MLPClassifier_: 
    def __init__(self, X, y, params_dict=None):
        if params_dict is not None: 
            self.clf = MLPClassifier(**params_dict)
        else: 
            self.clf = MLPClassifier(max_iter=1000)
        self.clf.fit(X, y)
        self.X_train = X
        self.y_train = y

    def predict(self, X_test): 
        y_test = self.clf.predict(X_test) 
        # clf.score(X, y) 
        y_test_proba = self.clf.predict_proba(X_test)
        return y_test, y_test_proba 

    def search_params(self, X_test, params=None, search_mode = 'GridSearchCV'): 
        if params is None: 
            params = {'solver': ['lbfgs'], 
                        'max_iter': [1000, 1500, 1800 ], 
                        'alpha': 10.0 ** -np.arange(1, 4), 
                        'hidden_layer_sizes':np.arange(10, 15), 
                        # 'random_state':[0,1,2,3]
                        } 
        fitted_model = search_pipeline(self.X_train, self.y_train, self.clf, param_grid=params, 
                        cv=5, scoring_fit={'AUC': 'roc_auc', 'acc': 'accuracy'}, search_mode = search_mode) 
        y_test = fitted_model.predict(X_test)
        y_test_proba = fitted_model.predict_proba(X_test)
        # print (grid.best_score_) 
        # print (grid.best_params_) 
        # print (grid.best_estimator_) 
        return y_test, y_test_proba, fitted_model.best_params_



from sklearn.ensemble import AdaBoostClassifier
class AdaBoostClassifier_: 
    def __init__(self, X, y, params_dict=None):
        if params_dict is not None: 
            self.clf = AdaBoostClassifier(**params_dict)
        else: 
            self.clf = AdaBoostClassifier( )
        self.clf.fit(X, y)
        self.X_train = X
        self.y_train = y

    def predict(self, X_test): 
        y_test = self.clf.predict(X_test) 
        # clf.score(X, y) 
        y_test_proba = self.clf.predict_proba(X_test)
        return y_test, y_test_proba 

    def search_params(self, X_test, params=None, search_mode = 'GridSearchCV'): 
        if params is None: 
            params = {
                        # 'base_estimator__max_depth': [i for i in range(2,11,2)],
                        # 'base_estimator__min_samples_leaf':[5, 10], 
                        'n_estimators':[10, 50, 100, 250, 500],
                        'learning_rate':[0.001, 0.01, 0.1, 1]
                    }

        fitted_model = search_pipeline(self.X_train, self.y_train, self.clf, param_grid=params, 
                        cv=5, scoring_fit={'AUC': 'roc_auc', 'acc': 'accuracy'}, search_mode = search_mode) 
        y_test = fitted_model.predict(X_test)
        y_test_proba = fitted_model.predict_proba(X_test)
        # print (grid.best_score_) 
        # print (grid.best_params_) 
        # print (grid.best_estimator_) 
        return y_test, y_test_proba, fitted_model.best_params_

from sklearn.naive_bayes import GaussianNB
class GaussianNB_: 
    def __init__(self, X, y, params_dict=None):
        if params_dict is not None: 
            self.clf = GaussianNB(**params_dict)
        else: 
            self.clf = GaussianNB()
        self.clf.fit(X, y)
        self.X_train = X
        self.y_train = y

    def predict(self, X_test): 
        y_test = self.clf.predict(X_test) 
        # clf.score(X, y) 
        y_test_proba = self.clf.predict_proba(X_test)
        return y_test, y_test_proba 

    def search_params(self, X_test, params=None, search_mode = 'GridSearchCV'): 
        if params is None: 
            params = {
                    'var_smoothing': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15]
                    }

        fitted_model = search_pipeline(self.X_train, self.y_train, self.clf, param_grid=params, 
                        cv=5, scoring_fit={'AUC': 'roc_auc', 'acc': 'accuracy'}, search_mode = search_mode) 
        y_test = fitted_model.predict(X_test)
        y_test_proba = fitted_model.predict_proba(X_test)
        # print (grid.best_score_) 
        # print (grid.best_params_) 
        # print (grid.best_estimator_) 
        return y_test, y_test_proba, fitted_model.best_params_

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
def ensemble_(X, y): 
    clf1 = LogisticRegression()
    clf2 = RandomForestClassifier( )
    clf3 = GaussianNB()
    eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard', )  ##weights=[2,1,1]
    eclf1 = eclf1.fit(X, y)
    print(eclf1.predict(X)) 