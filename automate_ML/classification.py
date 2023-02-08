import pandas as pd
import numpy as np
import warnings
import json
from preprocessing import Preprocessing, train_test_data, timenow


""""                                Classification Models                                   """

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier#, RidgeClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import ExtraTreeClassifier
from sklearn.neural_network import MLPClassifier


"""                                 Pre-processing tools                                     """

from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV

"""                                     Metrices                                            """

from sklearn import metrics
from sklearn.metrics import roc_curve, confusion_matrix#, roc_auc_score, auc

"""                                   Plotting tools                                        """

from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from tabulate import tabulate

global clf, clf_without_opt, _params, _model_name, clf_cv, df, _prediction_data, probability, unknown_prediction, cf_matrix, \
    X_train, X_test, y_train, y_test, Test_score, Train_score, y_train_predicted, y_test_predicted, _optimization, \
    test_prediction_score, train_prediction_score, y_train_proba_predicted, y_test_proba_predicted, best_params, \
    corr_fig, fpr, tpr, imp_fig, _scoring, Training_score, df_unknown, save_content

"""""""""""""""""""""""""""""""""""""""""""""""""Classifiers"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class Classification(Preprocessing):


    @staticmethod
    def model_names():
        model_names = ['AdaBoost', 'Bagging', 'CalibratedCV', 'CatBoost', 'DecisionTree', 'ExtraTrees', 'ExtraTree',
                       'GradientBoosting', 'KNeighbors', 'LogisticReg', 'LinearDA', 'LGBM', 'Linear_SVC', 'Mlp',
                       'NuSVC', 'RandomForest', 'RadiusNeighbor', 'Ridge', 'SVC', 'SGDC']

        return model_names


    @staticmethod
    def Model(model_name=None, params=None, proba=False, random_state=None):
        global clf, clf_without_opt, _params, _model_name

        _model_names = ['AdaBoost', 'Bagging', 'CalibratedCV', 'CatBoost', 'DecisionTree', 'ExtraTrees', 'ExtraTree',
                       'GradientBoosting', 'KNeighbors', 'LogisticReg', 'LinearDA', 'LGBM', 'Linear_SVC', 'Mlp',
                       'NuSVC', 'RandomForest', 'RadiusNeighbor', 'Ridge', 'SVC', 'SGDC']


        if model_name not in _model_names:
            model_name_error = "{0} is not a valid model_name or not available. " \
                               "Use from following: {1}".format(model_name, _model_names)

            raise NameError(model_name_error)
        else:
            pass

        print('\n', '\033[0;30;44m' + '         Solving Classification Problem        ' + '\033[0m', '\n')

        if model_name == 'AdaBoost':
            clf = AdaBoostClassifier()
            print('\033[1;32;40m' + "---> Classification model = 'AdaBoostClassifier'" + '\033[0m', '\n')
            _model_name = "Classification model = 'AdaBoostClassifier'"
            if params is None:
                _params = {
                    'n_estimators': list(range(50, 1000)),
                    'learning_rate': list(np.linspace(0.001, 0.1, num=30)),
                    'random_state': [random_state]
                           }
            else:
                _params = params
                clf_without_opt = AdaBoostClassifier(**params)


        if model_name == 'Bagging':
            clf = BaggingClassifier()
            print('\033[1;32;40m' + "---> Classification model = 'BaggingClassifier'" + '\033[0m', '\n')
            _model_name = "Classification model = 'BaggingClassifier'"
            if params is None:
                _params = {
                    'n_estimators': list(range(50, 1000)),
                    'max_samples': list(np.linspace(0.1, 1.0, num=30)),
                    'random_state': [random_state]
                            }
            else:
                _params = params
                clf_without_opt = BaggingClassifier(**params)


        if model_name == 'CalibratedCV':
            clf = CalibratedClassifierCV()
            print('\033[1;32;40m' + "---> Classification model = 'CalibratedClassifierCV'" + '\033[0m', '\n')
            _model_name = "Classification model = 'CalibratedClassifierCV'"
            if params is None:
                _params = {
                        'method': ['sigmoid', 'isotonic'],
                           'cv': [3, 4, 5]
                         }
            else:
                _params = params
                clf_without_opt = CalibratedClassifierCV(**params)


        if model_name == 'CatBoost':
            clf = CatBoostClassifier()
            print('\033[1;32;40m' + "---> Classification model = 'CatBoostClassifier'" + '\033[0m', '\n')
            _model_name = "Classification model = 'CatBoostClassifier'"
            if params is None:
                _params = {
                    'iterations': list(range(50, 500)),
                   'learning_rate': list(np.linspace(0.001, 0.1, num=30)),
                   'border_count': list(range(10, 200)),
                   'feature_border_type': ['GreedyLogSum'],
                   'random_state': [random_state],
                   'verbose': [0]
                           }
            else:
                _params = params
                clf_without_opt = CatBoostClassifier(**params)


        if model_name == 'DecisionTree':
            clf = DecisionTreeClassifier()
            print('\033[1;32;40m' + "---> Classification model = 'DecisionTreeClassifier'" + '\033[0m', '\n')
            _model_name = "Classification model = 'DecisionTreeClassifier'"
            if params is None:
                _params = {
                    'max_depth': list(range(2, 11)),
                    'min_samples_split': list(range(1, 11)),
                    'min_samples_leaf': list(range(1, 11)),
                    'random_state': [random_state]
                            }
            else:
                _params = params
                clf_without_opt = DecisionTreeClassifier(**params)


        if model_name == 'ExtraTrees':
            clf = ExtraTreesClassifier()
            print('\033[1;32;40m' + "---> Classification model = 'ExtraTreesClassifier'" + '\033[0m', '\n')
            _model_name = "Classification model = 'ExtraTreesClassifier'"
            if params is None:
                _params = {
                    'n_estimators': list(range(50, 500)),
                    'max_depth': list(range(2, 11)),
                    'min_samples_split': list(range(0, 11)),
                    'min_samples_leaf': list(range(0, 11)),
                    'random_state': [random_state]
                            }
            else:
                _params = params
                clf_without_opt = ExtraTreesClassifier(**params)


        if model_name == 'ExtraTree':
            clf = ExtraTreeClassifier()
            print('\033[1;32;40m' + "---> Classification model = 'ExtraTreeClassifier'" + '\033[0m', '\n')
            _model_name = "Classification model = 'ExtraTreeClassifier'"
            if params is None:
                _params = {
                    'max_depth': list(range(2, 11)),
                   'min_samples_split': list(range(0, 11)),
                   'min_samples_leaf': list(range(0, 11)),
                   'random_state': [random_state]
                           }
            else:
                _params = params
                clf_without_opt = ExtraTreeClassifier(**params)


        if model_name == 'GradientBoosting':
            clf = GradientBoostingClassifier()
            print('\033[1;32;40m' + "---> Classification model = 'GradientBoostingClassifier'" + '\033[0m', '\n')
            _model_name = "Classification model = 'GradientBoostingClassifier'"
            if params is None:
                _params = {
                    'learning_rate': list(np.linspace(0.001, 0.1, num=30)),
                    'max_depth': list(range(2, 11)),
                    'n_estimators': list(range(50, 500)),
                    'min_samples_split': list(range(0, 11)),
                    'random_state': [random_state]
                            }
            else:
                _params = params
                clf_without_opt = GradientBoostingClassifier(**params)


        if model_name == 'KNeighbors':
            clf = KNeighborsClassifier()
            print('\033[1;32;40m' + "---> Classification model = 'KNeighborsClassifier'" + '\033[0m', '\n')
            _model_name = "Classification model = 'KNeighborsClassifier'"
            if params is None:
                _params = {
                    'n_neighbors': list(range(2, 10)),
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2]
                            }
            else:
                _params = params
                clf_without_opt = KNeighborsClassifier(**params)


        if model_name == 'LogisticReg':
            clf = LogisticRegression()
            print('\033[1;32;40m' + "---> Classification model = 'LogisticRegression'" + '\033[0m', '\n')
            _model_name = "Classification model = 'LogisticRegression'"
            if params is None:
                _params = {
                    'C': list(np.linspace(0.1, 10.0, num=30)),
                    'class_weight': ['balanced'],
                    'max_iter': list(range(50, 500)),
                    'random_state': [random_state],
                    'solver': ['lbfgs', 'liblinear']
                            }
            else:
                _params = params
                clf_without_opt = LogisticRegression(**params)


        if model_name == 'LinearDA':
            clf = LinearDiscriminantAnalysis()
            print('\033[1;32;40m' + "---> Classification model = 'LinearDiscriminantAnalysis'" + '\033[0m', '\n')
            _model_name = "Classification model = 'LinearDiscriminantAnalysis'"
            if params is None:
                _params = {
                    'solver': ['svd', 'lsqr', 'eigen']
                            }
            else:
                _params = params
                clf_without_opt = LinearDiscriminantAnalysis(**params)


        if model_name == 'LGBM':
            clf = LGBMClassifier()
            print('\033[1;32;40m' + "---> Classification model = 'LGBMClassifier'" + '\033[0m', '\n')
            _model_name = "Classification model = 'LGBMClassifier'"
            if params is None:
                _params = {
                    'boosting_type': ['gbdt'],
                    'num_leaves': list(range(100, 1000)),
                    'learning_rate': list(np.linspace(0.001, 0.1, num=30)),
                    'n_estimators': list(range(50, 1000)),
                    'random_state': [random_state]
                         }
            else:
                _params = params
                clf_without_opt = LGBMClassifier(**params)


        if model_name == 'Linear_SVC':
            clf = LinearSVC()
            print('\033[1;32;40m' + "---> Classification model = 'LinearSVC'" + '\033[0m', '\n')
            _model_name = "Classification model = 'LinearSVC'"
            if params is None:
                _params = {
                    'penalty': ['l2'],
                    'dual': [False],
                    'C': list(np.linspace(0.1, 10.0, num=30)),
                    'class_weight': ['balanced'],
                    'max_iter': list(range(50, 500)),
                    'random_state': [random_state]
                            }
            else:
                _params = params
                clf_without_opt = LinearSVC(**params)


        if model_name == 'Mlp':
            clf = MLPClassifier()
            print('\033[1;32;40m' + "---> Classification model = 'Multi-layer Perceptron'" + '\033[0m', '\n')
            _model_name = "Classification model = 'MLPClassifier'"
            if params is None:
                _params = {
                    'hidden_layer_sizes': [4, 8],
                    'learning_rate_init': list(np.linspace(0.001, 0.1, num=30)),
                    'max_iter': [250],
                    'activation': ['relu', 'identity'],
                    'batch_size': ['auto'],
                    'solver': ['adam', 'lbfgs', 'sgd'],
                    'random_state': [random_state]
                            }
            else:
                _params = dict(params)
                clf_without_opt = MLPClassifier(**params)


        if model_name == 'NuSVC':
            clf = NuSVC()
            print('\033[1;32;40m' + "---> Classification model = 'NuSVC'" + '\033[0m', '\n')
            _model_name = "Classification model = 'NuSVC'"
            if params is None:
                _params = {
                    'kernel': ['rbf', 'poly'],
                    'gamma': ['auto', 'scale'],
                    'probability': [proba],
                    'random_state': [random_state]
                            }
            else:
                _params = params
                clf_without_opt = NuSVC(**params)


        if model_name == 'RandomForest':
            clf = RandomForestClassifier()
            print('\033[1;32;40m' + "---> Classification model = 'RandomForestClassifier'" + '\033[0m', '\n')
            _model_name = "Classification model = 'RandomForestClassifier'"
            if params is None:
                _params = {
                    'n_estimators': list(range(50, 1000)),
                    'max_depth': list(range(2, 11)),
                    'min_samples_split': list(range(0, 11)),
                    'min_samples_leaf': list(range(0, 11)),
                    'random_state': [random_state]
                        }
            else:
                _params = params
                clf_without_opt = RandomForestClassifier(**params)


        if model_name == 'RadiusNeighbor':
            clf = RadiusNeighborsClassifier()
            print('\033[1;32;40m' + "---> Classification model = 'RadiusNeighborsClassifier'" + '\033[0m', '\n')
            _model_name = "Classification model = 'RadiusNeighborsClassifier'"
            if params is None:
                _params = {
                    'radius': list(np.linspace(1.0, 5.0, num=20)),
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2]
                            }
            else:
                _params = params
                clf_without_opt = RadiusNeighborsClassifier(**params)


        if model_name == 'Ridge':
            clf = RidgeClassifier()
            print('\033[1;32;40m' + "---> Classification model = 'RidgeClassifier'" + '\033[0m', '\n')
            _model_name = "Classification model = 'RidgeClassifier'"
            if params is None:
                _params = {
                    'alpha': list(np.linspace(0.1, 10.0, num=20)),
                    'max_iter': list(range(50, 1000)),
                    'class_weight': ['balanced'],
                    'solver': ['auto'],
                    'random_state': [random_state]
                         }
            else:
                _params = params
                clf_without_opt = RidgeClassifier(**params)


        if model_name == 'SVC':
            clf = SVC()
            print('\033[1;32;40m' + "---> Classification model = 'SVC'" + '\033[0m', '\n')
            _model_name = "Classification model = 'SVC'"
            if params is None:
                _params = {
                    'gamma': list(np.linspace(0.01, 10.0, num=30)),
                    'C': list(np.linspace(0.1, 10.0, num=30)),
                    'class_weight':['balanced'],
                    'probability':[proba],
                    'random_state':[random_state]
                            }
            else:
                _params = params
                clf_without_opt = SVC(**params)


        if model_name == 'SGDC':
            clf = SGDClassifier()
            print('\033[1;32;40m' + "---> Classification model = 'SGDClassifier'" + '\033[0m', '\n')
            _model_name = "Classification model = 'SGDClassifier'"
            if params is None:
                _params = {
                    'loss': ['log', 'modified_huber'],
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'max_iter': list(range(50, 1000)),
                    'class_weight': ['balanced'],
                    'random_state': [random_state]
                        }
            else:
                _params = params
                clf_without_opt = SGDClassifier(**params)

        """
        Parameters:
        
        model_name          = str		: 	Name of the machine learning algorithm to use (default = None)
        params              = dict		:	Set of parameters for machine learning algrorithm (default = None)
        proba               = bool		: 	Only needed for 'NuSVC' and 'SVC' when predicting probability (default = False)
        random_state        = int		:	Random number for reproducing the results (default = None)
        
        """


##############################################******Fit Function********################################################

    def fit(self, optimization='Bayesian', num_iter=20, cv=10, scoring='roc_auc'):
        global  X_train, X_test, y_train, y_test, clf_cv, Train_score, y_train_predicted, y_test_predicted, \
            Training_score, Test_score, best_params, _scoring, _optimization

        scoring_methods = ['roc_auc', 'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted','roc_auc_ovo_weighted',
                           'accuracy', 'f1', 'f1_macro', 'f1_micro', 'f1_weighted', 'recall', 'recall_macro',
                           'recall_micro', 'recall_weighted', 'precision', 'precision_macro', 'precision_micro',
                           'precision_weighted']

        multiclass_scoring_methods = ['accuracy', 'f1_macro', 'f1_micro', 'f1_weighted',
                                      'recall_macro', 'recall_micro', 'recall_weighted', 'precision_macro',
                                      'precision_micro', 'precision_weighted']

        if self.multiclass and scoring not in multiclass_scoring_methods:
            raise NameError("'{0}' metrics cannot be used for evaluating multiclass problems. "
                            "Use a metrics from the following list {1}".format(scoring, multiclass_scoring_methods))
        else:
            pass

        if scoring in scoring_methods:
            _scoring = scoring
        else:
            score_error = "{0} is not a valid scoring method. Valid scoring methods are following: {1} "\
                .format(scoring, scoring_methods)
            raise NameError(score_error)

        _optimization = optimization
        if optimization == 'Grid':
            clf_cv = GridSearchCV(clf, _params, cv=cv, scoring=scoring, error_score='raise')
        elif optimization == 'Randomized':
            clf_cv = RandomizedSearchCV(clf, _params, n_iter=num_iter, cv=cv, scoring=scoring,
                                        random_state=self.random_state, error_score='raise')
        elif optimization == 'Bayesian':
            clf_cv = BayesSearchCV(clf, _params, scoring=scoring, n_iter=num_iter, cv=cv,
                                   random_state=self.random_state, error_score='raise')
        elif optimization is None:
            clf_cv = clf_without_opt
        else:
            search_error = "{} is not a valid option for hyper-paramteres optimization. Available options are 'Grid', " \
                           "'Randomized' or 'Bayesian'".format(optimization)
            raise ValueError(search_error)

        X_train, X_test, y_train, y_test = train_test_data()

        clf_cv.fit(X_train, y_train)
        y_train_predicted = clf_cv.predict(X_train)
        y_test_predicted = clf_cv.predict(X_test)

        if optimization is None and scoring in scoring_methods:
            pass
        else:
            print("------------------------------------------------------------------------------------------------")
            best_params = clf_cv.best_params_
            best_estimator = clf_cv.best_estimator_
            best_score = clf_cv.best_score_
            print('Best estimator:', best_estimator)
            print('Best parameters:', best_params)
            print('Best score:', round(best_score, 6))
            print("------------------------------------------------------------------------------------------------\n")

        print('\033[1;32;40m' + "**************Train_score**************" + '\033[0m')
        if _scoring == 'roc_auc':
            Train_score = metrics.roc_auc_score(y_train, y_train_predicted, multi_class='ovr')
        elif _scoring == 'roc_auc_ovr':
            Train_score = metrics.roc_auc_score(y_train, y_train_predicted, multi_class='ovr', average='micro')
        if _scoring == 'roc_auc_ovo':
            Train_score = metrics.roc_auc_score(y_train, y_train_predicted, multi_class='ovo', average=None)
        elif _scoring == 'roc_auc_ovr_weighted':
            Train_score = metrics.roc_auc_score(y_train, y_train_predicted, multi_class='ovr', average='weighted')
        elif _scoring == 'roc_auc_ovo_weighted':
            Train_score = metrics.roc_auc_score(y_train, y_train_predicted, multi_class='ovo', average='weighted')
        elif _scoring == 'accuracy':
            Train_score = metrics.accuracy_score(y_train, y_train_predicted)
        elif _scoring == 'f1':
            Train_score = metrics.f1_score(y_train, y_train_predicted)
        elif _scoring == 'f1_macro':
            Train_score = metrics.f1_score(y_train, y_train_predicted, average='macro')
        elif _scoring == 'f1_micro':
            Train_score = metrics.f1_score(y_train, y_train_predicted, average='micro')
        elif _scoring == 'f1_weighted':
            Train_score = metrics.f1_score(y_train, y_train_predicted, average='weighted')
        elif _scoring == 'recall':
            Train_score = metrics.recall_score(y_train, y_train_predicted)
        elif _scoring == 'recall_macro':
            Train_score = metrics.recall_score(y_train, y_train_predicted, average='macro')
        elif _scoring == 'recall_micro':
            Train_score = metrics.recall_score(y_train, y_train_predicted, average='micro')
        elif _scoring == 'recall_weighted':
            Train_score = metrics.recall_score(y_train, y_train_predicted, average='weighted')
        elif _scoring == 'precision':
            Train_score = metrics.precision_score(y_train, y_train_predicted)
        elif _scoring == 'precision_macro':
            Train_score = metrics.precision_score(y_train, y_train_predicted, average='macro')
        elif _scoring == 'precision_micro':
            Train_score = metrics.precision_score(y_train, y_train_predicted, average='micro')
        elif _scoring == 'precision_weighted':
            Train_score = metrics.precision_score(y_train, y_train_predicted, average='weighted')

        print("{} = ".format(scoring), round(Train_score, 6))

        print('\033[1;32;40m' + "**************Test_score**************" + '\033[0m')
        if _scoring == 'roc_auc':
            Test_score = metrics.roc_auc_score(y_test, y_test_predicted, multi_class='ovr')
        elif _scoring == 'roc_auc_ovr':
            Test_score = metrics.roc_auc_score(y_test, y_test_predicted, average='macro', multi_class='ovr')
        if _scoring == 'roc_auc_ovo':
            Test_score = metrics.roc_auc_score(y_test, y_test_predicted, multi_class='ovo', average=None)
        elif _scoring == 'roc_auc_ovr_weighted':
            Test_score = metrics.roc_auc_score(y_test, y_test_predicted, multi_class='ovr', average='weighted')
        elif _scoring == 'roc_auc_ovo_weighted':
            Test_score = metrics.roc_auc_score(y_test, y_test_predicted, multi_class='ovo', average='weighted')
        elif _scoring == 'accuracy':
            Test_score = metrics.accuracy_score(y_test, y_test_predicted)
        elif _scoring == 'f1':
            Test_score = metrics.f1_score(y_test, y_test_predicted)
        elif _scoring == 'f1_macro':
            Test_score = metrics.f1_score(y_test, y_test_predicted, average='macro')
        elif _scoring == 'f1_micro':
            Test_score = metrics.f1_score(y_test, y_test_predicted, average='micro')
        elif _scoring == 'f1_weighted':
            Test_score = metrics.f1_score(y_test, y_test_predicted, average='weighted')
        elif _scoring == 'recall':
            Test_score = metrics.recall_score(y_test, y_test_predicted)
        elif _scoring == 'recall_macro':
            Test_score = metrics.recall_score(y_test, y_test_predicted, average='macro')
        elif _scoring == 'recall_micro':
            Test_score = metrics.recall_score(y_test, y_test_predicted, average='micro')
        elif _scoring == 'recall_weighted':
            Test_score = metrics.recall_score(y_test, y_test_predicted, average='weighted')
        elif _scoring == 'precision':
            Test_score = metrics.precision_score(y_test, y_test_predicted)
        elif _scoring == 'precision_macro':
            Test_score = metrics.precision_score(y_test, y_test_predicted, average='macro')
        elif _scoring == 'precision_micro':
            Test_score = metrics.precision_score(y_test, y_test_predicted, average='micro')
        elif _scoring == 'precision_weighted':
            Test_score = metrics.precision_score(y_test, y_test_predicted, average='weighted')

        print("{} = ".format(scoring), round(Test_score, 6))

        """
        Parameters:

        optimization 		= str		:	Method for searching the optimize hyperparameters for the model (default = 'Grid')
        num_iter    		= int		:	Number of iterations to run for hyperparameter optimization (default = 20)
        cv 			 		= int		:	cross-validation (default = 10)
        scoring 	 		= str  		:	Method for the evaluation of model: (default = 'roc_auc')
        
        """

    ##############################################******Make Predictions********############################################

    def predict(self, prediction_data='test', unknown_data=None, proba_prediction=False, save_csv=False,
                file_name='predicted_data'):
        global _prediction_data, y_test_proba_predicted, y_train_proba_predicted, test_prediction_score, \
            train_prediction_score, df, unknown_prediction, df_unknown, probability

        _prediction_data = prediction_data
        probability = proba_prediction

        if prediction_data is None:
            raise AssertionError("prediction_data is not defined")

        if proba_prediction:
            if _scoring not in ['roc_auc', 'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted']:

                warnings.warn(
                    "For predicting probabilities scoring metrics can only be 'roc_auc'. Other classification metrics can't handle a mix of binary and continuous targets")
            else:
                pass

        if prediction_data == 'test':
            print('\033[1;32;40m' + '*******Prediction_score on Test dataset*******' + '\033[0m')
            if proba_prediction:
                y_test_proba_predicted = clf_cv.predict_proba(X_test)[:, 1]
                if self.multiclass:
                    if _scoring == 'roc_auc':
                        test_prediction_score = metrics.roc_auc_score(y_test, y_test_proba_predicted, multi_class='ovr')
                    if _scoring == 'roc_auc_ovr':
                        test_prediction_score = metrics.roc_auc_score(y_test, y_test_proba_predicted, average='macro',
                                                                      multi_class='ovr')
                    if _scoring == 'roc_auc_ovo':
                        test_prediction_score = metrics.roc_auc_score(y_test, y_test_proba_predicted,average=None,
                                                                      multi_class='ovo')
                    if _scoring == 'roc_auc_ovr_weighted':
                        test_prediction_score = metrics.roc_auc_score(y_test, y_test_proba_predicted, average='weighted',
                                                                      multi_class='ovr')
                    if _scoring == 'roc_auc_ovo_weighted':
                        test_prediction_score = metrics.roc_auc_score(y_test, y_test_proba_predicted, average='weighted',
                                                                      multi_class='ovo')
                else:
                    test_prediction_score = metrics.roc_auc_score(y_test, y_test_proba_predicted)

                print('roc_auc_score' + ' =', round(test_prediction_score, 6), '\n')
            else:
                test_prediction_score = Test_score
                print(_scoring + ' =', round(test_prediction_score, 6), '\n')

            if self.verbosity == 2:
                if proba_prediction:
                    for i in range(len(X_test)):
                        print(y_test[i], y_test_proba_predicted[i])
                else:
                    for i in range(len(X_test)):
                        print(y_test[i], y_test_predicted[i])
            else:
                pass

            if save_csv:
                predicted_test_data = {'true': y_test, 'predicted': y_test_predicted}
                df = pd.DataFrame(predicted_test_data)
                df.index = X_test.index + 2
                df.to_csv(file_name + '_' + timenow() + '.csv', index_label='index')
            else:
                pass

        elif prediction_data == 'train':
            print('\033[1;32;40m' + '*******Prediction_score on Training dataset*******' + '\033[0m')
            if proba_prediction:
                y_train_proba_predicted = clf_cv.predict_proba(X_train)[:, 1]
                if self.multiclass:
                    if _scoring == 'roc_auc':
                        test_prediction_score = metrics.roc_auc_score(y_test, y_test_proba_predicted, multi_class='ovr')
                    if _scoring == 'roc_auc_ovr':
                        test_prediction_score = metrics.roc_auc_score(y_test, y_test_proba_predicted, average='macro',
                                                                      multi_class='ovr')
                    if _scoring == 'roc_auc_ovo':
                        test_prediction_score = metrics.roc_auc_score(y_test, y_test_proba_predicted, average=None,
                                                                      multi_class='ovo')
                    if _scoring == 'roc_auc_ovr_weighted':
                        test_prediction_score = metrics.roc_auc_score(y_test, y_test_proba_predicted,average='weighted',
                                                                      multi_class='ovr')
                    if _scoring == 'roc_auc_ovo_weighted':
                        test_prediction_score = metrics.roc_auc_score(y_test, y_test_proba_predicted,average='weighted',
                                                                      multi_class='ovo')
                else:
                    train_prediction_score = metrics.roc_auc_score(y_train, y_train_proba_predicted)

                print('roc_auc_score' + ' =', round(train_prediction_score, 6), '\n')
            else:
                train_prediction_score = Train_score
                print(_scoring + ' =', round(train_prediction_score, 6), '\n')

            if self.verbosity == 2:
                if proba_prediction:
                    for i in range(len(X_train)):
                        print(y_train[i], y_train_proba_predicted[i])
                else:
                    for i in range(len(X_train)):
                        print(y_train[i], y_train_predicted[i])
            else:
                pass

            if save_csv:
                predicted_train_data = {'true': y_train, 'predicted': y_train_predicted}
                df = pd.DataFrame(predicted_train_data)
                df.index = X_train.index + 2
                df.to_csv(file_name + '_' + timenow() + '.csv', index_label='index')
            else:
                pass

        elif prediction_data == 'unknown':
            if unknown_data is None:
                raise NameError("'unknown_data' is not defined. If the 'prediction_data' is 'unknown', 'unknown_data' must be define.")
            else:
                if self.normalization == 'zscore':
                    norm_data = zscore(unknown_data)
                    df_unknown = pd.DataFrame(norm_data, columns=unknown_data.columns)
                if self.normalization == 'minmax':
                    minmax_data = MinMaxScaler()
                    norm_data = minmax_data.fit_transform(unknown_data)
                    df_unknown = pd.DataFrame(norm_data, columns=unknown_data.columns)
                elif self.normalization is None:
                    df_unknown = unknown_data

            if proba_prediction:
                unknown_prediction = clf_cv.predict_proba(df_unknown)
            else:
                unknown_prediction = clf_cv.predict(df_unknown)
            if self.verbosity == 2:
                print([unknown_prediction])
            else:
                pass

            if save_csv:
                df = pd.DataFrame(unknown_prediction, columns=['pred'])
                df.index = df_unknown.index
                df.to_csv(file_name + '_' + timenow() + '.csv')
            else:
                pass

        else:
            raise NameError(
                "prediction_data cannot be '{}'. Either pass 'train', 'test', or 'unknown'".format(prediction_data))

        """
        Parameters:

        prediction_data		= bool		:	Dataset to make predictions (default = 'test')
        unknown_data		= Dataframe	:	Unknown dataset for predictions; required when prediction_data is 
                                            'unknown' (default = None)
        proba_prediction	= bool		:	Predict probabilities rather than the exact values for the target if set 
                                            True (default = False)
        save_csv	 		= bool		:	Save a csv file of predictions if set True (default = False)
        file_name	 		= str		:	Name for the csv file (default = 'predicted_data')

        """

    @staticmethod
    def Confusion_matrix(show_plot=True, annot=True, cmap='Blues', figsize=(16, 12), fontsize=14,
                         save_fig=False, fig_name="Confusion_matrix.png", xlabel='Predicted Values',
                         ylabel='Actual Values', title='Seaborn Confusion Matrix with labels\n',
                         dpi=600):

        global cf_matrix
        if _prediction_data == 'train' and probability is False:
            print('\n', '\033[0;30;44m' + ' Confusion_matrix for training dataset   ' + '\033[0m')
            cf_matrix = confusion_matrix(y_train, y_train_predicted)
            print(cf_matrix)
        elif _prediction_data == 'train' and probability is True:
            raise AssertionError("Confusion matrix is not available for probabilities")
        elif _prediction_data == 'test' and probability is False:
            print('\n', '\033[0;30;44m' + ' Confusion_matrix for test dataset   ' + '\033[0m')
            cf_matrix = confusion_matrix(y_test, y_test_predicted)
            print(cf_matrix)
        elif _prediction_data == 'test' and probability is True:
            raise AssertionError("Confusion matrix is not available for probabilities")

        ax = sns.heatmap(cf_matrix, annot=annot, cmap=cmap, linewidths=0.5)

        plt.figure(figsize=figsize)
        plt.rcParams['font.size'] = fontsize
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if show_plot:
            plt.show()
        else:
            pass
        if save_fig:
            ax.figure.savefig(fig_name + timenow() + '.png', format='png', dpi=dpi)

        """
        annot=True, cmap='Blues', figsize=(16, 12), fontsize=14,
                         save_fig=False, fig_name="Confusion_matrix.png", xlabel='Predicted Values',
                         ylabel='Actual Values', title='Seaborn Confusion Matrix with labels\n',
                         dpi=600
        Parameters:

        annot		= bool 	:	Print the confusion matrix values inside the heatmap if set True  (default = False)
        cmap 		= any  	: 	Color map for plot  (default = 'Blues')
        figsize 	= tuple : 	Tuple of two integers for determining the figure size    (default =(16, 12))
        fontsize 	= int  	:	Font size of color-bar and x, y axis   (default =14)
        save_fig 	= bool 	: 	Save plot in the current working directory if True  (default = False)
        figname 	= str   :	Name of fig if save_fig is True  (default = "Confusion_matrix.png")
        xlabel 	    = str   :	Title for x-axis  (default = "Predicted Values")
        ylabel	    = str   :	Title for y-axis  (default = "Actual Values")
        title 	    = str   :	Title for the figure  (default = "Seaborn Confusion Matrix with labels")
        dpi 	    = str   :	Quality of the figure  (default = 600)

        """

    ##############################################******Visualization********###############################################

    def plot_correlation(self, method='pearson', matrix_type='upper', annot=False, cmap='coolwarm', vmin=-1.0, vmax=1.0,
                         figsize=(16, 12), fontsize=14, save_fig=False, save_csv=False, fig_name="Correlation_plot.png",
                         dpi=600):
        global corr_fig

        methods = ['pearson', 'kendall', 'spearman']
        if method not in methods:
            method_error = "{} is not a valid method. Valid methods are 'pearson','kendall', and 'spearman'".format(
                method)
            raise TypeError(method_error)


        df_ = pd.DataFrame(self.inputs)
        correlation = df_.corr(method=method, min_periods=1)

        plt.figure(figsize=figsize)
        plt.rcParams['font.size'] = fontsize

        if matrix_type == 'full':
            corr_fig = sns.heatmap(correlation, annot=annot, cmap=cmap, linewidths=0.5,
                                   vmax=vmax, vmin=vmin)

        elif matrix_type == 'upper':
            mask_ut = np.tril(np.ones(correlation.shape)).astype(np.bool)
            corr_fig = sns.heatmap(correlation, mask=mask_ut, annot=annot, cmap=cmap, linewidths=0.5,
                                   vmax=vmax, vmin=vmin)

        elif matrix_type == 'lower':
            mask_ut = np.triu(np.ones(correlation.shape)).astype(np.bool)
            corr_fig = sns.heatmap(correlation, mask=mask_ut, annot=annot, cmap=cmap, linewidths=0.5,
                                   vmax=vmax, vmin=vmin)
        else:
            raise TypeError(
                "{} is not a valid matrix_type. Available matrix_type are 'full', 'upper', or 'lower'".format(
                    matrix_type))

        plt.show()

        df_corr = pd.DataFrame(correlation)
        if save_csv:
            df_corr.to_csv("Correlation.csv")

        if save_fig:
            corr_fig.figure.savefig(fig_name + timenow() + '.png', format='png', dpi=dpi)

        """
        Parameters:
        method='pearson', matrix_type='upper', annot=False, cmap='coolwarm', vmin=-1.0, vmax=1.0,
                         figsize=(16, 12), fontsize=14, save_fig=False, save_csv=False, fig_name="Correlation_plot.png",
                         dpi=600

        method 		= str  	: 	Method for plottting correlation matrix (default = 'pearson') Available methods = 'perason', 'kendall', or 'spearman'  
        matrix_type	= bool 	:	Type of correlation-matrix for plotting  (default = upper); Available = 'full', 'upper', 'lower'
        annot		= bool 	:	Print the correlation values in the heatmap if set True  (default = False)
        cmap 		= any  	: 	Color map for plot  (default = coolwarm)
        vmin		= float	:	Minimum value for color bar (default = -1.0)
        vmax		= float	:	Maximum value for color bar (default =  1.0)
        figsize 	= tuple : 	Tuple of two integers for determining the figure size    (default =(16, 12))
        fontsize 	= int  	:	Font size of color-bar and x, y axis   (default =14)
        save_fig 	= bool 	: 	Save plot in the current working directory if True  (default = False)
        save_csv 	= bool 	: 	Save a csv file if True  (default = False)
        figname 	= str   :	Name of fig if save_fig is True  (default = "Correlation_plot.png")
        dpi 	    = str   :	Quality of the figure  (default = 600)
        

        """

    def plot_feature_imp(self, kind="barh", random_no=None, figsize=(22, 16), fontsize=20, color='#ff8000', lw=5.0,
                         save_fig=False, fig_name="Feature_imp_Plot(MI).png", dpi=600):
        global imp_fig
        MI = mutual_info_classif(X_train, y_train, random_state=random_no)
        MI = pd.Series(MI)
        MI.index = X_train.columns
        MI.sort_values(ascending=True, inplace=True)
        if self.verbosity in [1, 2]:
            print('\n', '\033[0;30;44m' + '        Mutual Importance        ' + '\033[0m')
            print(MI)
        else:
            pass

        plot_kind = ['barh', 'bar', 'pie', 'line', 'area']
        if kind in plot_kind:
            if kind == "pie":
                imp_fig = MI.plot(kind=kind, normalize=False)
            else:
                imp_fig = MI.plot(kind=kind, figsize=figsize, fontsize=fontsize, color=color, lw=lw)

            plt.show()

        else:
            error = "{} is not a valid type for plotting feature importance. Only 'barh', 'bar', 'pie', 'line', 'area' can be used for plotting".format(
                kind)
            raise TypeError(error)

        if save_fig:
            imp_fig.figure.savefig(fig_name + timenow() + '.png', format='png', dpi=dpi)

        """
        Parameters:

        kind 		= str		: 	Type of plot: (default = 'barh'); Available types = 'barh', 'bar', 'pie', 'line', 'area'  
        random_no 	= any		:	Random number to reproduce results (default = None)
        figsize 	= tuple  	: 	Tuple of two integers for determining the figure size (default =(22, 16))		 
        fontsize 	= int  		:	Font size of color-bar and x, y axis (default =20)
        color 		= str  		: 	Color for plot    (default = '#ff8000')	
        lw 			= float 	: 	Width of bars if kind == 'bar' or 'barh' (default = 5.0)
        save_fig 	= bool 		: 	Save plot in the current working directory if True (default = False)
        figname 	= str   	:	Name of fig if save_fig is True (default = "Feature_imp_Plot(MI).png")

        """

    @staticmethod
    def plot_roc_curve(plot_for='test', figsize=(9, 7), lines_fmt=None, label='ROC_curve', fontsize=18, ticksize=18,
                       xlabel='False positive rate', ylabel='True positive rate', legend='lower right', alpha=0.8,
                       save_fig=False, fig_name='roc_plot', dpi=600):

        global fpr, tpr

        if lines_fmt is None:
            lines_fmt = dict(color=["#339966", "#cc0000"], lw=3)
        pd.options.display.width = 0

        flatui = lines_fmt['color']
        palette = sns.color_palette(flatui)
        sns.set_palette(palette)

        linewidth = lines_fmt['lw']

        labelsize = fontsize
        ticksize = ticksize

        # mean_fpr = np.linspace(0, 1, 1000)

        if plot_for == 'train':
            y_proba=clf_cv.predict_proba(X_train)[:, 1]
            fpr, tpr, _ = roc_curve(y_train, y_proba)
        if plot_for == 'test':
            y_proba = clf_cv.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)

        # auc = roc_auc_score(y_test, y_proba)

        plt.legend(loc=4)
        roc_fig, ax = plt.subplots(figsize=figsize)
        plt.rc('font', family='serif')
        ax.plot(fpr, tpr, label=label, lw=linewidth, alpha=alpha)
        ax.plot(fpr, fpr, linestyle='--', lw=linewidth, label='Chance', alpha=alpha)

        ax.set(xlim=[-0.02, 1.02], ylim=[-0.02, 1.02])
        ax.legend(loc=legend)
        ax.set_xlabel(xlabel, fontsize=labelsize)
        ax.set_ylabel(ylabel, fontsize=labelsize)
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)

        plt.legend(fontsize='x-large')

        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.6)
            ax.spines[axis].set_color('dimgrey')
        plt.show()

        if save_fig:
            roc_fig.savefig(fig_name + timenow() + '.png', format='png', dpi=dpi)

        """
        Parameters:
        plot_for='test', figsize=(9, 7), lines_fmt=None, label='ROC_curve', fontsize=18, ticksize=18,
                       xlabel='False positive rate', ylabel='True positive rate', legend='lower right', alpha=0.8,
                       save_fig=False, fig_name='roc_plot', dpi=600

        plot_for    = str       :   Determines whether to plot results for training or testing dataset (default='test')     
        figsize 	= tuple 	: 	Tuple of two integers for determining the figure size  (default=(9, 7))		 
        lines_fmt 	= dict		: 	Dictionary for the formatting of lines i.e. 'color' and linewidth('lw')	 
                                    (default={'color': ["#339966", "#cc0000"], 'lw': 3}
        label 		= str		:	Set label inside the plot (default ='ROC_curve')
        fontsize 	= int 		: 	Set fontsize for the x and y labels  (default=18)
        xlabel      = str       :   Label for x-axis (default=False positive rate)
        ylabel      = str       :   Label for y-axis (default=True positive rate)
        legend      = str       :   Text to right in the legend (default='lower right')
        alpha       = float     :   Intensity of colors (default=0.8)
        ticksize 	= int 		:	Set fontsize for the x and y ticks   (default=18)
        save_fig 	= bool 		: 	Save Figure in the current directory if set True    (default=False)
        fig_name 	= str  		: 	Name for the figure     (default='roc_plot')
        dpi         = int       :   Quality of the image (default=600)

        """


    def save_data(self, filename=None, verbosity=2):
        global save_content

        save_content = {}
        if verbosity is 0:
            save_content['inputs'] = list(self.inputs)
            save_content['target'] = list(self.target)
            save_content['scoring_method'] = _scoring
            save_content['problem'] = self.problem
            save_content['model_name'] = _model_name
            if _optimization is not None:
                save_content['best_parameters'] = best_params
            else:
                pass
            save_content['Train_score'] = Train_score
            save_content['Test_score'] = Test_score

        elif verbosity in [1, 2]:
            save_content['inputs'] = list(self.inputs)
            save_content['target'] = list(self.target)
            save_content['random_state'] = self.random_state
            save_content['test_size'] = self.test_size
            save_content['scaling_method'] = self.normalization
            save_content['verbosity'] = self.verbosity
            save_content['optimization_method'] = _optimization
            save_content['problem'] = self.problem,
            save_content['model_name'] = _model_name
            if _optimization is not None:
                save_content['best_parameters'] = best_params
            else:
                pass
            save_content['scoring_method'] = _scoring
            save_content['Train_score'] = Train_score
            save_content['Test_score'] = Test_score


        json_converted = json.dumps(save_content, indent=5)
        print(json_converted)

        if filename is None:
            save_file = open("results" + timenow() + '.txt', "w")
            save_file.write(json_converted)
            save_file.close()
        else:
            save_file = open(filename+ timenow()+'.txt', "w")
            save_file.write(json_converted)
            save_file.close()

    """
    file_name 	= str  		: 	Name for the file     (default='roc_plot')
    verbosity   = int       :   Quality of the image (default=600)
    """