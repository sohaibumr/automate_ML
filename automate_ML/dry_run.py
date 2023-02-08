import warnings
from preprocessing import Preprocessing
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, ShuffleSplit
import pandas as pd
from tabulate import tabulate
from matplotlib import pyplot as plt

"""                                Classification models                                    """

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import ExtraTreeClassifier
from sklearn.neural_network import MLPClassifier

"""                                Regression models                                    """

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.neighbors import RadiusNeighborsRegressor, KNeighborsRegressor
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.linear_model import ARDRegression, LinearRegression, SGDRegressor, HuberRegressor

global models

class all_Models(Preprocessing):

    def best_model(self, n_splits=100, scoring=None, save_txt=True, filename='Models_score', show=True,
                   save_fig=False, fig_name='Models Comparison', dpi=600):

        global models

        if scoring is None:
            raise AttributeError("scoring is not defined. Define a scoring method according to the type of your problem")

        if self.problem is None:
            raise AttributeError("Define a type of problem to solve i.e. 'Classification' or 'Regression'")

        elif self.problem is 'Classification':

            classification_scoring_methods = ['roc_auc', 'roc_auc_ovr', 'roc_auc_micro', 'roc_auc_ovo', 'roc_auc_ovr_weighted',
                               'roc_auc_ovo_weighted','accuracy', 'f1', 'f1_macro', 'f1_micro', 'f1_weighted', 'recall',
                               'recall_macro', 'recall_micro', 'recall_weighted', 'precision', 'precision_macro',
                               'precision_micro', 'precision_weighted']

            multiclass_scoring_methods = ['accuracy', 'f1_macro', 'f1_micro', 'f1_weighted', 'recall_macro', 'recall_micro',
                                          'recall_weighted', 'precision_macro', 'precision_micro', 'precision_weighted']

            if self.multiclass and scoring not in multiclass_scoring_methods:
                raise NameError("'{0}' metrics cannot be used for evaluating multiclass problems. "
                                "Use a metrics from the following list {1}".format(scoring, multiclass_scoring_methods))
            else:
                pass

            if scoring in classification_scoring_methods:
                pass
            else:
                score_error = "{0} is not a valid scoring method. Valid scoring methods are following: {1} "\
                    .format(scoring, classification_scoring_methods)
                raise NameError(score_error)


            print('\n', '\033[0;30;44m' + "********************* Models Evaluation has Started *********************" + '\033[0m', '\n')
            models = [
                      ('AdaB', AdaBoostClassifier()),
                      ('Bagging', BaggingClassifier()),
                      ('CalCV', CalibratedClassifierCV()),
                      ('DecisionTree', DecisionTreeClassifier()),
                      ('ExtraTrees', ExtraTreesClassifier()),
                      ('ExtraTree', ExtraTreeClassifier()),
                      ('GradientBoosting', GradientBoostingClassifier()),
                      ('KNeighbors', KNeighborsClassifier()),
                      ('LogisticRegression', LogisticRegression(solver='liblinear', multi_class='ovr')),
                      ('LDA', LinearDiscriminantAnalysis()),
                      ('LGBM', LGBMClassifier()),
                      ('Linear_SVC', LinearSVC()),
                      ('NuSvc', NuSVC(nu=0.1)),
                      ('RandomForest', RandomForestClassifier()),
                      ('Ridge', RidgeClassifier()),
                      ('SGDC', SGDClassifier()),
                      ('SVC', SVC(gamma='auto'))
                            ]

        elif self.problem is 'Regression':

            regression_scoring_methods = ['r2', 'explained_variance', 'max_error', 'neg_mean_absolute_error', 'neg_mean_squared_error',
						   'neg_root_mean_squared_error', 'neg_mean_squared_log_error', 'neg_median_absolute_error',
						   'neg_mean_poisson_deviance', 'neg_mean_gamma_deviance', 'neg_mean_absolute_percentage_error']

            if scoring in regression_scoring_methods:
                pass
            else:
                score_error = "{0} is not a valid scoring method. Valid scoring methods are following: {1} " \
                    .format(scoring, regression_scoring_methods)
                raise NameError(score_error)

            print('\n',
                  '\033[0;30;44m' + "********************* Models Evaluation has Started *********************" + '\033[0m',
                  '\n')
            models = [
                ('AdaBoost', AdaBoostRegressor()),
                ('ARDRegression', ARDRegression()),
                ('Bagging', BaggingRegressor()),
                # ('CatBoost', CatBoostRegressor()),
                ('DecisionTree', DecisionTreeRegressor()),
                ('ExtraTrees', ExtraTreesRegressor()),
                ('ExtraTree', ExtraTreeRegressor()),
                ('GradientBoosting', GradientBoostingRegressor()),
                ('HistGradientBoosting', HistGradientBoostingRegressor()),
                ('Huber', HuberRegressor()),
                ('KNeighbors', KNeighborsRegressor()),
                ('LGBM', LGBMRegressor()),
                ('LinearSVR', LinearSVR()),
                ('LinearRegression', LinearRegression()),
                ('MlPRegressor', MLPRegressor()),
                ('NuSVR', NuSVR()),
                ('RandomForest', RandomForestRegressor()),
                ('RadiusNeighbors', RadiusNeighborsRegressor()),
                ('SVR', SVR()),
                ('SGDRegressor', SGDRegressor())
            ]

        else:
            raise NameError("{} is not a valid option. Choose either 'Classification' or 'Regression'.".format(self.problem))


        results = []
        names = []
        for name, _model in models:
            if self.problem is 'Classification':
                ss_split = StratifiedShuffleSplit(n_splits=n_splits, test_size=self.test_size,
                                                  random_state=self.random_state)
                cv_results = cross_val_score(_model, self.inputs, self.target, cv=ss_split, scoring=scoring)
                results.append(cv_results.mean())
                names.append(name)
                print("%s: score=%f std=(%f)" % (name, cv_results.mean(), cv_results.std()))

            elif self.problem is 'Regression':
                ss_split = ShuffleSplit(n_splits=n_splits, test_size=self.test_size,
                                                  random_state=self.random_state)
                cv_results = cross_val_score(_model, self.inputs, self.target, cv=ss_split, scoring=scoring)
                results.append(cv_results.mean())
                names.append(name)
                print("%s: score=%f" % (name, cv_results.mean()))

        result = pd.Series(results)
        result.index = names
        result.sort_values(ascending=False, inplace=True)
        print('\n',
              '\033[0;30;44m' + "**************** Models evaluation has been Completed ****************" + '\033[0m')
        print('\n', '\033[0;30;44m' + 'Sorted Models w.r.t score' + '\033[0m')
        result_tab = result.to_frame()
        tabulated_results = tabulate(result_tab, headers=['Models', 'Score'], tablefmt='fancy_grid')
        print(tabulated_results)

        # plot
        comparison_fig = result.plot(kind='bar')
        if show:
            plt.show()
        if save_fig:
            comparison_fig.savefig(fig_name + timenow() + '.png', format='png', dpi=dpi)
        else:
            pass
        if save_txt:
            result.to_csv(filename+'.csv', index_label='Model', header=['Score'])

        """
        Parameters:

        n_splits 	 = int 		: 	No of splits  (default=100)		 
        scoring 	 = str 		: 	Scoring method  (default=None)
        save_txt 	 = bool 	:	Save a txt files with model names and corresponding scores   (default=True)
        filename 	 = str 		: 	Name of the txt file   (default='Models_score')
        show 		 = bool  	: 	Print out the sorted table and plots a bar chart of the models with corresponding 
                                    scores if set True   (default=True)
        save_fig	 = bool  	: 	Whether to save the figure or not (default=False)
        fig_name	 = str  	: 	Name of the png file (default=Models Comparison)
        dpi     	 = int  	: 	Determines the quality of the figure (default=600)
        
        """