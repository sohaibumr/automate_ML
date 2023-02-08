import json

import numpy as np

from preprocessing import Preprocessing, train_test_data
from reg_models_dependencies import *
from preprocessing import timenow
import math

""""                                Classification Models                                   """
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.neighbors import RadiusNeighborsRegressor, KNeighborsRegressor
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.linear_model import ARDRegression, LinearRegression, SGDRegressor, HuberRegressor



"""                                 Pre-processing tools                                     """

from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV

"""                                     Metrices                                            """

from sklearn import metrics


"""                                   Plotting tools                                        """

from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression



global reg, _params, _model_name, reg_without_opt, model, reg_cv, Train_score, Test_score, \
	y_train_predicted, y_test_predicted, X_train, X_test, y_train, y_test, _optimization, _scoring, r2_train, r2_test, \
	best_params, best_estimator, df, test_prediction, train_prediction, self_prediction, unknown_prediction, df_unknown, \
	corr_fig, imp_fig, save_content


"""""""""""""""""""""""""""""""""""""""""""""""""Regression Models"""""""""""""""""""""""""""""""""""""""""""""""""""""""

class Regression(Preprocessing):

	@staticmethod
	def model_names():
		model_names = ['AdaBoost', 'ARD', 'Bagging', 'CatBoost', 'DecisionTree', 'ExtraTrees', 'ExtraTree',
					   'GradientBoosting', 'HistGradientBoosting', 'Huber', 'KNeighbors', 'LGBM', 'LinearSVR',
					   'LinearRegression', 'MlPRegressor', 'NuSVR', 'RandomForest', 'RadiusNeighbors', 'SVR',
					   'SGDRegressor']

		return model_names

	@staticmethod
	def Model(model_name=None, params=None, random_state=300):
		global reg, _params, reg_without_opt, _model_name

		_model_names = ['AdaBoost', 'ARD', 'Bagging', 'CatBoost', 'DecisionTree', 'ExtraTrees', 'ExtraTree',
					   'GradientBoosting', 'HistGradientBoosting', 'Huber', 'KNeighbors', 'LGBM', 'LinearSVR',
					   'LinearRegression', 'MlPRegressor', 'NuSVR', 'RandomForest', 'RadiusNeighbors', 'SVR',
					   'SGDRegressor']

		if model_name not in _model_names:
			model_name_error = "{0} is not a valid model_name or not available. " \
							   "Use from following: {1}".format(model_name, _model_names)

			raise NameError(model_name_error)
		else:
			pass


		print('\n', '\033[0;30;44m' + '         Solving Regression Problem        ' + '\033[0m', '\n')


		if model_name == 'AdaBoost':
			reg = AdaBoostRegressor()
			_model_name = 'AdaBoostRegressor'

			if params is None:
				_params = {
					"n_estimators": list(range(100, 1000)),
					"learning_rate": list(np.linspace(0.001, 0.1, num=30)),
					"loss": ['linear', 'square', 'exponential'],
					 "random_state": [random_state]
						   }
			else:
				_params= dict(params)
				reg_without_opt = AdaBoostRegressor(**params)


		if model_name == 'ARD':
			reg = ARDRegression()
			_model_name = 'ARDRegression'

			if params is None:
				_params = {
					"n_iter": list(range(100, 1000)),
					"alpha_1": [1e-04],
					"alpha_2": [1e-04],
					 "lambda_1": [1e-04],
					"lambda_2": [1e-04]
						   }
			else:
				_params= dict(params)
				reg_without_opt = ARDRegression(**params)


		if model_name == 'Bagging':
			reg = BaggingRegressor()
			_model_name = 'BaggingRegressor'

			if params is None:
				_params = {
					#"max_leaf_nodes": list(range(10, 500)),
					"max_samples": list(range(1, 20)),
					"n_estimators": list(range(100, 1000)),
					 "random_state": [random_state]
						   }
			else:
				_params= dict(params)
				reg_without_opt = BaggingRegressor(**params)


		if model_name == 'CatBoost':
			reg = CatBoostRegressor()
			_model_name = 'CatBoostRegressor'

			if params is None:
				_params = {
					"iterations" : list(range(10, 100)),
					"l2_leaf_reg": list(range(1, 10)),
					"model_size_reg": list(np.linspace(1.0, 10.0, num=30)),
					"rsm": [0.2, 0.9],
					"border_count": list(range(10, 500)),
					"random_state": [random_state]
						   }
			else:
				_params= params
				reg_without_opt = CatBoostRegressor(**params)


		if model_name == 'DecisionTree':
			reg = DecisionTreeRegressor()
			_model_name = 'DecisionTreeRegressor'

			if params is None:
				_params = {
					"criterion": ['squared_error', 'absolute_error', 'friedman_mse'],
					"splitter": ['best', 'random'],
					"max_leaf_nodes": list(range(10, 500)),
					"max_depth": list(range(1, 10)),
					"min_samples_split": list(range(1, 20)),
					"min_samples_leaf": list(range(1, 20)),
					 "random_state": [random_state]
						   }
			else:
				_params= dict(params)
				reg_without_opt = DecisionTreeRegressor(**params)


		if model_name == 'ExtraTrees':
			reg = ExtraTreesRegressor()
			_model_name = 'ExtraTreesRegressor'

			if params is None:
				_params = {
					"n_estimators": list(range(100, 1000)),
					"criterion": ['squared_error', 'absolute_error', 'friedman_mse'],
					"max_leaf_nodes": list(range(10, 500)),
					"max_depth": list(range(1, 10)),
					"min_samples_split": list(range(1, 20)),
					"min_samples_leaf": list(range(1, 20)),
					 "random_state": [random_state]
						   }
			else:
				_params= dict(params)
				reg_without_opt = ExtraTreesRegressor(**params)


		if model_name == 'ExtraTree':
			reg = ExtraTreeRegressor()
			_model_name = 'ExtraTreeRegressor'

			if params is None:
				_params = {
					"criterion": ['squared_error', 'absolute_error', 'friedman_mse'],
					"splitter": ['best', 'random'],
					"max_leaf_nodes": list(range(10, 500)),
					"max_depth": list(range(1, 10)),
					"min_samples_split": list(range(1, 20)),
					"min_samples_leaf": list(range(1, 20)),
					 "random_state": [random_state]
						   }
			else:
				_params= dict(params)
				reg_without_opt = ExtraTreeRegressor(**params)


		if model_name == 'GradientBoosting':
			reg = GradientBoostingRegressor()
			_model_name = 'GradientBoostingRegressor'

			if params is None:
				_params = {
					"loss": ['squared_error', 'absolute_error', 'huber'],
					"max_leaf_nodes": list(range(10, 500)),
					"max_depth": list(range(1, 10)),
					"min_samples_leaf": list(range(1, 50)),
					"learning_rate": list(np.linspace(0.001, 0.1, num=30)),
					"n_estimators": list(range(100, 1000)),
					 "random_state": [random_state]
						   }
			else:
				_params= dict(params)
				reg_without_opt = GradientBoostingRegressor(**params)


		if model_name == 'HistGradientBoosting':
			reg = HistGradientBoostingRegressor()
			_model_name = 'HistGradientBoostingRegressor'

			if params is None:
				_params = {
					"loss": ['squared_error', 'absolute_error', 'poisson'],
					"max_iter": list(range(10, 200)),
					"max_leaf_nodes": list(range(10, 1000)),
					"max_depth": list(range(1, 10)),
					"min_samples_leaf": list(range(1, 50)),
					"learning_rate": list(np.linspace(0.001, 0.1, num=30)),
					"n_estimators": list(range(100, 1000)),
					 "random_state": [random_state]
						   }
			else:
				_params= dict(params)
				reg_without_opt = GradientBoostingRegressor(**params)


		if model_name == 'Huber':
			reg = HuberRegressor()
			_model_name = 'HuberRegressor'

			if params is None:
				_params = {
					'epsilon': list(np.linspace(0.0, 1.0, num=30)),
					"max_iter": list(range(10,500)),
						}
			else:
				_params= dict(params)
				reg_without_opt = HuberRegressor(**params)


		if model_name == 'KNeighbors':
			reg = KNeighborsRegressor()
			_model_name = 'KNeighborsRegressor'

			if params is None:
				_params = {
					'n_neighbors': list(range(3, 8)),
					'weights': ['uniform', 'distance'],
					'algorithm': ['auto', 'ball_tree', 'kd_tree'],
					'leaf_size': list(range(1, 100)),
						   }
			else:
				_params= dict(params)
				reg_without_opt = KNeighborsRegressor(**params)


		if model_name == 'LGBM':
			reg = LGBMRegressor()
			_model_name = 'LGBMRegressor'

			if params is None:
				_params = {
					"boosting_type": ["gbdt","dart","rf"],
					"num_leaves": list(range(10, 1000)),
					"bagging_freq": list(range(1, 10)),
					"bagging_fraction": list(np.linspace(0.1, 0.9, num=30)),
					"learning_rate": list(np.linspace(0.001, 0.1, num=30)),
					"n_estimators": list(range(100, 1000)),
					 "random_state": [random_state]
						   }
			else:
				_params= dict(params)
				reg_without_opt = LGBMRegressor(**params)


		if model_name == 'LinearSVR':
			reg = LinearSVR()
			_model_name = 'LinearSVR'

			if params is None:
				_params = {
					'epsilon': list(np.linspace(0.0, 1.0, num=30)),
					'C': list(np.linspace(1.0, 10.0, num=30)),
					'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
				   	'dual': [False],
					'max_iter': list(range(5, 500)),
					"random_state": [random_state]
						   }
			else:
				_params= dict(params)
				reg_without_opt = LinearSVR(**params)


		if model_name == 'LinearRegression':
			reg = LinearRegression()
			_model_name = 'LinearRegression'

			if params is None:
				_params = {
						"fit_intercept": [True]
						   }
			else:
				_params= dict(params)
				reg_without_opt = LinearRegression(**params)


		if model_name == 'MlPRegressor':
			reg = MLPRegressor()
			_model_name = 'MLPRegressor'

			if params is None:
				_params = {
					'hidden_layer_sizes': [4, 8],
					'activation': ['relu', 'identity'],
					'solver': ['adam', 'lbfgs', 'sgd'],
					"learning_rate": ['constant', 'invscaling', 'adaptive'],
					'learning_rate_init': list(np.linspace(0.01, 0.1, num=30)),
					'max_iter': list(range(5, 500)),
				   	'batch_size': ['auto'],
					'random_state': [random_state]
						   }
			else:
				_params= dict(params)
				reg_without_opt = MLPRegressor(**params)


		if model_name == 'NuSVR':
			reg = NuSVR()
			_model_name = 'NuSVR'

			if params is None:
				_params = {
					'nu': list(np.linspace(0.0, 10.0, num=30)),
					'C': list(np.linspace(1.0, 10.0, num=30)),
					'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
					'degree': list(range(1, 10)),
					'gamma': ['auto'],
					'coef0': list(np.linspace(0.0, 0.3, num=30)),
					'max_iter': list(range(5, 500)),
						   }
			else:
				_params= dict(params)
				reg_without_opt = NuSVR(**params)


		if model_name == 'RandomForest':
			reg = RandomForestRegressor()
			_model_name = 'RandomForestRegressor'

			if params is None:
				_params = {
					"n_estimators": list(range(100, 1000)),
					"criterion": ['squared_error', 'absolute_error', 'friedman_mse'],
					"max_leaf_nodes": list(range(10, 500)),
					"max_depth": list(range(1, 10)),
					"min_samples_split": list(range(1, 20)),
					"min_samples_leaf": list(range(1, 20)),
					 "random_state": [random_state]
						   }
			else:
				_params= dict(params)
				reg_without_opt = RandomForestRegressor(**params)


		if model_name == 'RadiusNeighbors':
			reg = RadiusNeighborsRegressor()
			_model_name = 'RadiusNeighborsRegressor'

			if params is None:
				_params = {
					'radius': list(np.linspace(2.0, 6.0, num=15)),
					'weights': ['uniform', 'distance'],
					'algorithm': ['auto', 'ball_tree', 'kd_tree'],
					'leaf_size': list(range(1, 100)),
						   }
			else:
				_params= dict(params)
				reg_without_opt = RadiusNeighborsRegressor(**params)


		if model_name == 'SVR':
			reg = SVR()
			_model_name = 'SVR'

			if params is None:
				_params = {
					'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
					'degree': list(range(1, 10)),
				   	'gamma': ['auto'],
					'coef0': list(np.linspace(0.0, 0.3, num=30)),
					'C': list(np.linspace(1.0, 10.0, num=30)),
					'max_iter': list(range(5, 500)),
						   }
			else:
				_params= dict(params)
				reg_without_opt = SVR(**params)


		if model_name == 'SGDRegressor':
			reg = SGDRegressor()
			_model_name = 'SGDRegressor'

			if params is None:
				_params = {
					"loss": ['squared_error'],
					"penalty": ['l1', 'l2', 'elasticnet'],
					"max_iter": list(range(10,500)),
					'epsilon': list(np.linspace(0.0, 1.0, num=30)),
					"learning_rate": ['optimal', 'constant', 'invscaling'],
					"random_state": [random_state]
						}
			else:
				_params= dict(params)
				reg_without_opt = SGDRegressor(**params)


###*********************************************Fitting and Predictions*********************************************###

	def fit(self, optimization='Grid', num_iter=20, cv=10, scoring='r2'):
		global X_train, X_test, y_train, y_test, _scoring, model,  reg_cv, Train_score, Test_score, y_train_predicted, \
			y_test_predicted, r2_train, r2_test, Test_score, best_params, best_estimator, _optimization

		scoring_methods = ['r2', 'explained_variance', 'max_error', 'neg_mean_absolute_error', 'neg_mean_squared_error',
						   'neg_root_mean_squared_error', 'neg_mean_squared_log_error', 'neg_median_absolute_error',
						   'neg_mean_poisson_deviance', 'neg_mean_gamma_deviance', 'neg_mean_absolute_percentage_error']

		if scoring in scoring_methods:
			_scoring = scoring
		else:
			score_error = "{0} is not a valid scoring method. Valid scoring methods are following: {1} ".\
				format(scoring, scoring_methods)

			raise NameError(score_error)

		_optimization = optimization
		if optimization is 'Grid':
			reg_cv = GridSearchCV(reg, _params, cv=cv, scoring=scoring, error_score='raise')
		elif optimization is 'Randomized':
			reg_cv = RandomizedSearchCV(reg, _params, n_iter=num_iter, cv=cv, scoring=scoring,
										random_state=self.random_state, error_score='raise')
		elif optimization is 'Bayesian':
			reg_cv = BayesSearchCV(reg, _params, scoring=scoring, n_iter=num_iter, cv=cv, random_state=self.random_state,
								   error_score='raise')
		elif optimization is None:
			reg_cv = reg_without_opt
		else:
			search_error = "{} is not a valid option for hyper-paramteres optimization. Available options are 'Grid', " \
						   "'Randomized' or 'Bayesian'".format(optimization)
			raise NameError (search_error)

		X_train, X_test, y_train, y_test = train_test_data()

		reg_cv.fit(X_train, y_train)
		y_train_predicted = reg_cv.predict(X_train)
		y_test_predicted = reg_cv.predict(X_test)

		if optimization is None and scoring in scoring_methods:
			pass
		else:
			print("------------------------------------------------------------------------------------------------")
			best_params = reg_cv.best_params_
			best_estimator = reg_cv.best_estimator_
			best_score = reg_cv.best_score_
			print('Best estimator:', best_estimator)
			print('Best parameters:', best_params)
			print('Best score:', round(best_score, 6))
			print("------------------------------------------------------------------------------------------------\n")


		print('\033[1;32;40m' + "**************Train_score**************" + '\033[0m')
		if scoring == 'r2':
			Train_score = metrics.r2_score(y_train, y_train_predicted)
		elif scoring ==  'explained_variance':
			Train_score = metrics.explained_variance_score(y_train, y_train_predicted)
		elif scoring == 'max_error':
			Train_score = metrics.max_error(y_train, y_train_predicted)
		elif scoring == 'neg_mean_absolute_error':
			Train_score = -1 * (metrics.mean_absolute_error(y_train, y_train_predicted))
		elif scoring == 'neg_mean_squared_error':
			Train_score = -1 * (metrics.mean_squared_error(y_train, y_train_predicted))
		elif scoring == 'neg_root_mean_squared_error':
			Train_score = -1 * math.sqrt((metrics.mean_squared_error(y_train, y_train_predicted)))
		elif scoring ==  'neg_mean_squared_log_error':
			Train_score = -1 * (metrics.mean_squared_log_error(y_train, y_train_predicted))
		elif scoring == 'neg_median_absolute_error':
			Train_score = -1 * (metrics.median_absolute_error(y_train, y_train_predicted))
		elif scoring == 'neg_mean_poisson_deviance':
			Train_score = -1 * (metrics.mean_poisson_deviance(y_train, y_train_predicted))
		elif scoring == 'neg_mean_gamma_deviance':
			Train_score = -1 * (metrics.mean_gamma_deviance(y_train, y_train_predicted))
		elif scoring == 'neg_mean_absolute_percentage_error':
			Train_score = -1 * (metrics.mean_absolute_percentage_error(y_train, y_train_predicted))


		print("{} =".format(scoring) + str(round(Train_score, 6)))

		print('\033[1;32;40m' + "**************Train_score**************" + '\033[0m')
		if scoring == 'r2':
			Test_score = metrics.r2_score(y_test, y_test_predicted)
		elif scoring ==  'explained_variance':
			Test_score = metrics.explained_variance_score(y_test, y_test_predicted)
		elif scoring == 'max_error':
			Test_score = metrics.max_error(y_test, y_test_predicted)
		elif scoring == 'neg_mean_absolute_error':
			Test_score = -1 * (metrics.mean_absolute_error(y_test, y_test_predicted))
		elif scoring == 'neg_mean_squared_error':
			Test_score = -1 * (metrics.mean_squared_error(y_test, y_test_predicted))
		elif scoring == 'neg_root_mean_squared_error':
			Test_score = -1 * math.sqrt((metrics.mean_squared_error(y_test, y_test_predicted)))
		elif scoring ==  'neg_mean_squared_log_error':
			Test_score = -1 * (metrics.mean_squared_log_error(y_test, y_test_predicted))
		elif scoring == 'neg_median_absolute_error':
			Test_score = -1 * (metrics.median_absolute_error(y_test, y_test_predicted))
		elif scoring == 'neg_mean_poisson_deviance':
			Test_score = -1 * (metrics.mean_poisson_deviance(y_test, y_test_predicted))
		elif scoring == 'neg_mean_gamma_deviance':
			Test_score = -1 * (metrics.mean_gamma_deviance(y_test, y_test_predicted))
		elif scoring == 'neg_mean_absolute_percentage_error':
			Test_score = -1 * (metrics.mean_absolute_percentage_error(y_test, y_test_predicted))

		print("{} =".format(scoring) + str(round(Test_score, 6)))


	"""
	Parameters:

	test_size    		= float		:	For specifying test fraction for dataset (default = 0.20)
	random_no    		= any		: 	Random number for reproducing the results    (default = None)
	optimization 		= str		:	Method for searching the best hyperparameters for the model  (default = 'Grid'); Available methods are = 'Grid', 'Bayesian' and 'Randomized'
	cv 			 		= integer	:	cross-validation
	scoring 	 		= str  		:	Method for the evaluation of model: (default = 'r2'); Available methods are = 'explained_variance', 'max_error', 'neg_mean_absolute_error', 'neg_mean_squared_error',
						   				'neg_root_mean_squared_error', 'neg_mean_squared_log_error', 'neg_median_absolute_error', 'r2',
						   				'neg_mean_poisson_deviance', 'neg_mean_gamma_deviance', 'neg_mean_absolute_percentage_error'.
	return_dataset   	= str		:   Returns a csv file of training or test dataset (default = None); Available = 'train' or 'test'.

	"""

	def predict(self, prediction_data='test', unknown_data=None, save_csv=False, file_name= 'predicted_data'):
		global df, unknown_prediction, test_prediction, train_prediction, df_unknown


		if prediction_data is None:
			raise AssertionError("prediction_data is not defined")

		if prediction_data is 'test':
			test_prediction = reg_cv.predict(X_test)
			print('\033[1;32;40m' + '*******Prediction_score on Test dataset*******' + '\033[0m')
			test_prediction_score = Test_score
			print(_scoring, '=',  round(test_prediction_score, 6), '\n')

			if self.verbosity is 2:
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

		elif prediction_data is 'train':
			train_prediction = reg_cv.predict(X_train)
			print('\033[0;30;44m' + '*******Prediction_score on Training dataset*******' + '\033[0m')
			train_prediction_score = Train_score
			print(_scoring, '=', round(train_prediction_score, 6), '\n')

			if self.verbosity is 2:
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

		elif prediction_data is 'unknown':
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

			unknown_prediction = reg_cv.predict(df_unknown)

			if self.verbosity is 2:
				print([unknown_prediction])
			else:
				pass

			if save_csv:
				df = pd.DataFrame(unknown_prediction, columns=['pred'])
				df.index = df_unknown.index
				df.to_csv(file_name + '_' + timenow() + '.csv')
			else:
				pass

	"""
	Parameters:

	prediction_data		= bool		:	Dataset to make predictions; only if predict_unknown is True (default = 'test')
	unknown_data		= Dataframe	:	Unknown dataset for predictions; required when prediction_data is 'unknown' (default = None)
	save_csv	 		= bool		:	Whether to save a csv file of predictions or not (default = False)
	file_name	 		= str		:	Name for the csv file

	"""

##############################################******Visualization********###############################################


	def plot_correlation(self, method='pearson', matrix_type='upper', annot=False, cmap='coolwarm',  vmin=-1.0, vmax=1.0,
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

		if matrix_type == 'full':
			plt.figure(figsize=figsize)
			plt.rcParams['font.size'] = fontsize
			corr_fig = sns.heatmap(correlation, annot=annot, cmap=cmap, linewidths=0.5,
								 vmax=vmax, vmin=vmin)
			plt.show()

		elif matrix_type == 'upper':
			mask_ut = np.tril(np.ones(correlation.shape)).astype(np.bool)
			plt.figure(figsize=figsize)
			plt.rcParams['font.size'] = fontsize
			corr_fig = sns.heatmap(correlation, mask=mask_ut, annot=annot, cmap=cmap, linewidths=0.5,
								 vmax=vmax, vmin=vmin)
			plt.show()

		elif matrix_type == 'lower':
			mask_ut = np.triu(np.ones(correlation.shape)).astype(np.bool)
			plt.figure(figsize=figsize)
			plt.rcParams['font.size'] = fontsize
			corr_fig = sns.heatmap(correlation, mask=mask_ut, annot=annot, cmap=cmap, linewidths=0.5,
								 vmax=vmax, vmin=vmin)
			plt.show()

		else:
			raise TypeError(
				"{} is not a valid matrix_type. Available matrix_type are 'full', 'upper', or 'lower'".format(
					matrix_type))

		df_corr = pd.DataFrame(correlation)
		if save_csv:
			df_corr.to_csv("Correlation.csv")

		if save_fig:
			corr_fig.figure.savefig(fig_name + timenow() + '.png', format='png', dpi=dpi)

	"""
	Parameters:

	method 		= str  	: 	Method for plottting correlation matrix (default = 'pearson') Available methods = 'perason', 'kendall', or 'spearman'  
	matrix_type	= bool 	:	Type of correlation-matrix for plotting  (default = upper); Available = 'full', 'upper', 'lower'
	annot		= bool 	:	Whether to show the correlation with numbers or not  (default = False)
	cmap 		= any  	: 	Color map for plot  (default = coolwarm)
	vmin		= float	:	Minimum value for color bar (default = -1.0)
	vmax		= float	:	Maximum value for color bar (default =  1.0)
	figsize 	= tuple : 	Tuple of two integers for determining the figure size    (default =(16, 12))
	fontsize 	= int  	:	Font size of color-bar and x, y axis   (default =14)
	save_fig 	= bool 	: 	save plot in the current working directory if True  (default = False)
	save_csv 	= bool 	: 	save a csv file if True  (default = False)
	figname 	= str   :	name of fig if save_fig is True  (default = "Correlation_plot.png")

	"""

	def plot_feature_imp(self, kind="barh", random_no=None, figsize=(22, 16), fontsize=20, color='#ff8000', lw=5.0,
						 save_fig=False, fig_name="Feature_imp_Plot(MI).png", dpi=600):
		global imp_fig

		MI = mutual_info_regression(X_train, y_train, random_state=random_no)
		MI = pd.Series(MI)
		MI.index = X_train.columns
		MI.sort_values(ascending=True, inplace=True)
		if self.verbosity in [1, 2]:
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
	def plot_scatter(plot_for="Test", facecolor='red', alpha=0.5, marker='o', xlabel='True', ylabel='Predicted', title='Regression_plot',
					 save_fig=True, fig_name="Scatter_plot", dpi=600):

		if plot_for is 'Train':
			ax = sns.regplot(y_train, y_train_predicted, marker=marker)
			plt.xlabel(xlabel)
			plt.ylabel(ylabel)
			plt.title(title)
			ax.text(max(y_train/2), min(y_train*2.5), ('R2 =', round(r2_score(y_train, y_train_predicted) , 3)), bbox=dict(facecolor=facecolor, alpha=alpha))
			plt.show()
		elif plot_for is 'Test':
			ax  = sns.regplot(y_test, y_test_predicted, marker=marker)
			plt.xlabel(xlabel)
			plt.ylabel(ylabel)
			plt.title(title)
			ax.text(max(y_test / 2), min(y_test * 2.5), ('R2 =', round(r2_score(y_test, y_test_predicted) , 3)),
					bbox=dict(facecolor=facecolor, alpha=alpha))
			plt.show()
		else:
			Name_error = "{} is not available. Available options are 'Train', or 'Test'".format(plot_for)
			raise NameError(Name_error)

		if save_fig:
			ax.figure.savefig(fig_name + timenow() + '.png', format='png', dpi=dpi)

	"""
	Parameters:
	facecolor='red', alpha=0.5, xlabel='True', ylabel='Predicted', title='Regression_plot',
					 save_fig=True, dpi=600
					 
	return_train = bool 	: Scatter plot for the training dataset 
	facecolor	 = str		:	
	alpha		 = float	: Determine the intensity of colors
	xlabel		 = bool		: Label for x-axis
	ylabel		 = bool		: Label for y-axis
	title		 = str		: Title of the figure
	save_fig	 = bool		: Name of the file to save figure
	dpi			 = int		: Determine the quality of the figure to save
		
	"""

	def save_data(self, filename=None, verbosity=0):
		global save_content

		if verbosity is 0:
			save_content = {}
			save_content['inputs']=list(self.inputs)
			save_content['target']=list(self.target)
			save_content['scoring_method']=_scoring
			save_content['problem']=self.problem
			save_content['model_name']=_model_name
			if _optimization is not None:
				save_content['best_parameters']=best_params
			else:
				pass
			save_content['Train_score']=Train_score
			save_content['Test_score']=Test_score


		elif verbosity in [1,2]:
			save_content = {}
			save_content['inputs']=list(self.inputs)
			save_content['target']=list(self.target)
			save_content['random_state']=self.random_state
			save_content['test_size']=self.test_size
			save_content['scaling_method']=self.normalization
			save_content['verbosity']=self.verbosity
			save_content['optimization_method']=_optimization
			save_content['problem'] = self.problem,
			save_content['model_name']=_model_name
			if _optimization is not None:
				save_content['best_parameters']=best_params
			else:
				pass
			save_content['scoring_method']=_scoring
			save_content['Train_score']=Train_score
			save_content['Test_score']=Test_score


		json_converted = json.dumps(save_content, indent=5)
		print(json_converted)

		if filename is None:
			save_file = open("results" + timenow() + '.txt', "w")
			save_file.write(json_converted)
			save_file.close()
		else:
			save_file = open(filename + timenow() + '.txt', "w")
			save_file.write(json_converted)
			save_file.close()