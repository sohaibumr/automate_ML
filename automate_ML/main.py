import scipy
import pandas as pd
import numpy as np
import warnings
""""                                Classification Models                                   """

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, RidgeClassifierCV
from sklearn.linear_model._stochastic_gradient import SGDClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import ExtraTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression


"""                                 Pre-processing tools                                     """

from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedShuffleSplit, GridSearchCV, KFold, RandomizedSearchCV
from collections import Counter
from sklearn import preprocessing
from datetime import datetime
from skopt import BayesSearchCV

"""                                     Metrices                                            """

from sklearn import metrics
from sklearn.metrics import mean_squared_error, roc_auc_score, roc_curve, auc, accuracy_score, confusion_matrix

"""                                   Plotting tools                                        """

from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from matplotlib import axes
from sklearn.metrics import auc, plot_roc_curve
import matplotlib.colors as cl
from tabulate import tabulate
global clf, _params, clf_fit, ss, clf_cv, df, self_prediction, prediction, probability, unknown_prediction, cf_matrix, X_train, X_test, y_train, y_test, test_prediction, train_prediction, figsav, Training_scores_mean, Training_scores_std, Test_scores_mean, Test_scores_std, df_unknown


"""""""""""""""""""""""""""""""""""""""""""""""""Classifiers"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class Classification:

	def __init__(self, data=None,   inputs=None, target=None, nan_values=None,  normalization=None, test_size=0.20, random_state=315,
				 return_dataset=None, verbosity=0):
		global X_train, X_test, y_train, y_test
		warnings.filterwarnings("ignore")

		if data is None:
			raise NameError ("'data' is not defined. Define a dataframe.")
		elif data.isnull().values.any():
			if nan_values == 'impute':
				data.fillna(data.mean(), inplace=True)
				self.data=data
			elif nan_values == 'remove':
				data.dropna(inplace=True)
				data.reset_index(drop=True)
				self.data=data
			elif nan_values is None:
				raise ValueError("'data' contain NaN values, remove or impute the NaN values")
			else:
				error = "'nan_values' can either be 'impute' or 'remove' and you passed '{}'".format(nan_values)
				raise NameError(error)
		else:
			self.data=data

		if inputs is None:
			raise NameError ("'inputs' is not defined")
		else:
			if verbosity in [1, 2]:
				print('\n', '\033[0;30;44m' + '---> Unscaled Data' + '\033[0m')
				print(self.data[inputs], '\n')
			if normalization == 'zscore':
				if verbosity in [1,2]:
					print('\n', '\033[0;30;44m' + '---> Data Scaled with zscore' + '\033[0m')
				norm_data = zscore(self.data[inputs])
				self.inputs= pd.DataFrame(norm_data, columns=self.data[inputs].columns)
			elif normalization == 'minmax':
				if verbosity in [1,2]:
					print('\n', '\033[0;30;44m' + '---> Data Scaled with minmax' + '\033[0m')
				minmax_data = MinMaxScaler()
				norm_data= minmax_data.fit_transform(self.data[inputs])
				self.inputs = pd.DataFrame(norm_data, columns=self.data[inputs].columns)
			elif normalization is None:
				self.inputs = (self.data[inputs] - self.data[inputs].mean()) / self.data[inputs].std()
			if verbosity == 2:
				print(self.inputs, '\n')

		if target is None:
			raise NameError ("'target' is not defined")
		else:
			self.target = self.data[target]

		self.normalization = normalization
		self.verbosity=verbosity
		self.test_size=test_size
		self.random_state=random_state

		X_train, X_test, y_train, y_test = train_test_split(self.inputs, self.target, test_size=self.test_size,
															stratify=self.target, random_state=self.random_state)
		if return_dataset == 'train':
			df_train = pd.DataFrame(y_train)
			df_train.index = y_train.index
			df_train.to_csv('Train_data' + '_' + self.timenow + '.csv')
		elif return_dataset == 'test':
			df_test = pd.DataFrame(y_test)
			df_test.index = y_test.index
			df_test.to_csv('Test_data' + '_' + self.timenow + '.csv')
		elif return_dataset is None:
			pass
		else:
			error = "return_dataset can either be 'train' or 'test' and you passed '{}'".format(return_dataset)
			raise NameError(error)

		if self.verbosity in [1,2]:
			print("target", Counter(self.target))
			print("y_train:", Counter(y_train), ',',  "y_test:", Counter(y_test), '\n')
		else:
			pass

		"""
		Parameters:
	
		data =  Dataframe		: 	Dataset for evaluating a model  (default = None)
		inputs = Dataframe		:	Feature set (default = None)
		target = Dataframe		: 	Target which you want to predict  (default = None)
		nan_values = str		:	Whether to 'impute' or 'remove' NaN value in the dataset. (default=None)	
		normalization = str 	:	Method for normalizing the dataset (default = "None")
		test_size = float		:	Size od testing dataset (default = 0.20)
		random_state = int		:	random number for the reproducing the results (deafult = 315)
		return_dataset = str	:	Dataset to be returned as a csv file. (default = None)
		verbosity = integer		:	Degree for printing output messages in the terminal (default = 0, can be 0,1, or 2)
	
		"""

	@property
	def timenow(self):
		T = str(datetime.now())
		t = T.replace(':', '')
		return t

	@property
	def model_names(self):
		model_names = ['AdaBoost', 'Bagging', 'CalibratedCV', 'CatBoost', 'DecisionTree', 'ExtraTrees', 'ExtraTree',
					   'GradientBoosting', 'KNeighbors', 'LogisticReg', 	'LinearDA', 'LGBM', 'Linear_SVC', 'Mlp',
					   'NUSvc', 'RandomForest', 'RadiusNeighbor', 'RidgeCV', 'Ridge', 'Svc', 'SGDC' ]

		return print(model_names)


	@staticmethod
	def AdaBoost(params=None):
		global clf, _params
		clf = AdaBoostClassifier()
		print('\033[1;32;40m' + "---> Classification model = 'AdaBoost'" + '\033[0m', '\n')
		if params is None:
			_params = {'n_estimators': [10, 50, 100], 'learning_rate': [0.001, 0.04, 0.05, 0.09, 0.1], 'random_state':[315]}
		else:
			_params = params


	@staticmethod
	def Bagging(params=None):
		global clf, _params
		clf = BaggingClassifier()
		print('\033[1;32;40m' + "---> Classification model = 'Bagging'" + '\033[0m', '\n')
		if params is None:
			_params = {'n_estimators': [10, 50, 100, 250, 500], 'max_samples': [0.1, 1.0], 'random_state':[315]}
		else:
			_params = params


	@staticmethod
	def CalibratedCV(params=None):
		global clf, _params
		clf = CalibratedClassifierCV()
		print('\033[1;32;40m' + "---> Classification model = 'Calibrated Classifier CV'" + '\033[0m', '\n')
		if params is None:
			_params = {'method': ['sigmoid', 'isotonic'], 'cv': [3, 5]}
		else:
			_params = params


	@staticmethod
	def CatBoost(params=None):
		global clf, _params
		clf = CatBoostClassifier()
		print('\033[1;32;40m' + "---> Classification model = 'CatBoost'" + '\033[0m', '\n')
		if params is None:
			_params = {'iterations': [10, 20, 30], 'learning_rate': [0.001, 0.01], 'border_count': [100, 200], 'feature_border_type': ['GreedyLogSum'], 'random_state':[315], 'verbose' : [0]}
		else:
			_params = params


	@staticmethod
	def DecisionTree(params=None):
		global clf, _params
		clf = DecisionTreeClassifier()
		print('\033[1;32;40m' + "---> Classification model = 'Decision Trees'" + '\033[0m', '\n')
		if params is None:
			_params = {'max_depth': [2, 5, 10], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 5, 10], 'random_state':[315]}
		else:
			_params = params


	@staticmethod
	def ExtraTrees(params=None):
		global clf, _params
		clf = ExtraTreesClassifier()
		print('\033[1;32;40m' + "---> Classification model = 'ExtraTrees Classifier'" + '\033[0m', '\n')
		if params is None:
			_params = {'n_estimators': [100, 500],  'max_depth': [2, 5], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 5, 10], 'random_state': [315]}
		else:
			_params = params


	@staticmethod
	def ExtraTree(params=None):
		global clf, _params
		clf = ExtraTreeClassifier()
		print('\033[1;32;40m' + "---> Classification model = 'Extra Tree Classifier'" + '\033[0m', '\n')
		if params is None:
			_params = {'max_depth': [2, 5], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 5, 10], 'random_state': [315]}
		else:
			_params = params


	@staticmethod
	def GradientBoosting(params=None):
		global clf, _params
		clf = GradientBoostingClassifier()
		print('\033[1;32;40m' + "---> Classification model = 'Gradient Boosting'" + '\033[0m', '\n')
		if params is None:
			_params = {'learning_rate':[0.001, 0.02, 0.04, 0.09, 0.1], 'max_depth': [2, 5, 10], 'n_estimators': [100, 500, 1000], 'min_samples_split': [2, 5, 10], 'random_state':[315]}
		else:
			_params = params


	@staticmethod
	def KNeighbors(params=None):
		global clf, _params
		clf = KNeighborsClassifier()
		print('\033[1;32;40m' + "---> Classification model = 'KNeighbors'" + '\033[0m', '\n')
		if params is None:
			_params = {'n_neighbors': [5, 7, 10, 15, 20, 22, 30, 35], 'weights': ['uniform', 'distance'], 'p': [1, 2]}
		else:
			_params = params


	@staticmethod
	def LogisticReg(params=None):
		global clf, _params
		clf = LogisticRegression()
		print('\033[1;32;40m' + "---> Classification model = 'Logistic Regression'" + '\033[0m', '\n')
		if params is None:
			_params = {'C': [0.1, 1, 10], 'class_weight': ['balanced'], 'max_iter': [500, 1000, 1500], 'random_state' : [315], 'solver' : ['lbfgs', 'liblinear']}
		else:
			_params = params


	@staticmethod
	def LinearDA(params=None):
		global clf, _params
		clf = LinearDiscriminantAnalysis()
		print('\033[1;32;40m' + "---> Classification model = 'Linear Discrimination Analysis'" + '\033[0m', '\n')
		if params is None:
			_params = {'solver': ['svd', 'lsqr', 'eigen']}
		else:
			_params = params


	@staticmethod
	def LGBM(params=None):
		global clf, _params
		clf = LGBMClassifier()
		print('\033[1;32;40m' + "---> Classification model = 'Light Gradient Boosting Machine'" + '\033[0m', '\n')
		if params is None:
			_params = {'boosting_type': ['gbdt'], 'num_leaves': [500, 1000, 1500, 2000], 'learning_rate': [0.001, 0.01, 0.1], 'n_estimators': [100, 200, 300, 500], 'random_state':[315]}
		else:
			_params = params


	@staticmethod
	def Linear_SVC(params=None):
		global clf, _params
		clf = LinearSVC()
		print('\033[1;32;40m' + "---> Classification model = 'Linear SVC'" + '\033[0m', '\n')
		if params is None:
			_params = {'penalty': ['l2'], 'dual': [False], 'C': [0.1, 1.0, 10.0], 'class_weight': ['balanced'], 'random_state': [315], 'max_iter': [100, 200, 500]}
		else:
			_params = params


	# @staticmethod
	# def Mlp(params=None):
	# 	global clf, _params
	# 	clf = MLPClassifier()
	# 	print('\033[1;32;40m' + "---> Classification model = 'Multi-layer Perceptron'" + '\033[0m', '\n')
	# 	if params is None:
	# 		_params = {'hidden_layer_sizes':[(4, 8)],'learning_rate_init': [0.01, 0.03, 0.05], 'max_iter': [250], 'activation':['relu', 'identity'],
	# 							'batch_size':['auto',32, 64], 'solver':['adam', 'lbfgs', 'sgd'], 'random_state': [315]}
	# 	else:
	# 		_params = dict(params)


	@staticmethod
	def NUSvc(params=None, proba=False):
		global clf, _params
		clf = NuSVC()
		print('\033[1;32;40m' + "---> Classification model = 'Nu-Support Vector'" + '\033[0m', '\n')
		if params is None:
			_params = {'kernel': ['rbf', 'poly'], 'gamma': ['auto', 'scale'], 'probability':[proba], 'random_state':[315]}
		else:
			_params = params


	@staticmethod
	def RandomForest(params=None):
		global clf, _params
		clf = RandomForestClassifier()
		print('\033[1;32;40m' + "---> Classification model = 'Random Forest'" + '\033[0m', '\n')
		if params is None:
			_params = {'n_estimators': [500, 1000, 1200, 1500], 'max_depth': [2, 3, 4, 5, 6, 8, 10], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 5, 10], 'random_state': [315]}
		else:
			_params = params


	@staticmethod
	def RadiusNeighbor(params=None):
		global clf, _params
		clf = RadiusNeighborsClassifier()
		print('\033[1;32;40m' + "---> Classification model = 'Radius Neighbors'" + '\033[0m', '\n')
		if params is None:
			_params = {'radius': [21, 25, 27, 29, 31, 35], 'weights': ['uniform', 'distance'], 'p': [1, 2]}
		else:
			_params = params


	@staticmethod
	def RidgeCV(params=None):
		global clf, _params
		clf = RidgeClassifierCV()
		print('\033[1;32;40m' + "---> Classification model = 'Ridge CV Classifier'" + '\033[0m', '\n')
		if params is None:
			_params = {'alphas': [(0.1, 1.0, 5.0, 10.0)], 'scoring': ['accuracy'], 'cv':[5], 'class_weight': ['balanced']}
		else:
			_params = params

	@staticmethod
	def Ridge(params=None):
		global clf, _params
		clf = RidgeClassifier()
		print('\033[1;32;40m' + "---> Classification model = 'Ridge Classifier'" + '\033[0m', '\n')
		if params is None:
			_params = {'alpha': [0.1, 1.0, 5.0, 10.0], 'max_iter': [100, 200, 500], 'class_weight': ['balanced'], 'solver': ['auto'], 'random_state':[315]}
		else:
			_params = params


	@staticmethod
	def Svc(params=None, proba=False):
		global clf, _params
		clf = SVC()
		print('\033[1;32;40m' + "---> Classification model = 'Support Vector'" + '\033[0m', '\n')
		if params is None:
			_params = {'gamma': [1.0, 0.1, 0.01, 0.001], 'C': [0.1, 1.0, 10.0], 'class_weight':['balanced'], 'probability':[proba], 'random_state':[315]}
		else:
			_params = params



	@staticmethod
	def SGDC(params=None):
		global clf, _params
		clf = SGDClassifier()
		print('\033[1;32;40m' + "---> Classification model = 'Stochastic Gradient Descent'" + '\033[0m', '\n')
		if params is None:
			_params = {'loss': ['log', 'modified_huber'], 'penalty':['l1','l2','elasticnet'], 'max_iter':[100, 200, 500], 'class_weight':['balanced'], 'random_state':[315]}
		else:
			_params = params


##############################################******Fit Function********################################################


	def fit(self, optimization='Grid', num_iter=20, cv=10, scoring='roc_auc'):
		global clf_fit, clf_cv, Training_scores_mean, Training_scores_std, Test_scores_mean, Test_scores_std
		scoring_methods = ['roc_auc', 'roc_auc_ovr', 'roc_auc_ovo', 'accuracy', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted', 'f1', 'recall', 'precision']

		if scoring in scoring_methods:
			pass
		else:
			score_error = "{} is not a valid scoring method. Valid scoring methods are following: 'roc_auc', 'roc_auc_ovr','roc_auc_ovo', 'accuracy', 'roc_auc_ovr_weighted', 'f1', 'recall', 'precision'".format(scoring)
			raise NameError(score_error)

		if optimization == 'Grid':
			clf_cv = GridSearchCV(clf, _params, cv=cv, scoring=scoring)
		elif optimization == 'Randomized':
			clf_cv = RandomizedSearchCV(clf, _params, n_iter=num_iter, cv=cv, scoring=scoring, random_state=self.random_state)
		elif optimization == 'Bayesian':
			clf_cv = BayesSearchCV(clf, _params, scoring=scoring, n_iter=num_iter, cv=cv, random_state=self.random_state)
		elif optimization is None:
			clf_cv = clf
		else:
			search_error = "{} is not a valid option for hyper-paramteres optimization. Available options are 'Grid', 'Randomized' or 'Bayesian'".format(optimization)
			raise ValueError (search_error)


		clf_fit = clf_cv.fit(X_train, y_train)
		print("-----------------------------------------------------------------------------------------------------")
		print('\033[1;32;40m' + 'Best estimator:' + '\033[0m', clf_cv.best_estimator_)
		print('\033[1;32;40m' + 'Best parameters:' + '\033[0m', clf_cv.best_params_)
		print("-----------------------------------------------------------------------------------------------------", '\n')

		# Training_score
		Training_score = cross_val_score(clf_cv.best_estimator_, X_train, y_train, cv=cv, scoring=scoring)
		Training_scores_mean = Training_score.mean()
		Training_scores_std = Training_score.std()

		# Test_score
		Test_score = cross_val_score(clf_cv.best_estimator_, X_test, y_test, cv=cv, scoring=scoring)
		Test_scores_mean = Test_score.mean()
		Test_scores_std = Test_score.std()


		print('\033[1;32;40m' + "**************Train_score**************" + '\033[0m')
		print("{} = ".format(scoring), round(Training_scores_mean, 4), '+/-', round(Training_scores_std, 4) * 2, 'std', round(Training_scores_std, 4))
		print('\033[1;32;40m' + "**************Test_score***************" + '\033[0m')
		print("{} = ".format(scoring),  round(Test_scores_mean, 4), '+/-', round(Test_scores_std, 4) * 2, 'std', round(Test_scores_std, 4), '\n')


		"""
		Parameters:
	
		optimization 		= str		:	Method for searching the best hyperparameters for the model  (default = 'Grid'); Available methods are = 'Grid', 'Bayesian' and 'Randomized'
		num_iterations		= int		:	Number of iterations to run for hyperparameter optimization (default = 20)
		cv 			 		= int		:	cross-validation (default = 10)
		scoring 	 		= str  		:	Method for the evaluation of model: (default = 'roc_auc'); Available methods are = ['roc_auc', 'roc_auc_ovr', 'roc_auc_ovo', 'accuracy', 'roc_auc_ovr_weighted', 'roc_auc_ovr_weighted', 'f1', 'precision', 'recall']
	
		"""

##############################################******Make Predictions********############################################

	def make_prediction(self, prediction_data= 'test', unknown_data=None, proba_prediction=False, save_csv=False, file_name= 'predicted_data'):
		global prediction, df, unknown_prediction, test_prediction, train_prediction, df_unknown, probability
		prediction=prediction_data
		probability=proba_prediction
		if prediction_data is None:
			raise AssertionError("prediction_data is not defined")
		elif prediction_data == 'test':
			if proba_prediction:
				test_prediction = clf_fit.predict_proba(X_test)[:,1]
			else:
				test_prediction = clf_fit.predict(X_test)
				print('\033[1;32;40m' + '*******Prediction_score on testing data*******' + '\033[0m')
				print("roc_auc_score", round(roc_auc_score(y_test, test_prediction), 4), '\n')

			if self.verbosity == 2:
				for i in range(len(X_test)):
					print(y_test[i], test_prediction[i])
			else:
				pass
			if save_csv:
				predicted_test_data = {'true':y_test, 'predicted':test_prediction}
				df = pd.DataFrame(predicted_test_data)
				df.index = X_test.index
				df.to_csv(file_name + '_' + self.timenow + '.csv')

		elif prediction_data == 'train':
			if proba_prediction:
				train_prediction = clf_fit.predict_proba(X_train)[:,1]
			else:
				train_prediction = clf_fit.predict(X_train)
				print('\033[0;30;44m' + '*******Prediction_score on training data*******' + '\033[0m')
				print("roc_auc_score", round(roc_auc_score(y_train, train_prediction), 4))


			if self.verbosity == 2:
				for i in range(len(X_train)):
					print(y_train[i], train_prediction[i])
			else:
				pass
			if save_csv:
				predicted_train_data = {'true': y_train, 'predicted': train_prediction}
				df = pd.DataFrame(predicted_train_data)
				df.index = self.inputs.index
				df.to_csv(file_name + '_' + self.timenow + '.csv')

		elif prediction_data == 'unknown':
			if unknown_data is None:
				raise NameError("'unknown_data' is not defined. Define an unknown dataset")
			else:
				if self.normalization == 'zscore':
					norm_data = zscore(unknown_data)
					df_unknown = pd.DataFrame(norm_data, columns=unknown_data.columns)
				if self.normalization == 'minmax':
					minmax_data = MinMaxScaler()
					norm_data = minmax_data.fit_transform(unknown_data)
					df_unknown = pd.DataFrame(norm_data, columns=unknown_data.columns)
				elif self.normalization is None:
					df_unknown = (unknown_data - unknown_data.mean()) / unknown_data.std()
			if proba_prediction:
				unknown_prediction = clf_fit.predict_proba(df_unknown)
			else:
				unknown_prediction = clf_fit.predict(df_unknown)
			if self.verbosity == 2:
				print([unknown_prediction])
			else:
				pass
			if save_csv:
				df = pd.DataFrame(unknown_prediction, columns=['pred'])
				df.index = df_unknown.index
				df.to_csv(file_name + '_' + self.timenow + '.csv')


		"""
		Parameters:
	
		prediction_data		= bool		:	Dataset to make predictions (default = 'test')
		unknown_data		= Dataframe	:	Unknown dataset for predictions; required when prediction_data is 'unknown' (default = None)
		proba_prediction	= bool		:	Predict probabilities rather than the exact values for the target if set True (default = False)
		save_csv	 		= bool		:	Save a csv file of predictions if set True (default = False)
		file_name	 		= str		:	Name for the csv file (default = 'predicted_data')
		
		"""

	def Confusion_matrix(self, show_plot=True, annot=True, cmap='Blues', figsize=(16, 12), fontsize=14,
						 save_fig=False, fig_name="Confusion_matrix.png"):

		global cf_matrix
		if prediction == 'train' and probability is False:
			print("Confusion_matrix for training dataset")
			cf_matrix=confusion_matrix(y_train, train_prediction)
			print(cf_matrix)
		elif prediction == 'train' and probability is True:
			raise AssertionError("Confusion matrix is not available for probabilities")
		elif prediction == 'test' and probability is False:
			print("Confusion_matrix for testing dataset")
			cf_matrix=confusion_matrix(y_test, test_prediction)
			print(cf_matrix)
		elif prediction == 'test' and probability is True:
			raise AssertionError("Confusion matrix is not available for probabilities")


		ax = sns.heatmap(cf_matrix, annot=annot, cmap=cmap,  linewidths=0.5)

		plt.figure(figsize=figsize)
		plt.rcParams['font.size'] = fontsize
		ax.set_title('Seaborn Confusion Matrix with labels\n\n')
		ax.set_xlabel('Predicted Values')
		ax.set_ylabel('Actual Values ')

		ax.xaxis.set_ticklabels(['False', 'True'])
		ax.yaxis.set_ticklabels(['False', 'True'])

		if show_plot:
			plt.show()
		else:
			pass
		if save_fig:
			ax.figure.savefig(fig_name + self.timenow + '.png', format='png', dpi=600)

		"""
		Parameters:
	
		show_plot	= bool	: 	Visualize confusion matrix if set True (default = False)  
		annot		= bool 	:	Print the confusion matrix values in the heatmap if set True  (default = False)
		cmap 		= any  	: 	Color map for plot  (default = 'Blues')
		figsize 	= tuple : 	Tuple of two integers for determining the figure size    (default =(16, 12))
		fontsize 	= int  	:	Font size of color-bar and x, y axis   (default =14)
		save_fig 	= bool 	: 	Save plot in the current working directory if True  (default = False)
		figname 	= str   :	Name of fig if save_fig is True  (default = "Correlation_plot.png")
	
		"""

##############################################******Visualization********###############################################


	def plot_correlation(self, method='pearson', matrix_type='upper', annot=False, cmap='coolwarm',  vmin=-1.0, vmax=1.0,
						  figsize=(16, 12), fontsize=14, save_fig=False, save_csv=False, fig_name="Correlation_plot.png"):
		global figsav

		methods = ['pearson', 'kendall', 'spearman']
		if method not in methods:
			method_error = "{} is not a valid method. Valid methods are 'pearson','kendall', and 'spearman'".format(method)
			raise TypeError(method_error)

		# for inputs in self.inputs:
		df_ = pd.DataFrame(self.inputs)
		correlation = df_.corr(method=method, min_periods=1)

		if matrix_type == 'full':
			plt.figure(figsize=figsize)
			plt.rcParams['font.size'] = fontsize
			figsav = sns.heatmap(correlation, annot=annot, cmap=cmap, linewidths=0.5,
									 vmax=vmax, vmin=vmin)
			plt.show()

		elif matrix_type == 'upper':
			mask_ut = np.tril(np.ones(correlation.shape)).astype(np.bool)
			plt.figure(figsize=figsize)
			plt.rcParams['font.size'] = fontsize
			figsav = sns.heatmap(correlation, mask=mask_ut, annot=annot, cmap=cmap, linewidths=0.5,
									 vmax=vmax, vmin=vmin)
			plt.show()

		elif matrix_type == 'lower':
			mask_ut = np.triu(np.ones(correlation.shape)).astype(np.bool)
			plt.figure(figsize=figsize)
			plt.rcParams['font.size'] = fontsize
			figsav = sns.heatmap(correlation, mask=mask_ut, annot=annot, cmap=cmap, linewidths=0.5,
									 vmax=vmax, vmin=vmin)
			plt.show()

		else:
			raise TypeError("{} is not a valid matrix_type. Available matrix_type are 'full', 'upper', or 'lower'".format(matrix_type))

		df_corr = pd.DataFrame(correlation)
		if save_csv:
			df_corr.to_csv("Correlation.csv")

		if save_fig:
			figsav.figure.savefig(fig_name+self.timenow+'.png', format='png', dpi=600)


		"""
		Parameters:
		
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
		
		"""

	def plot_feature_imp(self, kind="barh", random_no=None, figsize=(22,16), fontsize=20, color='#ff8000', lw=5.0, save_fig=False, fig_name="Feature_imp_Plot(MI).png"):
		global figsav
		MI = mutual_info_classif(X_train, y_train, random_state=random_no)
		MI = pd.Series(MI)
		MI.index = X_train.columns
		MI.sort_values(ascending=True, inplace=True)
		if self.verbosity in [1, 2]:
			print(MI)
		else:
			pass

		plot_kind=['barh', 'bar', 'pie', 'line', 'area']
		if kind in plot_kind:
			if kind == "pie":
				figsav = MI.plot(kind=kind, normalize=False)
				plt.show()
			else:
				figsav = MI.plot(kind=kind, figsize=figsize, fontsize=fontsize, color=color, lw=lw)
				plt.show()
		else:
			error="{} is not a valid type for plotting feature importance. Only 'barh', 'bar', 'pie', 'line', 'area' can be used for plotting".format(kind)
			raise TypeError(error)

		if save_fig:
			figsav.figure.savefig(fig_name+self.timenow+'.png', format='png', dpi=600)


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


	def plot_roc_curve(self, figsize=(9, 7), lines_fmt=None, label='ROC_curve', fontsize=18, ticksize=18, save_fig=False, fig_name='roc_plot'):

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
		y_proba = clf_fit.predict_proba(X_test)[::,1]
		fpr, tpr, _ = roc_curve(y_test, y_proba)
		# auc = roc_auc_score(y_test, y_proba)
		plt.legend(loc=4)
		fig, ax = plt.subplots(figsize=figsize)
		plt.rc('font', family='serif')
		ax.plot(fpr, tpr, label=label, lw=linewidth, alpha=.8)
		ax.plot(fpr, fpr, linestyle='--', lw=linewidth, label='Chance', alpha=.8)

		ax.set(xlim=[-0.02, 1.02], ylim=[-0.02, 1.02])
		ax.legend(loc="lower right")
		ax.set_xlabel('False positive rate', fontsize=labelsize)
		ax.set_ylabel('True positive rate', fontsize=labelsize)
		plt.xticks(fontsize=ticksize)
		plt.yticks(fontsize=ticksize)

		plt.legend(fontsize='x-large')

		for axis in ['top', 'bottom', 'left', 'right']:
			ax.spines[axis].set_linewidth(1.6)
			ax.spines[axis].set_color('dimgrey')
		plt.show()

		if save_fig:
			fig.savefig(fig_name+self.timenow+'.png', format='png', dpi=600)


		"""
		Parameters:
	
		figsize 	= tuple 	: 	Tuple of two integers for determining the figure size  (default =(9, 7))		 
		lines_fmt 	= dict		: 	Dictionary for the formatting of lines i.e. 'color' and linewidth('lw')	 (default = {'color': ["#339966", "#cc0000"], 'lw': 3}
		label 		= str		:	Set label inside the plot (default = 'ROC_curve')
		fontsize 	= int 		: 	Set fontsize for the x and y labels  (default = 18)
		ticksize 	= int 		:	Set fontsize for the x and y ticks   (default = 18)
		save_fig 	= bool 		: 	Save Figure in the current directory if set True    (default = False)
		fig_name 	= str  		: 	Name for the figure     (default = 'roc_plot')
	
		"""

class Models(Classification):

	def best_model(self, n_splits=100, test_size=0.20, random_state=None, scoring='roc_auc', save_txt=True, filename='Models_score', show=True):

		warnings.filterwarnings("ignore")
		print('\n', '\033[0;30;44m' + "********************* Evaluation Started *********************" + '\033[0m', '\n')
		models = []
		models.append(('AdaB', AdaBoostClassifier()))
		models.append(('Bag', BaggingClassifier()))
		models.append(('CalCV', CalibratedClassifierCV()))
		models.append(('DT', DecisionTreeClassifier()))
		models.append(('ETs', ExtraTreesClassifier()))
		models.append(('ET', ExtraTreeClassifier()))
		models.append(('GB', GradientBoostingClassifier()))
		models.append(('KNN', KNeighborsClassifier()))
		models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
		models.append(('LDA', LinearDiscriminantAnalysis()))
		models.append(('LGBM', LGBMClassifier()))
		models.append(('Lin_SVC', LinearSVC()))
		models.append(('Nusvc', NuSVC(nu=0.1)))
		models.append(('RF', RandomForestClassifier()))
		models.append(('RC', RidgeClassifier()))
		models.append(('RC_CV', RidgeClassifierCV()))
		models.append(('SGDC', SGDClassifier()))
		models.append(('SVC', SVC(gamma='auto')))



		results = []
		names = []
		for name, model in models:
			ss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
			cv_results = cross_val_score(model, self.inputs, self.target, cv=ss, scoring=scoring)
			results.append(cv_results.mean())
			names.append(name)
			print("%s: score=%f std=(%f)" % (name, cv_results.mean(), cv_results.std()))



		result = pd.Series(results)
		result.index = names
		result.sort_values(ascending=False, inplace=True)
		print('\n', '\033[0;30;44m' + "**************** Models evaluation has been Completed ****************" + '\033[0m')


		if show:
			print('\n', '\033[0;30;44m' + 'Sorted Models w.r.t score' + '\033[0m')
			result_tab = result.to_frame()
			tabulated_results = tabulate(result_tab, headers =['Models', 'Score'], tablefmt='fancy_grid')
			print(tabulated_results)
			# plot
			result.plot(kind='bar')
			plt.show()
		else:
			pass
		if save_txt:
			result.to_csv(filename+'.csv')

		"""
		Parameters:

		n_splits 	 = int 		: 	No of splits  (default =100)		 
		test_size 	 = float	: 	Fraction of datset to be chosen for testing	 (default = 0.20)
		random_state = int		:	Any random no to reproduce the results (default = None)
		scoring 	 = str 		: 	Scoring method  (default = 'roc_auc')
		save_txt 	 = bool 	:	Save a txt files with model names and corresponding scores   (default = True)
		filename 	 = str 		: 	Name of the txt file   (default = 'Models_score')
		show 		 = bool  	: 	Print out the sorted table and plots a bar chart of the models with corresponding scores if set True   (default = True)

		"""