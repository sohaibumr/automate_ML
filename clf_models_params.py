
from clf_models_dependencies import *
global clf, params, clf_fit, ss, clf_cv, df, self_prediction, unknown_prediction, X_train, X_test, y_train, y_test, test_prediction, train_prediction, figsav, Training_scores_mean, Training_scores_std, Test_scores_mean, Test_scores_std
"""""""""""""""""""""""""""""""""""""""""""""""""Classifiers"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


class do_classification:

	def __init__(self, data=None, inputs=None, target=None,  normalization=None, predict_unknown=False, unknown_data=None,
				 correlation=False, feature_imp=False,  ROC_curve=False, verbosity=0):


		self.data = data
		column = inputs.columns

		if inputs is None:
			raise NameError ("'inputs' is not defined")
		else:
			if normalization == 'zscore':
				if verbosity in [1,2]:
					print('\n', '---> Data is Scaled with zscore')
				norm_data = zscore(inputs)
				self.inputs= pd.DataFrame(norm_data, columns=column)
			if normalization == 'minmax':
				if verbosity in [1,2]:
					print('---> Data is Scaled with minmax')
				minmax_data = MinMaxScaler()
				norm_data= minmax_data.fit_transform(inputs)
				self.inputs = pd.DataFrame(norm_data, columns=column)
			if normalization is None:
				self.inputs = (inputs - inputs.mean()) / inputs.std()
			if verbosity == 2:
				print(self.inputs, '\n')

		if target is None:
			raise NameError ("'target' is not defined")
		else:
			self.target = target

		self.normalization = normalization
		self.predict_unknown = predict_unknown
		self.unknown_data = unknown_data
		self.correlation = correlation
		self.feature_imp = feature_imp
		self.ROC_curve = ROC_curve
		self.verbosity = verbosity

	"""
	Parameters:

	data =  any			: 	Dataset for evaluating a model  (default = None)
	inputs = any		:	Feature set (default = None)
	target = any		: 	Target which you want to predict  (default = None)
	normalization: any  :	Method for normalizing the dataset (default = "None"
	predict_unknown = 	:	Set True if want to make prediction for 'unknown_data' (default = False)	
	unknown_data = any  :	Dataset to make prediction  (default = None)
	correlation = bool	:	Plots correlation heatmap if True (default = False)
	feature_imp = bool	:	Plots feature importance plot using Mutual information method (MI) if True (deafult = False)
	ROC_curve = bool	:	Plots roc curve if True (default = False)
	verbosity = integer	:	Degree for printing output messages in the terminal (default = 0, can be 0,1, or 2)

	"""

	@property
	def timenow(self):
		T = str(datetime.now())
		t = T.replace(':', '')
		return t


	@staticmethod
	def ExtraTrees(_params=None):
		global clf, params
		clf = ExtraTreesClassifier()
		print("---> Classification model = 'ExtraTrees Classifier'", '\n')
		if _params is None:
			params = {'n_estimators': [100, 500],  'max_depth': [2, 5], 'random_state': [27]}
		else:
			params = _params


	@staticmethod
	def ExtraTree(_params=None):
		global clf, params
		clf = ExtraTreeClassifier()
		print("---> Classification model = 'Extra Tree Classifier'", '\n')
		if _params is None:
			params = {'max_depth': [5, 10, 20], 'min_samples_split':[0.1, 0.2, 0.5], 'max_features':['auto', 'sqrt', 'log2']}
		else:
			params = _params


	@staticmethod
	def LogisticReg(_params=None):
		global clf, params
		clf = LogisticRegression()
		print("---> Classification model = 'Logistic Regression'", '\n')
		if _params is None:
			params = {'C': [0.1, 1, 10], 'class_weight': ['balanced'], 'max_iter': [500, 1000, 1500], 'random_state' : [27], 'solver' : ['lbfgs', 'liblinear']}
		else:
			params = _params


	@staticmethod
	def CatBoost(_params=None):
		global clf, params
		clf = CatBoostClassifier()
		print("---> Classification model = 'CatBoost'", '\n')
		if _params is None:
			params = {'iterations': [10, 20, 30], 'learning_rate': [0.001, 0.01], 'border_count': [100, 200], 'feature_border_type': ['GreedyLogSum']}
		else:
			params = _params


	@staticmethod
	def LGBM(_params=None):
		global clf, params
		clf = LGBMClassifier()
		print("---> Classification model = 'Light Gradient Boosting Machine'", '\n')
		if _params is None:
			params = {'boosting_type': ['gbdt'], 'num_leaves': [500, 1000, 1500, 2000], 'learning_rate': [0.001, 0.01, 0.1], 'n_estimators': [100, 200, 300, 500]}
		else:
			params = _params


	@staticmethod
	def Bagging(_params=None):
		global clf, params
		clf = BaggingClassifier()
		print("---> Classification model = 'Bagging'", '\n')
		if _params is None:
			params = {'base_estimator': [None], 'n_estimators': [10, 50, 100], 'max_samples': [0.1, 1.0]}
		else:
			params = _params


	@staticmethod
	def AdaBoost(_params=None):
		global clf, params
		clf = AdaBoostClassifier()
		print("---> Classification model = 'AdaBoost'", '\n')
		if _params is None:
			params = {'base_estimator': [None], 'n_estimators': [10, 50, 100], 'learning_rate': [0.01, 0.1]}
		else:
			params = _params


	@staticmethod
	def RandomForest(_params=None):
		global clf, params
		clf = RandomForestClassifier()
		print("Classification model = 'Random Forest'", '\n')
		if _params is None:
			params = {'max_depth': [2, 3, 4, 5, 6, 8, 10], 'n_estimators': [500, 1000, 1200, 1500]}
		else:
			params = _params


	@staticmethod
	def KNeighbors(_params=None):
		global clf, params
		clf = KNeighborsClassifier()
		print("Classification model = 'KNeighbors'", '\n')
		if _params is None:
			params = {'n_neighbors': [5, 7, 10, 15, 20, 22, 30, 35], 'weights': ['uniform', 'distance'], 'p': [1, 2]}
		else:
			params = _params


	@staticmethod
	def RadiusNeighbor(_params=None):
		global clf, params
		clf = RadiusNeighborsClassifier()
		print("---> Classification model = 'Radius Neighbors'", '\n')
		if _params is None:
			params = {'radius': [21, 25, 27, 29, 31, 35], 'weights': ['uniform', 'distance'], 'p': [1, 2]}
		else:
			params = _params


	@staticmethod
	def GradientBoosting(_params=None):
		global clf, params
		clf = GradientBoostingClassifier()
		print("Classification model = 'Gradient Boosting'", '\n')
		if _params is None:
			params = {'max_depth': [2, 5, 10], 'n_estimators': [100, 500, 1000]}
		else:
			params = _params


	@staticmethod
	def DecisionTrees(_params=None):
		global clf, params
		clf = DecisionTreeClassifier()
		print("Classification model = 'Decision Trees'", '\n')
		if _params is None:
			params = {'max_depth': [2, 5, 10], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 5, 10]}
		else:
			params = _params


	@staticmethod
	def Svc(_params=None):
		global clf, params
		clf = SVC()
		print("Classification model = 'Support Vector'", '\n')
		if _params is None:
			params = {'gamma': [1.0, 0.1, 0.01, 0.001], 'C': [0.1, 1.0, 10.0], 'class_weight':['balanced']}
		else:
			params = _params


	@staticmethod
	def NUSvc(_params=None):
		global clf, params
		clf = NuSVC()
		print("Classification model = 'Nu-Support Vector'", '\n')
		if _params is None:
			params = {'kernel': ['rbf', 'poly'], 'gamma': ['auto', 'scale']}
		else:
			params = _params


	@staticmethod
	def SGDC(_params=None):
		global clf, params
		clf = SGDClassifier()
		print("Classification model = 'Stochastic Gradient Descent'", '\n')
		if _params is None:
			params = {'loss': ['hinge', 'log', 'modified_huber'], 'class_weight':['balanced']}
		else:
			params = _params


	@staticmethod
	def LinearDA(_params=None):
		global clf, params
		clf = LinearDiscriminantAnalysis()
		print("Classification model = 'Linear Discrimination Analysis'", '\n')
		if _params is None:
			params = {'solver': ['svd', 'lsqr', 'eigen'], 'shrinkage':['auto']}
		else:
			params = _params


	@staticmethod
	def CalibratedCV(_params=None):
		global clf, params
		clf = CalibratedClassifierCV()
		print("Classification model = 'Calibrated Classifier CV'", '\n')
		if _params is None:
			params = {'method': ['sigmoid', 'isotonic'], 'n_jobs':[5, 10, 20, 30]}
		else:
			params = _params



	@staticmethod
	def Mlp(_params=None):
		global clf, params
		clf = MLPClassifier()
		print("Classification model = 'Multi-layer Perceptron'", '\n')
		if _params is None:
			params = {'hidden_layer_sizes':[(8, 8), (8, 6), (8, 4)],'learning_rate_init': [0.01, 0.03, 0.05], 'max_iter': [250], 'activation':['relu', 'identity'],
								'batch_size':['auto',32, 64], 'solver':['adam', 'lbfgs', 'sgd'], 'random_state': [315]}
		else:
			params = _params


##############################################******Fit Function********################################################


	def fit(self, test_size=0.20, random_state=None, optimization='Grid', cv=10, scoring='roc_auc', return_dataset=None):
		global clf_fit, clf_cv, X_train, X_test, y_train, y_test, Training_scores_mean, Training_scores_std, Test_scores_mean, Test_scores_std
		scoring_methods = ['roc_auc', 'roc_auc_ovr', 'roc_auc_ovo', 'accuracy', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted', 'f1', 'recall', 'precision']

		if scoring in scoring_methods:
			pass
		else:
			score_error = "{} is not a valid scoring method. Valid scoring methods are following: 'roc_auc', 'roc_auc_ovr','roc_auc_ovo', 'accuracy', 'roc_auc_ovr_weighted', 'f1', 'recall', 'precision'".format(scoring)
			raise NameError(score_error)

		X_train, X_test, y_train, y_test = train_test_split(self.inputs, self.target, test_size=test_size, stratify=self.target, random_state=random_state)
		if self.verbosity in [1,2]:
			print("target", Counter(self.target))
			print("y_train:", Counter(y_train), ',',  "y_test:", Counter(y_test), '\n')
		else:
			pass

		if return_dataset is 'train':
			df_train = pd.DataFrame(y_train)
			df_train.index = y_train.index
			df_train.to_csv('Train_data' + '_' + self.timenow + '.csv')
		elif return_dataset is 'test':
			df_test = pd.DataFrame(y_test)
			df_test.index = y_test.index
			df_test.to_csv('Test_data' + '_' + self.timenow + '.csv')
		else:
			pass


		if optimization is 'Grid':
			clf_cv = GridSearchCV(clf, params, cv=cv, scoring=scoring)
		elif optimization is 'Randomized':
			no_iter= (int(input("Enter no. of iterations=")))
			print(no_iter)
			clf_cv = RandomizedSearchCV(clf, params, n_iter=no_iter, cv=cv, scoring=scoring, random_state=random_state)
		elif optimization is 'Bayesian':
			no_iter = int(input("Enter no. of iterations: "))
			clf_cv = BayesSearchCV(clf, params, scoring=scoring, n_iter=no_iter, cv=cv, random_state=random_state)

		else:
			search_error = "{} is not a valid option for hyper-paramteres optimization. Available options are 'Grid', 'Randomized' or 'Bayesian'".format(optimization)
			raise ValueError (search_error)


		clf_fit = clf_cv.fit(X_train, y_train)
		print("-----------------------------------------------------------------------------------------------------")
		print('Best estimator:', clf_cv.best_estimator_)
		print('Best parameters:', clf_cv.best_params_)
		print("-----------------------------------------------------------------------------------------------------", '\n')

		# Training_score
		Training_score = cross_val_score(clf_cv.best_estimator_, X_train, y_train, cv=cv, scoring=scoring)
		Training_scores_mean = Training_score.mean()
		Training_scores_std = Training_score.std()

		# Test_score
		Test_score = cross_val_score(clf_cv.best_estimator_, X_test, y_test, cv=cv, scoring=scoring)
		Test_scores_mean = Test_score.mean()
		Test_scores_std = Test_score.std()


		print("*******Train_score*******")
		print("{} = ".format(scoring), round(Training_scores_mean, 4), '+/-', round(Training_scores_std, 4) * 2, 'std', round(Training_scores_std, 4))
		print("*******Test_score*******")
		print("{} = ".format(scoring),  round(Test_scores_mean, 4), '+/-', round(Test_scores_std, 4) * 2, 'std', round(Test_scores_std, 4), '\n')


	"""
	Parameters:

	test_size    		= float		:	For specifying test fraction for dataset (default = 0.20)
	random_no    		= any		: 	Random number for reproducing the results    (default = None)
	optimization 		= str		:	Method for searching the best hyperparameters for the model  (default = 'Grid'); Available methods are = 'Grid', 'Bayesian' and 'Randomized'
	cv 			 		= integer	:	cross-validation
	scoring 	 		= str  		:	Method for the evaluation of model: (default = 'roc_auc'); Available methods are = 'roc_auc', 'roc_auc_ovr', 'roc_auc_ovo', 'accuracy', 'roc_auc_ovr_weighted', 'roc_auc_ovr_weighted', 'f1', 'precision', 'recall'
	return_dataset   	= str		:   Returns a csv file of training or test dataset (default = None); Available = 'train', 'test'													

	"""

##############################################******Make Predictions********############################################

	def make_prediction(self, prediction_data= 'test', proba_prediction=False, save_csv=False, file_name= 'predicted_data'):
		global df, unknown_prediction, test_prediction, train_prediction, df_unknown
		if self.predict_unknown:
			if prediction_data is None:
				raise AssertionError("prediction_data is not defined")
			if prediction_data is 'test':
				if proba_prediction:
					test_prediction = clf_fit.predict_proba(X_test)[:,1]
				else:
					test_prediction = clf_fit.predict(X_test)

				print('*******Prediction_score on testing data*******')
				print("roc_auc_score", round(roc_auc_score(y_test, test_prediction), 4), '\n')
				if self.verbosity is 2:
					for i in range(len(X_test)):
						print(y_test[i], test_prediction[i])
				else:
					pass
				if save_csv:
					predicted_test_data = {'true':y_test, 'predicted':test_prediction}
					df = pd.DataFrame(predicted_test_data)
					df.index = X_test.index
					df.to_csv(file_name + '_' + self.timenow + '.csv')

			if prediction_data is 'train':
				if proba_prediction:
					test_prediction = clf_fit.predict_proba(X_test)[:,1]
				else:
					train_prediction = clf_fit.predict(X_train)

				print('*******Prediction_score on training data*******')
				print("roc_auc_score", round(roc_auc_score(y_train, train_prediction), 4))
				if self.verbosity is 2:
					for i in range(len(X_train)):
						print(y_train[i], train_prediction[i])
				else:
					pass
				if save_csv:
					predicted_train_data = {'true': y_train, 'predicted': train_prediction}
					df = pd.DataFrame(predicted_train_data)
					df.index = self.inputs.index
					df.to_csv(file_name + '_' + self.timenow + '.csv')

			if prediction_data is 'unknown':
				if self.unknown_data is None:
					raise NameError("'unknown_data' is not defined")
				else:
					if self.normalization is 'zscore':
						norm_data = zscore(self.unknown_data)
						df_unknown = pd.DataFrame(norm_data, columns=self.unknown_data.columns)
					if self.normalization is 'minmax':
						minmax_data = MinMaxScaler()
						norm_data = minmax_data.fit_transform(self.unknown_data)
						df_unknown = pd.DataFrame(norm_data, columns=self.unknown_data.columns)
					elif self.normalization is None:
						df_unknown = (self.unknown_data - self.unknown_data.mean()) / self.unknown_data.std()
				if proba_prediction:
					unknown_prediction = clf_fit.predict_proba(df_unknown)
				else:
					unknown_prediction = clf_fit.predict(df_unknown)
				if self.verbosity is 2:
					print([unknown_prediction])
				else:
					pass
				if save_csv:
					df = pd.DataFrame(unknown_prediction, columns=['pred'])
					df.index = df_unknown.index
					df.to_csv(file_name + '_' + self.timenow + '.csv')



	"""
	Parameters:

	prediction_data		= bool		:	Dataset to make predictions; only if predict_unknown is True (default = 'test')
	save_csv	 		= bool		:	Whether to save a csv file of predictions or not (default = False)
	file_name	 		= str		:	Name for the csv file
	
	"""


##############################################******Visualization********###############################################


	def plot_correlation(self, method='pearson', matrix_type='upper', annot=False, cmap='coolwarm',  vmin=-1.0, vmax=1.0,
						  figsize=(16, 12), fontsize=14, save_fig=False, save_csv=False, fig_name="Correlation_plot.png"):
		global figsav

		if self.correlation:
			methods = ['pearson', 'kendall', 'spearman']
			if method not in methods:
				method_error = "{} is not a valid method. Valid methods are 'pearson','kendall', and 'spearman'".format(method)
				raise ValueError(method_error)

			# for inputs in self.inputs:
			df_ = pd.DataFrame(self.inputs)
			correlation = df_.corr(method=method, min_periods=1)

			if matrix_type is 'full':
				plt.figure(figsize=figsize)
				plt.rcParams['font.size'] = fontsize
				figsav = sns.heatmap(correlation, annot=annot, cmap=cmap, linewidths=0.5,
										 vmax=vmax, vmin=vmin)
				plt.show()

			elif matrix_type is 'upper':
				mask_ut = np.tril(np.ones(correlation.shape)).astype(np.bool)
				plt.figure(figsize=figsize)
				plt.rcParams['font.size'] = fontsize
				figsav = sns.heatmap(correlation, mask=mask_ut, annot=annot, cmap=cmap, linewidths=0.5,
										 vmax=vmax, vmin=vmin)
				plt.show()

			elif matrix_type is 'lower':
				mask_ut = np.triu(np.ones(correlation.shape)).astype(np.bool)
				plt.figure(figsize=figsize)
				plt.rcParams['font.size'] = fontsize
				figsav = sns.heatmap(correlation, mask=mask_ut, annot=annot, cmap=cmap, linewidths=0.5,
										 vmax=vmax, vmin=vmin)
				plt.show()

			else:
				raise NameError("{} is not a valid matrix_type. Available matrix_type are 'full', 'upper', or 'lower'".format(matrix_type))

			df_corr = pd.DataFrame(correlation)
			if save_csv:
				df_corr.to_csv("Correlation.csv")

			if save_fig:
				figsav.figure.savefig(fig_name+self.timenow+'.png', format='png', dpi=600)


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

	def plot_feature_imp(self, kind="barh", random_no=None, figsize=(22,16), fontsize=20, color='#ff8000', lw=5.0, save_fig=False, fig_name="Feature_imp_Plot(MI).png"):
		if self.feature_imp:
			X_train, X_test, y_train, y_test = train_test_split(self.inputs, self.target, test_size=0.2, stratify=self.target)
			MI = mutual_info_classif(X_train, y_train, random_state=random_no)
			MI = pd.Series(MI)
			MI.index = X_train.columns
			MI.sort_values(ascending=True, inplace=True)
			print(MI)

			plot_kind=['barh', 'bar', 'pie', 'line', 'area']
			if kind in plot_kind:
				if kind is "pie":
					figsav = MI.plot(kind=kind, normalize=False)
					plt.show()
				else:
					figsav = MI.plot(kind=kind, figsize=figsize, fontsize=fontsize, color=color, lw=lw)
					plt.show()
			else:
				error="{} is not a valid type for plotting feature importance. Only 'barh', 'bar', 'pie', 'line', 'area' can be used for plotting".format(kind)
				raise ValueError(error)

			if save_fig:
				figsav.figure.savefig(fig_name+self.timenow+'.png', format='png', dpi=600)


	"""
	Parameters:

	kind 		= str		: 	Type of plot: (default = 'barh'); Available types = 'barh', 'bar', 'pie', 'line', 'area'  
	random_no 	= any		:	If want to set any random_state (default = None)
	figsize 	= tuple  	: 	Tuple of two integers for determining the figure size (default =(22, 16))		 
	fontsize 	= int  		:	Font size of color-bar and x, y axis (default =20)
	color 		= str  		: 	Color for plot    (default = '#ff8000')	
	lw 			= float 	: 	Width of bars if kind == 'bar or barh' (default = 5.0)
	save_fig 	= bool 		: 	Save plot in the current working directory if True (default = False)
	figname 	= str   	:	Name of fig if save_fig is True (default = "Feature_imp_Plot(MI).png")

	"""


	def plot_roc(self, figsize=(9, 7), lines_fmt=dict(color=["#339966", "#cc0000"], lw=3), label='ROC_curve', fontsize=18, ticksize=18, fig_name='roc_plot', save_fig=False):

		if self.ROC_curve:
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
	fig_name 	= str  		: 	Name for the figure if want to save figure    (default = 'roc_plot')
	save_fig 	= bool 		: 	Save Figure in the current directory if True    (default = False)

	"""
