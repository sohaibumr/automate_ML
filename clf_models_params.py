from clf_models_dependencies import *
global clf, params, clf_fit, ss, clf_cv, df

"""""""""""""""""""""""""""""""""""""""""""""""""Classifiers"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


class do_classification:

	def __init__(self, data=None, inputs=None, output=None, normalization=None, predict_unknown=False, unknown_data=None, roc_curve=False,
				 feature_imp=False, correlation=False):


		self.data = data

		column = inputs.columns
		if inputs is None:
			raise NameError ("'inputs' is not defined")
		else:
			if normalization == 'zscore':
				print('Scaling with zscore')
				norm_data = zscore(inputs)
				self.inputs= pd.DataFrame(norm_data, columns=column)
			if normalization == 'minmax':
				print('Scaling with minmax')
				minmax_data = MinMaxScaler()
				norm_data= minmax_data.fit_transform(inputs)
				self.inputs = pd.DataFrame(norm_data, columns=column)
			if normalization is None:
				self.inputs = (inputs - inputs.mean()) / inputs.std()
			print(self.inputs)


		if output is None:
			raise NameError ("'output' is not defined")
		else:
			self.output = output


		self.predict_unknown = predict_unknown

		column = unknown_data.columns
		if not predict_unknown:
			self.unknown_data = unknown_data
		elif predict_unknown:
			if unknown_data is None:
				raise NameError ("'unknown_data' is not defined")
			else:
				if normalization == 'zscore':
					norm_data = zscore(inputs)
					self.unknown_data = pd.DataFrame(norm_data, columns=column)
				if normalization == 'minmax':
					minmax_data = MinMaxScaler()
					norm_data = minmax_data.fit_transform(unknown_data)
					self.unknown_data = pd.DataFrame(norm_data, columns=column)
				elif normalization is None:
					self.unknown_data = (unknown_data - unknown_data.mean()) / unknown_data.std()


		self.feature_imp = feature_imp
		self.correlation = correlation
		self.roc_curve = roc_curve

	"""
	Parameters:

	data =  any			: 	Dataset for evaluating a model  (default = None)
	inputs = any		:	Feature set (default = None)
	output = any		: 	Target which you want to predict  (default = None)
	normalization: any  :	Method for normalizing the dataset (default = "None"
	unknown_data = any  :	Dataset to make prediction  (default = None)
	predict_unknown = 	:	Set True if want to make prediction for 'unknown_data' (default = False)
	feature_imp = bool	:	Plots feature importance plot using Mutual information method (MI) if True (deafult = False)
	correlatiion = bool	:	Plots correlation heatmap if True (default = False)

	"""
	@property
	def timenow(self, ):
		T = str(datetime.now())
		t = T.replace(':', '')
		return t


	def ExtraTrees(self, _params={'n_estimators': [100, 500],  'max_depth': [2, 5], 'random_state': [27]}):
		global clf, params
		clf = ExtraTreesClassifier()
		print("ExtraTrees Classification")
		params = _params


	def ExtraTree(self, _params={'max_depth': [5, 10, 20], 'min_samples_split':[0.1, 0.2, 0.5], 'max_features':['auto', 'sqrt', 'log2']}):
		global clf, params
		clf = ExtraTreeClassifier()
		print("Extra Tree Classifier")
		params = _params


	def LogisticReg(self, _params={'C': [0.1, 1, 10], 'class_weight': ['balanced'], 'max_iter': [1000], 'random_state' : [27], 'solver' : ['lbfgs', 'liblinear']}):
		global clf, params
		clf = LogisticRegression()
		print("Logistic Regression")
		params = _params


	def CatBoost(self, _params={'iterations': [10, 20, 30], 'learning_rate': [0.001, 0.01], 'border_count': [100, 200], 'feature_border_type': ['GreedyLogSum']}):
		global clf, params
		clf = CatBoostClassifier()
		print("CatBoost")
		params = _params


	def LGBM(self, _params={'boosting_type': ['gbdt'], 'num_leaves': [500, 1000, 1500, 2000], 'learning_rate': [0.001, 0.01, 0.1], 'n_estimators': [100, 200, 300, 500]}):
		global clf, params
		clf = LGBMClassifier()
		print("Light Gradient Boosting Machine")
		params = _params


	def Bagging(self, _params={'base_estimator': [None], 'n_estimators': [10, 50, 100], 'max_samples': [0.1, 1.0]}):
		global clf, params
		clf = BaggingClassifier()
		print("Bagging")
		params = _params


	def AdaBoost(self, _params={'base_estimator': [None], 'n_estimators': [10, 50, 100], 'learning_rate': [0.01, 0.1]}):
		global clf, params
		clf = AdaBoostClassifier()
		print("AdaBoost")
		params = _params


	def RandomForest(self, _params={'max_depth': [2, 3, 4, 5, 6, 8, 10], 'n_estimators': [100, 200, 400, 600, 800, 1000, 1200, 1500]}):
		global clf, params
		clf = RandomForestClassifier()
		print("Random Forest")
		params = _params


	def KNeighbors(self, _params={'n_neighbors': [5, 7, 11, 15, 17, 21, 25, 27, 29, 31, 35], 'weights': ['uniform', 'distance'], 'p': [1, 2]}):
		global clf, params
		clf = KNeighborsClassifier()
		print("KNeighbors")
		params = _params


	def RadiusNeighbor(self, _params={'radius': [21, 25, 27, 29, 31, 35], 'weights': ['uniform', 'distance'], 'p': [1, 2]}):
		global clf, params
		clf = RadiusNeighborsClassifier()
		print("Radius Neighbors")
		params = _params


	def GradientBoosting(self, _params={'max_depth': [2, 5, 10], 'n_estimators': [100, 500, 1000]}):
		global clf, params
		clf = GradientBoostingClassifier()
		print("Gradient Boosting")
		params = _params


	def DecissionTrees(self, _params={'max_depth': [2, 5, 10], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 5, 10]}):
		global clf, params
		clf = DecisionTreeClassifier()
		print("Decision Trees")
		params = _params


	def Svc(self, _params={'gamma': [1.0, 0.1, 0.01, 0.001], 'C': [0.1, 1.0, 10.0], 'class_weight':['balanced']}):
		global clf, params
		clf = SVC()
		print("Support Vector")
		params = _params


	def NUSvc(self, _params={'kernel': ['rbf', 'poly'], 'gamma': ['auto', 'scale']}):
		global clf, params
		clf = NuSVC()
		print("Nu-Support Vector")
		params = _params


	# def LinearSvc(self, _params={'penalty': ['l1', 'l2'], 'C': [1.0, 5.0, 10.0], 'class_weight':['balanced']}):
	# 	global clf, params
	# 	clf = LinearSVC()
	# 	print("Linear Support Vector")
	# 	params = _params


	def SGDC(self, _params={'loss': ['hinge', 'log', 'modified_huber'], 'class_weight':['balanced']}):
		global clf, params
		clf = SGDClassifier()
		print("Stochastic Gradient Descent")
		params = _params


	def LinearDA(self, _params={'solver': ['svd', 'lsqr', 'eigen'], 'shrinkage':['auto']}):
		global clf, params
		clf = LinearDiscriminantAnalysis()
		print("Linear Discrimination Analysis")
		params = _params


	def CalibratedCV(self, _params={'method': ['sigmoid', 'isotonic'], 'n_jobs':[5, 10, 20, 30]}):
		global clf, params
		clf = CalibratedClassifierCV()
		print("Calibrated Classifier CV")
		params = _params



	def fit(self, n_splits=100, test_size=0.20, random_no=None, search='Grid', scoring='roc_auc'):
		global clf_fit, ss, clf_cv
		scoring_methods = ['roc_auc', 'roc_auc_ovr', 'roc_auc_ovo', 'accuracy', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted',
						   'f1', 'recall', 'precision']

		if scoring in scoring_methods:
			pass
		else:
			score_error = "{} is not a valid scoring method. Valid scoring methods are following: 'roc_auc', 'roc_auc_ovr','roc_auc_ovo', 'accuracy', 'roc_auc_ovr_weighted', 'f1', 'recall', 'precision'".format(scoring)
			raise ValueError(score_error)


		ss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_no)

		if search == 'Grid':
			clf_cv = GridSearchCV(clf, params, cv=ss.split(self.inputs, self.output), scoring=scoring)
		elif search == 'Randomized':
			no_iter= (int(input("Enter no. of iterations=")))
			print(no_iter)
			clf_cv = RandomizedSearchCV(clf, params, n_iter=no_iter, cv=ss.split(self.inputs, self.output), scoring=scoring)
		elif search == 'Bayesian':
			no_iter = int(input("Enter no. of iterations: "))
			clf_cv = BayesSearchCV(clf, params, scoring=scoring, n_iter=no_iter, cv=5, random_state=random_no)

		else:
			search_error = "{} is not a valid search  parameter. It is either 'Grid', 'Randomized' or 'Bayesian'".format(search)
			raise ValueError (search_error)


		clf_fit = clf_cv.fit(self.inputs, self.output)
		print('Best estimator:', clf_cv.best_estimator_)
		print('Best parameters:', clf_cv.best_params_)
		clfbest_scores = cross_val_score(clf_cv.best_estimator_, self.inputs, self.output, cv=ss, scoring=scoring)
		clfbest_scores_mean = clfbest_scores.mean()
		clfbest_scores_std = clfbest_scores.std()


		if scoring is 'roc_auc':
			print(f"roc_auc_score = {clfbest_scores_mean:.4f} +/- {clfbest_scores_std * 2 :.4f}, std = {clfbest_scores_std:.4f}")
		elif scoring is 'roc_auc_ovr':
			print(f"roc_auc_ovr_score = {clfbest_scores_mean:.4f} +/- {clfbest_scores_std * 2 :.4f}, std = {clfbest_scores_std:.4f}")
		elif scoring is 'roc_auc_ovo':
			print(f"roc_auc_ovo_score = {clfbest_scores_mean:.4f} +/- {clfbest_scores_std * 2 :.4f}, std = {clfbest_scores_std:.4f}")
		elif scoring is 'accuracy':
			print(f"accuracy = {clfbest_scores_mean:.4f} +/- {clfbest_scores_std * 2 :.4f}, std = {clfbest_scores_std:.4f}")
		elif scoring is 'roc_auc_ovr_weighted':
			print(f"roc_auc_ovr_weighted_score = {clfbest_scores_mean:.4f} +/- {clfbest_scores_std * 2 :.4f}, std = {clfbest_scores_std:.4f}")
		elif scoring is 'roc_auc_ovo_weighted':
			print(f"roc_auc_ovo_weighted_score = {clfbest_scores_mean:.4f} +/- {clfbest_scores_std * 2 :.4f}, std = {clfbest_scores_std:.4f}")
		elif scoring is 'f1':
			print(f"f1_score = {clfbest_scores_mean:.4f} +/- {clfbest_scores_std * 2 :.4f}, std = {clfbest_scores_std:.4f}")
		elif scoring is 'recall':
			print(f"recall = {clfbest_scores_mean:.4f} +/- {clfbest_scores_std * 2 :.4f}, std = {clfbest_scores_std:.4f}")
		elif scoring is 'precision':
			print(f"precision = {clfbest_scores_mean:.4f} +/- {clfbest_scores_std * 2 :.4f}, std = {clfbest_scores_std:.4f}")


	"""
	Parameters:

	n_splits =  int		: 	No of cross-validation splits  (default = 100)
	test_size = float	:	For specifying test fraction for dataset (default = 0.20)
	random_no = any		: 	Random number for reproducing the results    (default = None)
	search = str		:	Method for the search of best parameters  (default = 'Grid')
	scoring = str  		:	Method for the evaluation of model: 'roc_auc', 'roc_auc_ovr', 'roc_auc_ovo', 'accuracy', 
							'roc_auc_ovr_weighted', 'roc_auc_ovr_weighted', 'f1', 'precision', 'recall'			(default = 'roc_auc')

	"""



	def make_prediction(self, itself=False, save_csv=False, file_name= 'predicted_data'):
		global df
		if self.predict_unknown:
			if itself:
				self_prediction = clf_fit.predict(self.inputs)
				print(accuracy_score(self.output, self_prediction))
				for i in range(len(self.inputs)):
					print(self.output[i], self_prediction[i])
			else:
				unknown_prediction = clf_fit.predict(self.unknown_data)
				print([unknown_prediction])
		if save_csv:
			if itself:
				df = pd.DataFrame(self_prediction, columns=['pred'])
				df.index = self.inputs.index
				df.to_csv(file_name + '_' + self.timenow + '.csv')
			else:
				df = pd.DataFrame(unknown_prediction, columns=['pred'])
				df.index = self.unknown_data.index
				df.to_csv(file_name+'_'+self.timenow+'.csv')



	def plot_roc(self, figsize=(9, 7), lines_fmt={'color':["#339966", "#cc0000"], 'lw':3}, label='Mean ROC', fontsize=18, ticksize=18,  fig_name='roc_plot',  save_fig=False):

		if self.roc_curve:
			pd.options.display.width = 0

			flatui = lines_fmt['color']
			palette = sns.color_palette(flatui)
			sns.set_palette(palette)

			linewidth0 = 1.5
			linewidth = lines_fmt['lw']

			labelsize = fontsize
			ticksize = ticksize

			mean_fpr = np.linspace(0, 1, 1000)

			def curves(X, y, model_fit):
				tprs = []
				aucs = []

				fig, ax = plt.subplots(figsize=figsize)

				for i, (train, test) in enumerate(ss.split(X, y)):
					clf.fit(X.iloc[train, :], y[train])
					viz = plot_roc_curve(clf, X.iloc[test, :], y[test], alpha=0.3, lw=linewidth0, ax=ax)

					interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
					interp_tpr[0] = 0.0
					tprs.append(interp_tpr)
					aucs.append(viz.roc_auc)

				plt.close(fig)

				mean_tpr = np.mean(tprs, axis=0)
				mean_tpr[-1] = 1.0
				std_tpr = np.std(tprs, axis=0)
				tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
				tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

				return mean_tpr, tprs_upper, tprs_lower

			fig, ax = plt.subplots(figsize=figsize)
			plt.rc('font', family='serif')


			mean_tpr, tprs_upper, tprs_lower = curves(self.inputs, self.output, clf_cv.best_estimator_)
			ax.plot(mean_fpr, mean_tpr, label=label, lw=linewidth, alpha=.8)
			ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')


			ax.plot(mean_fpr, mean_fpr, linestyle='--', lw=linewidth, label='Chance', alpha=.8)

			ax.set(xlim=[-0.02, 1.02], ylim=[-0.02, 1.02])
			ax.legend(loc="lower right")
			ax.set_xlabel('False positive rate', fontsize=fontsize)
			ax.set_ylabel('True positive rate', fontsize=fontsize)
			plt.xticks(fontsize=ticksize)
			plt.yticks(fontsize=ticksize)
			plt.legend(fontsize='x-large')

			for axis in ['top', 'bottom', 'left', 'right']:
				ax.spines[axis].set_linewidth(1.6)
				ax.spines[axis].set_color('dimgrey')

			plt.show()
			if save_fig:
				fig.savefig(fig_name+self.timenow+'.png', format='png', dpi=1200)




	"""
	Parameters:

	figsize =  tuple	: 	tuple of two integers    (default =(9, 7))
			[int, int] 
	lines_fmt = dict	: 	Dictionary for the formatting of lines i.e. 'color' and linewidth('lw')	   (default = {'color': ["#339966", "#cc0000"], 'lw': 3}
	label =	   str	 	:	Set label inside the plot         (default = 'Mean ROC')
	fontsize = int 		: 	Set fontsize for the x and y labels  (default = 18)
	ticksize =  int 	:	Set fontsize for the x and y ticks   (default = 18)
	save_fig = bool 	: 	Save Figure in the current directory if True    (default = False)
	fig_name =  str  	: 	Name for the figure if want to save figure    (default = 'roc_plot')

	"""




	def plot_correlation(self, method='pearson', full=False, upper=True, annot=False, cmap=plt.cm.coolwarm, vmax=1.0, vmin=-1.0,
						  figsize=(16, 12), fontsize=14, save_fig=False, save_csv=False, fig_name="Correlation_plot.png"):
		if self.correlation:
			for inputs in self.inputs:
				df = pd.DataFrame(self.inputs)

				correlation = df.corr(method=method, min_periods=1)

				if full and upper:
					raise TypeError ("Both 'full' and 'upper' is True")

				elif full:
					correlation
					plt.figure(figsize=figsize)
					plt.rcParams['font.size'] = fontsize
					figsav = sns.heatmap(correlation, annot=annot, cmap=cmap, linewidths=0.5,
											 vmax=vmax, vmin=vmin)
					plt.show()

				elif upper:
					mask_ut = np.tril(np.ones(correlation.shape)).astype(np.bool)
					plt.figure(figsize=figsize)
					plt.rcParams['font.size'] = fontsize
					figsav = sns.heatmap(correlation, mask=mask_ut, annot=annot, cmap=cmap, linewidths=0.5,
											 vmax=vmax, vmin=vmin)
					plt.show()

				else:
					mask_ut = np.triu(np.ones(correlation.shape)).astype(np.bool)
					plt.figure(figsize=figsize)
					plt.rcParams['font.size'] = fontsize
					figsav = sns.heatmap(correlation, mask=mask_ut, annot=annot, cmap=cmap, linewidths=0.5,
											 vmax=vmax, vmin=vmin)
					plt.show()

				df = pd.DataFrame(correlation)
				if save_csv:
					df.to_csv("Correlation.csv")

				if save_fig:
					figsav.figure.savefig(fig_name+self.timenow+'.png', format='png', dpi=1200)

				break

	"""
	Parameters:
	
	method =   str	: 	Methods for plottting correlation: 'perason', 'kendall', or 'spearman'  (default = 'pearson')
	full =	   bool :	plot full correlation if True         (default = False)
	upper =    bool : 	plot upper triangle for correlation if True and lower triangle if False  (default = True)
	annot =    bool :	write correlation values in boxes if True   (default = False)
	cmap =     any  : 	color map for plot    (default = coolwarm)
	save_csv = bool : 	save a csv file if True     (default = False)
	save_fig = bool : 	save plot in the current directory if True    (default = False)
	figsize =  tuple: 	tuple of two integers    (default =(16, 12))
			[int, int] 
	fontsize = int  :	font size of color-bar and x, y axis     (default =14)
	figname = str   :	name of fig if save_fig is True    (default = "Correlation_plot.png")
	
	"""



	def plot_feature_imp(self, kind="barh", random_no=None, figsize=(22,16), fontsize=20, color='#ff8000', lw=5.0, save_fig=False, fig_name="Feature_imp_Plot(MI).png"):
		if self.feature_imp:
			X_train, X_test, y_train, y_test = train_test_split(self.inputs, self.output, test_size=0.2, stratify=self.output)
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
				figsav.figure.savefig(fig_name+self.timenow+'.png', format='png', dpi=1200)



	"""
	Parameters:

	kind =   	str		: 	Type of plot:  'barh', 'bar', 'pie', 'line', 'area'  (default = 'barh')
	random_no = any		:	If want to set any random_state (default = None)
	figsize =   tuple	: 	tuple of two integers    (default =(22, 16))
			[int, int] 
	fontsize = int  	:	font size of color-bar and x, y axis     (default =20)
	color =     str  	: 	color for plot    (default = '#ff8000')	
	lw =    bool 		: 	width of bars if kind == 'bar or barh'  (default = 5.0)
	save_fig = bool 	: 	save plot in the current directory if True    (default = False)
	figname = str   	:	name of fig if save_fig is True    (default = "Feature_imp_Plot(MI).png")

	"""

