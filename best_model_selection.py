from clf_models_params import *


class Models:
	def __init__(self, data=None, inputs=None, outputs=None):

		self.data = data

		if inputs is None:
			raise NameError ("'inputs' is not defined")
		else:
			self.inputs = (inputs - inputs.mean()) / inputs.std()

		if outputs is None:
			raise NameError ("'outputs' is not defined")
		else:
			self.outputs = outputs


	def best_model(self, n_splits=100, test_size=0.20, random_state=None, scoring='roc_auc', save_txt=True, filename='Models_score'):

		models = []
		models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
		models.append(('LDA', LinearDiscriminantAnalysis()))
		models.append(('KNN', KNeighborsClassifier()))
		models.append(('DT', DecisionTreeClassifier()))
		models.append(('SVC', SVC(gamma='auto')))
		models.append(('ETs', ExtraTreesClassifier()))
		models.append(('LGBM', LGBMClassifier()))
		models.append(('Bag', BaggingClassifier()))
		models.append(('AdaB', AdaBoostClassifier()))
		models.append(('RF', RandomForestClassifier()))
		models.append(('RN', RadiusNeighborsClassifier(radius=25)))
		models.append(('GB', GradientBoostingClassifier()))
		models.append(('Nusvc', NuSVC(nu=0.1)))
		models.append(('SGDC', SGDClassifier()))
		models.append(('CalCV', CalibratedClassifierCV()))
		models.append(('ET', ExtraTreeClassifier()))


		results = []
		names = []
		for name, model in models:
			ss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
			cv_results = cross_val_score(model, self.inputs, self.outputs, cv=ss, scoring=scoring)
			results.append(cv_results.mean())
			names.append(name)
			print("%s: score=%f std=(%f)" % (name, cv_results.mean(), cv_results.std()))


		result = pd.Series(results)
		result.index = names
		result.sort_values(ascending=False, inplace=True)
		print('\n''--------------Sorted Models w.r.t score--------------''\n')
		result_tab = result.to_frame()
		tabulated_results = tabulate(result_tab, headers =['Models', 'Score'], tablefmt='fancy_grid')
		print(tabulated_results)
		if save_txt:
			result.to_csv(filename+'.txt', sep=' ', header=None)
		result.plot(kind='bar')
		plt.show()


