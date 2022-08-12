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
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from tabulate import tabulate
from classification import Classification
import warnings

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


