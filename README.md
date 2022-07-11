# ML_models for solving classification and regression problems
Machine learning models to solve classification and regression problems in a mechanized way.

## How to use:

````
from clf_models_dependencies import  *
from clf_models_params import do_classification

clf = do_classification(inputs=X, target=Y, normalization='minmax', predict_unknown=False, #unknown_data=Z, correlation=True, feature_imp=False, ROC_curve=False, verbosity=1)
clf.Svc()
clf.fit(test_size=0.30, random_state=12, cv=10, optimization='Grid', scoring='accuracy', return_dataset=None)
clf.make_prediction(prediction_data='test', proba_prediction=True, save_csv=False)
clf.plot_correlation(cmap='coolwarm', matrix_type='uper', save_fig=False, fig_name='Estab_corr', annot=True)
clf.plot_feature_imp()
clf.plot_roc()
````
