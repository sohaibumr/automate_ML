import sys
from clf_models_dependencies import  *
from clf_models_params import do_classification
from model_selection import Models



data =pd.read_csv(r'D:\2-projects\1-umer\BC\BC_ML-HER_OER_ORR.csv', index_col=0)


X = data[['std_pot', 'unpair_e','first_IP', 'mag_O', 'ang_min_O', 'q_n1_O', 'q_n3_O']]

Z = pd.read_csv(r'D:\2-projects\1-umer\project_1\0-ML_Models_Results\2-Classification_problem\2-coulumb_matrix_pred.csv', index_col=0)



y = data['OER_clf']


slct_model = Models(inputs=X, outputs=y)
slct_model.best_model()
sys.exit()


classification = do_classification(inputs=X, output=y, predict_unknown=True, roc_curve=False, normalization='minmax',  unknown_data=Z, feature_imp=False, correlation=False)
classification.ExtraTrees()
classification.fit(n_splits=10, test_size=0.20, random_no=27)
classification.make_prediction(itself=True, save_csv=True)
classification.plot_correlation(cmap='coolwarm', upper=True,save_fig=False, fig_name='lower', annot=True)
classification.plot_feature_imp()
classification.plot_roc()


