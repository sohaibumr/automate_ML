from automate_ML import Classification
import pandas as pd


data = pd.read_csv('drugs_classification.csv')

X = ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']

y = 'Drug'

print(data)

clf = Classification(data=data, inputs=X, target=y, problem='Regression', random_state=300)
clf.Model(model_name='ExtraTrees', random_state=300)
clf.fit(optimization='Bayesian', num_iter=5, scoring='accuracy')
clf.predict(prediction_data='test')
clf.plot_correlation()
clf.plot_feature_imp()
clf.Confusion_matrix()
print('Done')
