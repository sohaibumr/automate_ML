from automate_ML import all_Models
import pandas as pd

data = pd.read_csv('drugs_classification.csv')

X = ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']

y = 'Drug'

clf = all_Models(data=data, inputs=X, target=y, problem='Classification', random_state=300)
clf.best_model(n_splits=10, scoring='f1_macro')
print('Done')