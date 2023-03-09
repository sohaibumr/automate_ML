from automate_ML import Preprocessing
import pandas as pd


data = pd.read_csv('drugs_classification.csv')

X = ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']

y = 'Drug'

# To check if target is multiclass or not
pre_p = Preprocessing(data=data, inputs=X, target=y, problem='Classification')
print(pre_p.multiclass)

# Impute nan values in the dataset
pre_p2 = Preprocessing(data=data, inputs=X, target=y, problem='Classification', nan_values='impute')
pre_p2.preprocessed_data()

# Convert categorical features in the numerical form
pre_p2 = Preprocessing(data=data, inputs=X, target=y, problem='Classification', label_encode_inputs=True)
pre_p2.input_data()

# Convert categorical data in the target column to numerical form
pre_p2 = Preprocessing(data=data, inputs=X, target=y, problem='Classification', label_encode_target=True)
pre_p2.target_data()

# To get train or test dataset
pre_p.return_dataset(return_data='train')