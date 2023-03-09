from automate_ML import Regression
import pandas as pd


data = pd.read_csv('diamonds.csv')

X = ['carat', 'cut', 'color', 'clarity', 'depth', 'table']

y = 'price'


reg = Regression(data=data, inputs=X, target=y, problem='Regression', normalization='zscore', random_state=300)
reg.Model(model_name='ExtraTrees', random_state=300)
reg.fit(optimization='Random', num_iter=20, scoring='r2')
reg.predict(prediction_data='test', save_csv=False)
reg.plot_scatter(plot_for='test')
reg.plot_correlation()
reg.plot_feature_imp()
print('Done')
