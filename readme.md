# automate-ML

<img width="536" height="324" alt="automate-ML_fig" src="https://github.com/user-attachments/assets/e6a357eb-8da3-470d-94a3-6076700d2fc1" />


[![Downloads](https://static.pepy.tech/badge/automate-ML)](https://pepy.tech/project/automate-ML)

Simple and mechanized way to solve classification and regression problems. This python package is able to preprocess the data and output the results in the numerical as well as in the graphical form. <br /> <br />
The purpose of this python package is to:
*  Provide an efficient and easy way to perform the Machine learning tasks (Classification and Regression).
*  Organize the unstructured data through preprocessing.
*  Provide a uniform interface for optimizing hyper-parameters through skopt based bayesian, and sklearn based grid and random search.
*  Visualize the results with some important graphs.

## Installation

The stable version of the automate-ML can be installed using pip (https://pypi.org/project/automate-ML/):
````
pip install automate-ML

````
The package can also be downloaded or cloned using git:

````
git clone https://github.com/sohaibumr/automate_ML.git
````

## Quick Example:

**The usage of this package is explained through different examples (Check examples folder).**

````
from automate_ML import Classification
clf = Classification(data=dataframe, inputs=[list of features], target='variable to predict', problem='Classification', normalization='minmax', verbosity=0) 
````

> Call a machine learning model
````
clf.Model(model_name='ExtraTrees', random_state=300)
````
For the models either the default parameters can be used or a dictionary of parameters can be passed. For hyperparameters detail of each model visit scikit-learn or the homepage of the corresponding model. <br />

````
clf.Model(model_name='ExtraTrees',
        params={
      'n_estimators': list(range(50, 500)),
        'max_depth': list(range(2, 11)),
        'min_samples_split': list(range(0, 11))
            },
         random_state=300)

````
Call fit function for this model.
````
clf.fit(optimization='Bayesian', num_iter=20, cv=10, scoring='accuracy')
````

> Now make predictions while using 'train', 'test', or 'unknown' dataset
````
clf.predict(prediction_data='test')
````

> Get some useful plots:
````
clf.plot_correlation()
clf.plot_feature_imp()
clf.plot_roc_curve()
clf.Confusion_matrix()
````
## Documentation:

For the documentation see the docs folder.





