# Automate-ML

Simple and mechanized way to solve classification and regression problems. The current code is only for solving classification problems. For regression the code is under progress. <br /> <br />
Purposes of this repository are following:
*  To provide an efficient and easy way to perform the Machine learning tasks.
*  Visualization of results with some important graphs

## Installation

You can simply install automate-ML by using pip (https://pypi.org/project/automate-ML/):
````
pip install automate-ML 
````
You can also use github link:
````
https://github.com/sohaibunist/automate_ML.git 
````

## How to use:


> Make an instance:

First you can compare the performance of all the available ML models and then optimize the parameters for the best performing model.

````
from automate_ML.main import Models

select_model = Models(data=dataframe, inputs=[list of features], target= 'variable to predict')
select_model.best_model(show=True)
 
````
For the selected model you can perform further analysis by making an instance of Classification class.
````
from automate_ML.main import Classification

clf = Classification(data=dataframe, inputs=[list of features], target= 'variable to predict', normalization='minmax', verbosity=0)
````

> Call and fit a machine learning model:
  In the following case Support Vector Classifier (Svc) is used
````
clf.Svc()
clf.fit(optimization='Bayesian', num_iter=20, cv=10, scoring='accuracy')
````
> Now make predictions while using 'train', 'test', or 'unknown' dataset
````
clf.make_prediction(prediction_data='test')
````

> Get some useful plots:
````
clf.plot_correlation()
clf.plot_feature_imp()
clf.plot_roc_curve()
clf.Confusion_matrix()
````
> Available ML algorithms 

You can print out the list of available ML algorithms by following code:

 ````
 clf.model_names()
 ````

 ````
 AdaBoost()
 Bagging()
 CatBoost()
 CalibratedCV()
 DecisionTrees()
 ExtraTrees()
 ExtraTree()
 GradientBoosting()
 KNeighbors()
 LogisticReg()
 LGBM()
 LinearDA()
 Linear_SVC()
 Mlp()
 NUSvc()
 RandomForest()
 RadiusNeighbor()
 Ridge()
 RidgeCV()
 Svc() 
 SGDC()
 
````
For these models you can either use the default parameters that are used in this repository or you can pass a dictionary of your own hyperparameters. For hyperparameters detail of each model visit scikit-learn or the homepage of the corresponding model. <br />

Example:

 ````
 clf.AdaBoost()
 ````
OR
 ````
 clf.AdaBoost({'n_estimators': [10, 50, 100], 'learning_rate': [0.001, 0.04, 0.05, 0.09, 0.1], 'random_state':[315]})
 ````


## Parameters description:

> __init__()

The __init__ parameters for 'Classification' and 'Models' classes are same.

````
data            = Dataframe   :Dataset for evaluating a model  (default = None)
inputs          = Dataframe   :Feature set (default = None)
target          = Dataframe   :Target which you want to predict  (default = None)
nan_values      = str         :Whether to 'impute' or 'remove' NaN value in the dataset. (default=None)	
normalization   = str         :Method for normalizing the dataset (default = "None")
test_size       = float       :Size od testing dataset (default = 0.20)
random_state    = int         :random number for the reproducing the results (deafult = 315)
return_dataset  = str         :Dataset to be returned as a csv file. (default = None)
verbosity       = integer     :Degree for printing output messages in the terminal (default = 0, can be 0,1, or 2)

````  



> fit()
````
optimization    =str  : Method for searching the best hyperparameters for the model  (default = 'Grid'); Available methods are = 'Grid', 'Bayesian' and 'Randomized'
num_iter        =int  : Number of iterations to run for hyperparameter optimization (default = 20)
cv              =int  : cross-validation (default = 10)
scoring         =str  : Method for the evaluation of model: (default = 'roc_auc')

````

> make_prediction()
````
prediction_data   = bool      :Dataset to make predictions (default = 'test')
unknown_data      = Dataframe :Unknown dataset for predictions; required when prediction_data is 'unknown' (default = None)
proba_prediction  = bool      :Predict probabilities rather than the exact values for the target if set True (default = False)
save_csv          = bool      :Save a csv file of predictions if set True (default = False)
file_name         = str       :Name for the csv file (default = 'predicted_data')

````

> Confusion_matrix()
````
show_plot   =bool  : Visualize confusion matrix if set True (default = False)  
annot       =bool  : Print the confusion matrix values in the heatmap if set True  (default = False)
cmap        =any   : Color map for plot  (default = 'Blues')
figsize     =tuple : Tuple of two integers for determining the figure size    (default =(16, 12))
fontsize    =int   : Font size of color-bar and x, y axis   (default =14)
save_fig    =bool  : Save plot in the current working directory if True  (default = False)
figname     =str   : Name of fig if save_fig is True  (default = "Correlation_plot.png")

````

> plot_correlation()
````
method      = str   : Method for plottting correlation matrix (default = 'pearson') Available methods = 'perason', 'kendall', or 'spearman'  
matrix_type = bool  : Type of correlation-matrix for plotting  (default = upper); Available = 'full', 'upper', 'lower'
annot       = bool  : Print the correlation values in the heatmap if set True  (default = False)
cmap        = any   : Color map for plot  (default = coolwarm)
vmin        = float : Minimum value for color bar (default = -1.0)
vmax        = float : Maximum value for color bar (default =  1.0)
figsize     = tuple : Tuple of two integers for determining the figure size    (default =(16, 12))
fontsize    = int   : Font size of color-bar and x, y axis   (default =14)
save_fig    = bool  : Save plot in the current working directory if True  (default = False)
save_csv    = bool  : Save a csv file if True  (default = False)
figname     = str   : Name of figure. Only if save_fig is True  (default = "Correlation_plot.png")

````

> plot_feature_imp()
````
kind      = str   : Type of plot: (default = 'barh'); Available types = 'barh', 'bar', 'pie', 'line', 'area'  
random_no = any   : Random number to reproduce results (default = None)
figsize   = tuple : Tuple of two integers for determining the figure size (default =(22, 16))		 
fontsize  = int   : Font size of color-bar and x, y axis (default =20)
color     = str   : Color for plot    (default = '#ff8000')	
lw        = float : Width of bars if kind == 'bar' or 'barh' (default = 5.0)
save_fig  = bool  : Save plot in the current working directory if True (default = False)
figname   = str   : Name of figure. Only if save_fig is True (default = "Feature_imp_Plot(MI).png")

````


> plot_roc_curve()
````
figsize   = tuple : Tuple of two integers for determining the figure size  (default =(9, 7))		 
lines_fmt = dict  : Dictionary for the formatting of lines i.e. 'color' and linewidth('lw')	 (default = {'color': ["#339966", "#cc0000"], 'lw': 3}
label     = str   : Set label inside the plot (default = 'ROC_curve')
fontsize  = int   : Set fontsize for the x and y labels  (default = 18)
ticksize  = int   : Set fontsize for the x and y ticks   (default = 18)
save_fig  = bool  : Save Figure in the current directory if True    (default = False)
fig_name  = str   : Name for the figure. Only if save_fig is True    (default = 'roc_plot')


````

> best_model()
````
n_splits      =int    : No of splits  (default =100)		 
test_size     =float  : Fraction of datset to be chosen for testing	 (default = 0.20)
random_state  =int    : Any random no to reproduce the results (default = None)
scoring       =str    : Scoring method  (default = 'roc_auc')
save_txt      =bool   : Save a txt files with model names and corresponding scores   (default = True)
filename      =str    : Name of the txt file   (default = 'Models_score')
show          =bool   : Print out the sorted table and plots a bar chart of the models with corresponding scores if set True   (default = True)

````
