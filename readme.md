# ML_models for solving classification and regression problems
  Machine learning models to solve classification and regression problems in a mechanized way. The current code is only for solving classification problems. For      regression the code will soon be updated
  > Description of the parameters is provided below.

## How to use:

````
from clf_models_dependencies import  *
from clf_models_params import do_classification
````

> Make an instance of class do_classification:
````
clf = do_classification(data=dataframe, inputs=[list of features], target= 'variable to predict', normalization='minmax', verbosity=0)
````

> Call and fit a machine learning model:
  In the following case Support Vector Classifier (Svc) is used
````
clf.Svc()
clf.fit(test_size=0.20, random_state=12, cv=10, optimization='Grid', scoring='accuracy')
````
> Now make predictions while using 'train', 'test', or 'unknown' dataset
````
clf.make_prediction(prediction_data='test')
````

> Get some useful plots:
````
clf.plot_correlation()
clf.plot_feature_imp()
clf.plot_roc()
````
## Parameters description:

> __init__()
````
data            = any     : Dataset including features and target  (default = None)
inputs          = any     : Feature set (default = None)
target          = any     : Target that needs to be predicted  (default = None)
normalization   = any     : Method for the scaling of data (default = "None"
verbosity       = integer : Degree for printing output messages in the terminal (default = 0, can be 0,1, or 2)
  
````  

> ML algorithms
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

> fit()
 ````
test_size       = float   : For specifying test fraction for dataset (default = 0.20)
random_no       = any     : Random number for reproducing the results    (default = None)
optimization    = str     : Method for searching the best hyperparameters for the model  (default = 'Grid'); Available methods are = 'Grid', 'Bayesian' and 'Randomized'
cv              = integer : cross-validation
scoring         = str     : Method for the evaluation of model; (default = 'roc_auc'); Available methods are = 'roc_auc', 'roc_auc_ovr', 'roc_auc_ovo', 'accuracy',  'roc_auc_ovr_weighted', 'roc_auc_ovr_weighted', 'f1', 'precision', 'recall'
return_dataset  = str     : Returns a csv file of training or test dataset (default = None); Available = 'train', 'test'													

````

> make_prediction()
````
prediction_data   = bool : Dataset to make predictions; only if predict_unknown is True (default = 'test')
unknown_data      = any  : unknown dataset to evaluate the performance of the trained model. Only when prediction_data = 'unknown' (default = None)
proba_prediction  = bool : Predicts the probabilities rather than the exact values as output if True (default = False)
save_csv          = bool : Whether to save a csv file of predictions or not (default = False)
file_name         = str  : Name for the csv file

````

> plot_correlation()
````
method      = str   : Method for plottting correlation matrix (default = 'pearson') Available methods = 'perason', 'kendall', or 'spearman'  
matrix_type = bool  : Type of correlation-matrix for plotting  (default = upper); Available = 'full', 'upper', 'lower'
annot       = bool  : Whether to show the correlation with numbers or not  (default = False)
cmap        = any   : Color map for plot  (default = coolwarm)
vmin        = float : Minimum value for color bar (default = -1.0)
vmax        = float : Maximum value for color bar (default =  1.0)
figsize     = tuple : Tuple of two integers for determining the figure size    (default =(16, 12))
fontsize    = int   : Font size of color-bar and x, y axis   (default =14)
save_fig    = bool  : save plot in the current working directory if True  (default = False)
save_csv    = bool  : save a csv file if True  (default = False)
figname     = str   : name of fig if save_fig is True  (default = "Correlation_plot.png")

````

> plot_feature_imp()
````
kind      = str   : Type of plot: (default = 'barh'); Available types = 'barh', 'bar', 'pie', 'line', 'area'  
random_no = any   : If want to set any random_state (default = None)
figsize   = tuple : Tuple of two integers for determining the figure size (default =(22, 16))		 
fontsize  = int   : Font size of color-bar and x, y axis (default =20)
color     = str   : Color for plot    (default = '#ff8000')	
lw        = float : Width of bars if kind == 'bar or barh' (default = 5.0)
save_fig  = bool  : Save plot in the current working directory if True (default = False)
figname   = str   : Name of fig if save_fig is True (default = "Feature_imp_Plot(MI).png")

````


> plot_roc()
````
figsize   = tuple : Tuple of two integers for determining the figure size  (default =(9, 7))		 
lines_fmt = dict  : Dictionary for the formatting of lines i.e. 'color' and linewidth('lw')	 (default = {'color': ["#339966", "#cc0000"], 'lw': 3}
label     = str   : Set label inside the plot (default = 'ROC_curve')
fontsize  = int   : Set fontsize for the x and y labels  (default = 18)
ticksize  = int   : Set fontsize for the x and y ticks   (default = 18)
fig_name  = str   : Name for the figure if want to save figure    (default = 'roc_plot')
save_fig  = bool  : Save Figure in the current directory if True    (default = False)

````
