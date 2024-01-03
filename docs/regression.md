### class Regression(Preprocessing)

Regression class inherits the Preprocessing class which means it has the same init parameters as the Preprocessing class.

#### def __init__(self, data=None, inputs=None, target=None, nan_values=None, normalization=None, test_size=0.20, random_state=None, label_encode_inputs=True, label_encode_target=False, problem=None, raise_warnings=True, verbosity=0)

````
data                 = Dataframe  : 	Dataset for evaluating a model  (default = None)
inputs               = Dataframe  :	List of features (default = None)
target               = Dataframe  : 	Target variable you want to predict  (default = None)
nan_values           = str        :	Whether to 'impute' or 'remove' NaN value in the dataset.(default=None)	
normalization        = str        :	Method for normalizing the dataset (default = "None")
test_size            = float      :	Size of test dataset (default = 0.20)
random_state         = int        :	Random number for reproducing the results (default = None)
label_encode_inputs  = bool       :	Convert categorical data into numerical data for the given inputs (default = True)
label_encode_target  = bool       :	Convert categorical data into numerical data for the target (default = False)
problem              = str        :	The type of problem to solve (i.e 'Classification', or 'Regression')
raise_warnings       = bool       :	Whether to raise any warnings or not (default = True)
verbosity            = integer    :	Degree for printing output messages in the terminal (default = 0, possible values are 0,1, or 2)

````


#### def Model(model_name=None, params=None, proba=False, random_state=None)

Select a ML model for regression

````
model_name    = str   :  Name of the machine learning algorithm to use (default = None)
params        = dict  :  Set of parameters for machine learning algrorithm (default = None)
proba         = bool  :  Only needed for 'NuSVC' and 'SVC' when predicting probability (default = False)
random_state  = int   :  Random number for reproducing the results (default = None)
````

#### def fit(self, optimization='Bayesian', num_iter=20, cv=10, scoring='roc_auc')

Fit the ML model on your dataset

````
optimization  = str  :	Method for searching the best hyperparameters for the ML model (default = 'Grid', other available methods are 'Randomized' and 'Bayesian')
num_iter      = int  :	Number of iterations to run for hyperparameter optimization (default = 20).
cv            = int  :	cross-validation (default = 10)
scoring       = str  :	Method for the evaluation of model: (default = 'roc_auc')
````

#### def predict(self, prediction_data='test', unknown_data=None, proba_prediction=False, save_csv=False, file_name='predicted_data')

Make predictions for new datasets using the trained ML model

````
prediction_data  = bool       :	Dataset to make predictions (default = 'test'). Other available options are 'train' and 'unknown'.
unknown_data     = Dataframe  :	Unknown dataset for predictions; required when prediction_data is 'unknown' (default = None)
proba_prediction = bool       :	Predict probabilities rather than the exact values for the target if set True (default = False)          
save_csv         = bool       :	Save a csv file of predictions if set True (default = False)
file_name        = str        :	Name for the csv file (default = 'predicted_data')
````

#### def plot_correlation(self, method='pearson', matrix_type='upper', annot=False, cmap='coolwarm', vmin=-1.0, vmax=1.0, figsize=(12, 8), fontsize=14, save_fig=False, save_csv=False, fig_name="Correlation_plot.png", dpi=300):

Get Correlation values between inputs.

````
method        = str    :  Method for plottting correlation matrix (default = 'pearson'). Other available methods are 'perason', 'kendall', or 'spearman'  
matrix_type   = bool   :  Type of correlation-matrix for plotting  (default = upper); Available = 'full', 'upper', 'lower'
annot         = bool   :  Print the correlation values in the heatmap if set True  (default = False)
cmap          = any    :  Color map for plot  (default = coolwarm)
vmin          = float  :  Minimum value for color bar (default = -1.0)
vmax          = float  :  Maximum value for color bar (default =  1.0)
figsize       = tuple  :  Tuple of two integers for determining the figure size    (default =(16, 12))
fontsize      = int    :  Font size of color-bar and x, y axis   (default =14)
save_fig      = bool   :  Save plot in the current working directory if True  (default = False)
save_csv      = bool   :  Save a csv file if True  (default = False)
figname       = str    :  Name of fig if save_fig is True  (default = "Correlation_plot.png")
dpi           = str    :  Quality of the figure  (default = 600)
````

#### def plot_feature_imp(self, kind="barh", random_no=None, figsize=(12, 8), fontsize=20, color='#ff8000', lw=5.0, save_fig=False, fig_name="Feature_imp_Plot(MI).png", dpi=300):

Plot importance of each selected feature towards the prediction of the target. It uses mutual importance function of Scikit-learn.

````
kind       = str    :   Type of plot: (default = 'barh'); Available types = 'barh', 'bar', 'pie', 'line', 'area'  
random_no  = any    :	  Random number to reproduce results (default = None)
figsize    = tuple  :   Tuple of two integers for determining the figure size (default =(22, 16))		 
fontsize   = int    :	  Font size of color-bar and x, y axis (default =20)
color      = str    :   Color for plot    (default = '#ff8000')	
lw         = float  :   Width of bars if kind == 'bar' or 'barh' (default = 5.0)
save_fig   = bool   :   Save plot in the current working directory if True (default = False)
figname    = str    :	  Name of fig if save_fig is True (default = "Feature_imp_Plot(MI).png")

````

#### def plot_scatter(self, plot_for="test", facecolor='red', alpha=0.5, marker='o', xlabel='True', ylabel='Predicted', title='Regression_plot', save_fig=True, fig_name="Scatter_plot", dpi=300)

Plot a scatter diagram for the defined dataset.

````
plot_for    = str    :   Determines whether to plot results for training or testing dataset (default='test')
facecolor   = str    :	 Set color for the marker
alpha       = float  :   Determine the intensity of colors
marker      = str    :   shape of the points in the figure
xlabel      = bool   :   Label for x-axis
ylabel      = bool   :   Label for y-axis
title       = str    :   Title of the figure
save_fig    = bool   :   Name of the file to save figure
dpi         = int    :   Determine the quality of the figure to save
````

#### def save_data(self, filename=None, verbosity=2)

Save outputs as a text file.

````
file_name  = str  : 	Name for the file (default='classification_data')
verbosity  = int  :   Quantity of the data you want to save (default=2). Other options are 0 and 1.
````

