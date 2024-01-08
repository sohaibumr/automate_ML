### class Classification(Preprocessing)

Classification class inherits the Preprocessing class which means it has the same parameters as the Preprocessing class.

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

Select a ML model for classification

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

#### def Confusion_matrix(self, show_plot=True, annot=True, cmap='Blues', figsize=(12, 8), fontsize=14, save_fig=False, fig_name="Confusion_matrix.png", xlabel='Predicted Values', ylabel='Actual Values', title='Seaborn Confusion Matrix with labels\n', dpi=300)

Get confusion matrix

````
show_plot  = bool  :  Whether to show the plot or not (default = True).
annot      = bool  :  Print the confusion matrix values inside the heatmap if set True  (default = False)
cmap       = any   :  Color map for plot  (default = 'Blues')
figsize    = tuple :  Tuple of two integers for determining the figure size    (default =(16, 12))
fontsize   = int   :  Font size of color-bar and x, y axis   (default =14)
save_fig   = bool  :  Save plot in the current working directory if True  (default = False)
figname    = str   :  Name of fig if save_fig is True  (default = "Confusion_matrix.png")
xlabel     = str   :  Title for x-axis  (default = "Predicted Values")
ylabel     = str   :  Title for y-axis  (default = "Actual Values")
title      = str   :  Title for the figure  (default = "Seaborn Confusion Matrix with labels")
dpi        = str   :  Quality of the figure  (default = 600)
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
kind       = str    :  Type of plot: (default = 'barh'); Available types = 'barh', 'bar', 'pie', 'line', 'area'  
random_no  = any    :  Random number to reproduce results (default = None)
figsize    = tuple  :  Tuple of two integers for determining the figure size (default =(22, 16))		 
fontsize   = int    :  Font size of color-bar and x, y axis (default =20)
color      = str    :  Color for plot    (default = '#ff8000')	
lw         = float  :  Width of bars if kind == 'bar' or 'barh' (default = 5.0)
save_fig   = bool   :  Save plot in the current working directory if True (default = False)
figname    = str    :  Name of fig if save_fig is True (default = "Feature_imp_Plot(MI).png")

````

#### def plot_roc_curve(self, plot_for='test', figsize=(9, 7), lines_fmt=None, label='ROC_curve', fontsize=18, ticksize=18, xlabel='False positive rate', ylabel='True positive rate', legend='lower right', alpha=0.8, save_fig=False, fig_name='roc_plot', dpi=300):

Plot an roc_curve for the defined dataset.

````
plot_for    = str    :   Determines whether to plot results for training or testing dataset (default='test')     
figsize     = tuple  : 	 Tuple of two integers for determining the figure size  (default=(9, 7))		 
lines_fmt   = dict   : 	 Dictionary for the formatting of lines i.e. 'color' and linewidth('lw') (default={'color': ["#339966", "#cc0000"], 'lw': 3}
label       = str    :	 Set label inside the plot (default ='ROC_curve')
fontsize    = int    : 	 Set fontsize for the x and y labels  (default=18)
xlabel      = str    :   Label for x-axis (default=False positive rate)
ylabel      = str    :   Label for y-axis (default=True positive rate)
legend      = str    :   Text to right in the legend (default='lower right')
alpha       = float  :   Intensity of colors (default=0.8)
ticksize    = int    :	 Set fontsize for the x and y ticks   (default=18)
save_fig    = bool   :   Save Figure in the current directory if set True    (default=False)
fig_name    = str    : 	 Name for the figure     (default='roc_plot')
dpi         = int    :   Quality of the image (default=600)

````

#### def save_data(self, filename=None, verbosity=2)

Save outputs as a text file.

````
file_name  = str  :  Name for the file (default='classification_data')
verbosity  = int  :  Quantity of the data you want to save (default=2). Other options are 0 and 1.
````
