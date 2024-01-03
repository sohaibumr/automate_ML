
### def Model(model_name=None, params=None, proba=False, random_state=None)

````
model_name          = str		   : 	Name of the machine learning algorithm to use (default = None)

params              = dict		 :	  Set of parameters for machine learning algrorithm (default = None)

proba               = bool		 : 	Only needed for 'NuSVC' and 'SVC' when predicting probability (default = False)

random_state        = int		   :	  Random number for reproducing the results (default = None)
````

### def fit(self, optimization='Bayesian', num_iter=20, cv=10, scoring='roc_auc')

"""
optimization 		= str		:	Method for searching the best hyperparameters for the ML model (default = 'Grid', other available methods are                                 'Randomized' and 'Bayesian')

num_iter    		= int		:	Number of iterations to run for hyperparameter optimization (default = 20).

cv 			 		    = int		:	cross-validation (default = 10)

scoring 	 		  = str  	:	Method for the evaluation of model: (default = 'roc_auc')
"""

### def predict(self, prediction_data='test', unknown_data=None, proba_prediction=False, save_csv=False, file_name='predicted_data')

"""
prediction_data		= bool		  :	Dataset to make predictions (default = 'test')

unknown_data		  = Dataframe	:	Unknown dataset for predictions; required when prediction_data is 
                                    'unknown' (default = None)
                                    
proba_prediction	= bool		  :	Predict probabilities rather than the exact values for the target if set 
                                    True (default = False)
                                    
save_csv	 		    = bool		  :	Save a csv file of predictions if set True (default = False)

file_name	 		    = str		    :	Name for the csv file (default = 'predicted_data')
"""

### def Confusion_matrix(self, show_plot=True, annot=True, cmap='Blues', figsize=(12, 8), fontsize=14, save_fig=False, fig_name="Confusion_matrix.png", xlabel='Predicted Values', ylabel='Actual Values', title='Seaborn Confusion Matrix with labels\n', dpi=300)

"""
show_plot   = bool  :   Whether to show the plot or not (default = True).

annot		    = bool 	:	  Print the confusion matrix values inside the heatmap if set True  (default = False)

cmap 		    = any  	: 	Color map for plot  (default = 'Blues')

figsize 	  = tuple : 	Tuple of two integers for determining the figure size    (default =(16, 12))

fontsize 	  = int  	:	  Font size of color-bar and x, y axis   (default =14)

save_fig 	  = bool 	: 	Save plot in the current working directory if True  (default = False)

figname 	  = str   :	  Name of fig if save_fig is True  (default = "Confusion_matrix.png")

xlabel 	    = str   :	  Title for x-axis  (default = "Predicted Values")

ylabel	    = str   :	  Title for y-axis  (default = "Actual Values")

title 	    = str   :	  Title for the figure  (default = "Seaborn Confusion Matrix with labels")

dpi 	      = str   :	  Quality of the figure  (default = 600)
"""

### def plot_correlation(self, method='pearson', matrix_type='upper', annot=False, cmap='coolwarm', vmin=-1.0, vmax=1.0, figsize=(12, 8), fontsize=14, save_fig=False, save_csv=False, fig_name="Correlation_plot.png", dpi=300):

"""
method 		  = str  	: 	Method for plottting correlation matrix (default = 'pearson'). Other available methods are 'perason', 'kendall', or                               'spearman'  

matrix_type	= bool 	:	  Type of correlation-matrix for plotting  (default = upper); Available = 'full', 'upper', 'lower'

annot		    = bool 	:	  Print the correlation values in the heatmap if set True  (default = False)

cmap 		    = any  	: 	Color map for plot  (default = coolwarm)

vmin		    = float	:	  Minimum value for color bar (default = -1.0)

vmax		    = float	:	  Maximum value for color bar (default =  1.0)

figsize 	  = tuple : 	Tuple of two integers for determining the figure size    (default =(16, 12))

fontsize 	  = int  	:	  Font size of color-bar and x, y axis   (default =14)

save_fig 	  = bool 	: 	Save plot in the current working directory if True  (default = False)

save_csv 	  = bool 	: 	Save a csv file if True  (default = False)

figname 	  = str   :	  Name of fig if save_fig is True  (default = "Correlation_plot.png")

dpi 	      = str   :	  Quality of the figure  (default = 600)
"""



