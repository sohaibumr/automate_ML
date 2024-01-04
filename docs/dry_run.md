
### class all_Models(Preprocessing)

This class runs all the models that are implemented here. This class inherits the 'Proprocessing' class which means it uses all the parameters of class 'Preprocessing'.

#### def best_model(self, n_splits=100, scoring=None, save_txt=True, filename='Models_score', show=True, save_fig=False, fig_name='Models Comparison', dpi=600)

````
n_splits    = int    : 	No of splits  (default=100)		 
scoring     = str    : 	Scoring method  (default=None)
save_txt    = bool   :	Save a txt file with model names and corresponding scores (default=True)
filename    = str    : 	Name of the txt file   (default='Models_score')
show        = bool   : 	Print out the sorted table and plots a bar chart of the models with corresponding scores if set True (default=True)
save_fig    = bool   : 	Whether to save the figure or not (default=False)
fig_name    = str    : 	Name of the png file (default=Models Comparison)
dpi         = int    : 	Determines the quality of the figure (default=600)
````
