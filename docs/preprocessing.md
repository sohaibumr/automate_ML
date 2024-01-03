
### class Preprocessing:

#### def __init__(self, data=None, inputs=None, target=None, nan_values=None, normalization=None, test_size=0.20, random_state=None, label_encode_inputs=True, label_encode_target=False, problem=None, raise_warnings=True, verbosity=0)

````
data                 = Dataframe  : 	Dataset for evaluating a model  (default = None)
inputs               = Dataframe  :	Feature set (default = None)
target               = Dataframe  : 	Target which you want to predict  (default = None)
nan_values           = str        :	Whether to 'impute' or 'remove' NaN value in the dataset.(default=None)	
normalization        = str        :	Method for normalizing the dataset (default = "None")
test_size            = float      :	Size od testing dataset (default = 0.20)
random_state         = int        :	Random number for reproducing the results (deafult = None)
label_encode_inputs  = bool       :	Convert categorical data into numerical data in the given inputs (default = True)
label_encode_target  = bool       :	Convert categorical data into numerical data in the target (default = False)
problem              = str        :	The type of problem to solve (i.e 'Classification', or 'Regression')
raise_warnings       = bool       :	Whether to raise any warnings or not (default = True)
verbosity            = integer    :	Degree for printing output messages in the terminal (default = 0, possible values are 0,1, or 2)

````
