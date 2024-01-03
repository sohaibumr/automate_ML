
### class Preprocessing:

This class performs the preprocessing on the provided dataset i.e. deal with nan values, normalization of dataset, split the data into train and test subsets and so on.

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

#### def preprocessed_data(self, save_csv=True, filename='Preprocessed_data'):

````
save_csv      = bool :  Whether to save the preprocessed data or not  (default = True)
filename      = str  :  Name of the file (default = None)
````

#### def input_data(self, save_csv=True, filename='Input_data')

````
save_csv      = bool :  Whether to save the inputs data or not  (default = True)
filename      = str  :  Name of the file (default = None)
````
#### def target_data(self, save_csv=True, filename='Input_data'):

````
save_csv      = bool :  Whether to save the target data or not  (default = True)
filename      = str  :  Name of the file (default = None)
````
#### def train_test_data()

Return X_train, X_test, y_train, y_test
