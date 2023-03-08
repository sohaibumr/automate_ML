import pandas as pd
import numpy as np
import warnings

"""                                 Pre-processing tools                                     """

from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
from datetime import datetime


global X_train, X_test, y_train, y_test

"""""""""""""""""""""""""""""""""""""""""""""""""Classifiers"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def timenow():
    T = str(datetime.now())
    t = T.replace(':', '')
    return t


class Preprocessing:

    def __init__(self, data=None, inputs=None, target=None, nan_values=None, normalization=None, test_size=0.20,
                 random_state=None, label_encode_inputs=True, label_encode_target=False, problem=None,
                 raise_warnings=True, verbosity=0):

        global X_train, X_test, y_train, y_test

        if raise_warnings:
            pass
        else:
            warnings.filterwarnings("ignore")

        if data is None:
            raise NameError("'data' is not defined. Define a dataframe.")
        else:
            if isinstance(data, pd.DataFrame):
                if verbosity == 2:
                    print('\n', '\033[0;30;44m' + '         Original DataFrame          ' + '\033[0m')
                    print(data)
                else:
                    pass
            else:
                raise TypeError("Given data type '{}' is not supported. Provide a pandas dataframe".format(data.dtype))

        if data.isnull().values.any():
            if nan_values == 'impute':
                data.fillna(data.mean(), inplace=True)
                self.data = data
            elif nan_values == 'remove':
                data.dropna(inplace=True)
                data.reset_index(drop=True)
                self.data = data
            elif nan_values == 'interpolate':
                data.interpolate(method='linear')
            elif nan_values is None:
                raise ValueError("'data' contain NaN values, remove or impute the NaN values")
            else:
                error = "'nan_values' can either be 'impute' or 'remove' and you passed '{}'".format(nan_values)
                raise NameError(error)

        else:
            self.data = data

        if verbosity == 2:
            print(self.data)

        if inputs is None:
            raise NameError("'inputs' is not defined")
        else:
            if label_encode_inputs:
                columns = self.data[inputs]
                ColumnsToEncode = list(columns.select_dtypes(include=['category', 'object']))
                le = LabelEncoder()
                for feature in ColumnsToEncode:
                    try:
                        self.data[feature] = le.fit_transform(self.data[feature])
                    except:
                        Le_error = 'Error encoding ' + feature
                        raise AssertionError(Le_error)
                self.data = self.data
            if verbosity == 2:
                print('\n', '\033[0;30;44m' + '         Unscaled Cleaned DataFrame          ' + '\033[0m')
                print(self.data[inputs], '\n')

            if normalization == 'zscore':
                if verbosity in [1, 2]:
                    print('\n', '\033[0;30;44m' + '         DataFrame Scaled with zscore            ' + '\033[0m')
                norm_data = zscore(self.data[inputs])
                self.inputs = pd.DataFrame(norm_data, columns=self.data[inputs].columns)
            elif normalization == 'minmax':
                if verbosity in [1, 2]:
                    print('\n', '\033[0;30;44m' + '         DataFrame Scaled with minmax            ' + '\033[0m')
                minmax_data = MinMaxScaler()
                norm_data = minmax_data.fit_transform(self.data[inputs])
                self.inputs = pd.DataFrame(norm_data, columns=self.data[inputs].columns)
            elif normalization is None:
                self.inputs = pd.DataFrame(self.data[inputs], columns=self.data[inputs].columns)
                # self.inputs = self.data[inputs]
            else:
                raise NameError(
                    "{} is not a valid normalization method. Use 'minmax', 'zscore' or None".format(normalization))
            if verbosity == 2:
                print(self.inputs, '\n')

        if target is None:
            raise NameError("'target' is not defined")
        elif label_encode_target:
            le_t = LabelEncoder()
            self.target = le_t.fit_transform(self.data[target])
            if verbosity in [1, 2]:
                print("Original Classes", le_t.classes_)
                print("Transformed Classes", sorted(list(Counter(self.target).keys())))
        else:
            self.target = self.data[target]


        if verbosity == 2:
            print(self.target)

        self.normalization = normalization
        self.test_size = test_size
        self.random_state = random_state
        self.problem = problem
        self.verbosity = verbosity


        if problem is None:
            raise AttributeError("Define a problem to solve e.g. 'Classification' or 'Regression'")
        elif problem == 'Classification':
            X_train, X_test, y_train, y_test = train_test_split(self.inputs, self.target, test_size=self.test_size,
                                                            stratify=self.target, random_state=self.random_state)
            if self.verbosity in [1, 2]:
                print("target", Counter(self.target))
                print("y_train:", Counter(y_train), ',', "y_test:", Counter(y_test), '\n')
            else:
                pass

        elif problem == 'Regression':
            X_train, X_test, y_train, y_test = train_test_split(self.inputs, self.target, test_size=self.test_size,
                                                        random_state=self.random_state)
        else:
            raise NameError("problem should be either 'Classification' or 'Regression' and you provided {}".format(problem))

        if self.verbosity in [1, 2]:
            print("Training Dataset size: ", np.shape(X_train))
            print("Test Dataset size: ", np.shape(X_test))
        else:
            pass


        """
        Parameters:

        data                    = Dataframe		: 	Dataset for evaluating a model  (default = None)
        inputs                  = Dataframe		:	Feature set (default = None)
        target                  = Dataframe		: 	Target which you want to predict  (default = None)
        nan_values              = str		    :	Whether to 'impute' or 'remove' NaN value in the dataset.(default=None)	
        normalization           = str 	        :	Method for normalizing the dataset (default = "None")
        test_size               = float		    :	Size od testing dataset (default = 0.20)
        random_state            = int		    :	Random number for reproducing the results (deafult = None)
        label_encode_inputs     = bool	        :	Convert categorical data into numerical data in the given inputs (default = True)
        label_encode_target     = bool	        :	Convert categorical data into numerical data in the target (default = False)
        problem                 = str	        :	The type of problem to solve (i.e 'Classification', or 'Regression')
        raise_warnings          = bool	        :	Whether to raise any warnings or not (default = True)
        verbosity               = integer		:	Degree for printing output messages in the terminal 
                                                    (default = 0, possible values are 0,1, or 2)

        """

    def preprocessed_data(self, save_csv=True, filename='Preprocessed_data'):
        if save_csv:
            self.data.to_csv(filename+".csv")
        else:
            pass

        return self.data

    def input_data(self, save_csv=True, filename='Input_data'):
        if save_csv:
            self.inputs.to_csv(filename+".csv")
        else:
            pass

        return self.inputs


    def target_data(self, save_csv=True, filename='Input_data'):
        if save_csv:
            self.target.to_csv(filename+".csv")
        else:
            pass

        return self.target

    @staticmethod
    def return_dataset(return_data='train'):
        if return_data == 'train':
            df = pd.DataFrame(X_train)
            df.index = X_train.index
            df.to_csv('Train_data' + '_' + timenow() + '.csv')
        elif return_data == 'test':
            df = pd.DataFrame(X_test)
            df.index = X_train.index
            df.to_csv('Test_data' + '_' + timenow() + '.csv')
        else:
            error = "return_dataset can either be 'train' or 'test' while you passed '{}'".format(return_data)
            raise NameError(error)

        return df

    @property
    def multiclass(self):
        count_classes = Counter(self.target).keys()
        if len(Counter(count_classes).keys()) > 2:
            return True
        else:
            return False


def train_test_data():
    return X_train, X_test, y_train, y_test