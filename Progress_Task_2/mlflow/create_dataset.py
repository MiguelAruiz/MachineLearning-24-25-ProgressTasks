

######################################## IMPORTS ########################################
import pandas as pd
import configparser

#########################################################################################


################################## Global Configurations ################################
# Load configuration fileconfig["data"]["targets"]
CONFIG_FILE_PATH = "mlflow/test.conf"

config = configparser.ConfigParser()
config.read(CONFIG_FILE_PATH)
DATASET_PATH = config["data"]["dataset_path"]
TEST_DATASET_PATH = config["data"]["dataset_test_path"]
DATASET_INDEX_FEATURE = config["data"]["dataset_index"]
DATASET_TARGET_FEATURES = ["h1n1_vaccine", "seasonal_vaccine"]

#########################################################################################
class Dataset:
    '''
    ## Dataset
    
    This class represents a dataset. It handles dataset loading and splitting.
    
    ### Attributes
    
    - test: The test dataset.
    
    '''
    def __init__(self):
        '''
        Constructor for the Dataset class.
        '''
        data = pd.read_csv(DATASET_PATH)
        target = DATASET_TARGET_FEATURES
        data.set_index(DATASET_INDEX_FEATURE, inplace=True)
        self._y = data[target]
        self._X = data.drop(columns=target)
        test_data =  pd.read_csv(TEST_DATASET_PATH)
        test_data.set_index(DATASET_INDEX_FEATURE, inplace=True)
        self.test = test_data
    
    def with_correlation(self):
        '''
        ## with_correlation
        Method that returns a copy of the dataset features and targets.
        
        ### Returns
        (X, y): A tuple containing the dataset features and targets.
        '''
        
        return self._X.copy(), self._y.copy()
    