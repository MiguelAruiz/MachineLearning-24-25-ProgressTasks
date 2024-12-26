import re
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class Dataset:
    def __init__(self):
        data = pd.read_csv("../data/df_encoded.csv")
        target = ["h1n1_vaccine","seasonal_vaccine"]
        data.set_index("respondent_id", inplace=True)
        self._y = data[target]
        self._X = data.drop(columns=target)
        test_data =  pd.read_csv("../data/test_encoded.csv")
        test_data.set_index("respondent_id", inplace=True)
        self.test = test_data
    
    def with_correlation(self):
        return self._X.copy(), self._y.copy(), self.test.copy()
    
    def filter(self):
        X = pd.read_csv('../data/X_filtered.csv')
        y = pd.read_csv('../data/y_filtered.csv')
        X.set_index("respondent_id", inplace=True)
        y.set_index("respondent_id", inplace=True)
        return X, y
    
    def encoded(self):
        X = pd.read_csv('../data/new.csv', index_col="respondent_id")
        target = ["h1n1_vaccine","seasonal_vaccine"]
        y = X[target]
        X = X.drop(columns=target)
        test = pd.read_csv('../data/new_test.csv', index_col="respondent_id")
        return X, y, test
    
    def with_onehot(self):
        '''
        ## with_onehot
        Method that returns a copy of the dataset features and targets with one-hot encoding.

        ### Returns
        (X, y, test): A tuple containing the dataset features, targets and test dataset with one-hot encoding.
        '''
        encoder = OneHotEncoder()
        all_features = pd.concat([self._X, self.test])
        encoder.fit(all_features)
        X = encoder.transform(self._X)
        y = self._y
        test_transformed = encoder.transform(self.test)
        test_df = pd.DataFrame(test_transformed.toarray(), index=self.test.index)
        return X, y, test_df

    def no_outliers(self):
        '''
        ## no_outliers
        Method that returns the dataset encoded with all the features removing some outliers.

        ### Returns
        X, y, test
        '''
        X = pd.read_csv('../data/df_encoded_no_outliers.csv', index_col="respondent_id")
        target = ["h1n1_vaccine","seasonal_vaccine"]
        y = X[target]
        X = X.drop(columns=target)
        return X, y, self.test.copy()
    def no_outliers_onehot(self):
        '''
        ## no_outliers_onehot
        Method that returns the dataset encoded with all the features removing some outliers and one-hot encoding.

        ### Returns
        X, y, test
        '''
        X = pd.read_csv('../data/df_encoded_no_outliers.csv', index_col="respondent_id")
        target = ["h1n1_vaccine","seasonal_vaccine"]
        y = X[target]
        X = X.drop(columns=target)
        encoder = OneHotEncoder()
        all_features = pd.concat([X, self.test])
        encoder.fit(all_features)
        encoder.fit(X)
        X = encoder.transform(X)
        test_transformed = encoder.transform(self.test)
        test_df = pd.DataFrame(test_transformed.toarray(), index=self.test.index)
        return X, y, test_df
    
    def all_features(self):
        '''
        ## all_features
        Method that returns the dataset encoded with all the features without dropping the ones with random values.

        ### Returns
        X, y, test
        '''
        X = pd.read_csv("../data/df_encoded_all.csv", index_col="respondent_id")
        target = ["h1n1_vaccine","seasonal_vaccine"]
        y = X[target]
        X = X.drop(columns=target)
        test = pd.read_csv("../data/test_encoded_all.csv", index_col="respondent_id")
        assert X.columns.equals(test.columns), "Columns are not equal"
        return X, y, test
    
    def all_onehot(self):
        X,y,test = self.all_features()
        encoder = OneHotEncoder()
        all_features = pd.concat([X, test])
        encoder.fit(all_features)
        encoder.fit(X)
        X_transformed = encoder.transform(X)
        X_df = pd.DataFrame(X_transformed.toarray(), index=X.index)
        test_transformed = encoder.transform(test)
        test_df = pd.DataFrame(test_transformed.toarray(), index=test.index)
        return X, y, test_df
    
    def original_dataset(self):
        '''
        ## original_dataset
        Method that returns the original dataset without NaN values.

        ### Returns
        X, y, test
        '''
        X = pd.read_csv("../data/df_no_null.csv", index_col="respondent_id")
        target = ["h1n1_vaccine","seasonal_vaccine"]
        y = X[target]
        X = X.drop(columns=target)
        test = pd.read_csv("../data/test_no_null.csv", index_col="respondent_id")
        return X, y, test