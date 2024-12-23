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
        return self._X.copy(), self._y.copy()
    

    def with_onehot(self):
        encoder = OneHotEncoder()
        all_features = pd.concat([self._X, self.test])
        encoder.fit(all_features)
        X = encoder.transform(self._X)
        y = self._y
        test_transformed = encoder.transform(self.test)
        test_df = pd.DataFrame(test_transformed.toarray(), index=self.test.index)
        return X, y, test_df
    
    def with_division(self):
        h1_n1 = self._X.copy()
        seasonal = self._X.copy()
        h1_columns = ['doctor_recc_seasonal','opinion_seas_vacc_effective','opinion_seas_risk','opinion_seas_sick_from_vacc']
        seasonal_columns = ['h1n1_concern','h1n1_knowledge','doctor_recc_h1n1','opinion_h1n1_vacc_effective','opinion_h1n1_risk','opinion_h1n1_sick_from_vacc']
        h1_n1.drop(columns=h1_columns, inplace=True)
        seasonal.drop(columns=seasonal_columns, inplace=True)
        h1_n1_y = self._y['h1n1_vaccine']
        seasonal_y = self._y['seasonal_vaccine']
        test_h1_n1 = self.test.copy()
        test_seasonal = self.test.copy()
        test_h1_n1.drop(columns=h1_columns, inplace=True)
        test_seasonal.drop(columns=seasonal_columns, inplace=True)
        return (h1_n1, h1_n1_y, test_h1_n1), (seasonal, seasonal_y, test_seasonal)
