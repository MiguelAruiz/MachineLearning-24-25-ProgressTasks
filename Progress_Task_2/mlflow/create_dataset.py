import pandas as pd

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
    