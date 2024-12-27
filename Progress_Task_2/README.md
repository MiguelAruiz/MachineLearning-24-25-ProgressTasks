# Progress Task 2

## Task Description

The objective is to build a predictive model that estimates the likelihood of individuals receiving two different vaccines: the H1N1 flu vaccine and the seasonal flu vaccine.

All the information about the task is on the following site: [competition website](https://www.drivendata.org/competitions/66/flu-shot-learning/page/210/)

### Ranking 🏅
TEAM NAME: Gatopizza

FINAL POSITION: 478th 

SCORE: 0.8606

MODEL: Catboost with Optuna

## Important Note

`Model_Applications.ipynb` contains a summary of the task. The other notebooks contain the code used to build each model and further explanations.  

The experiment is not reproducible, this means that using our best parameters does not guarantee the same score.


## Installation

To install the required libraries, run the following command:

```bash
pip install -r requirements.txt
```

## Structure

```
📂 .    
    ├──── 📂 data
    ├──── 📂 mlflow
    ├──── 📄 Catboost.ipynb
    ├──── 📄 KNN.ipynb
    ├──── 📄 LightGBM.ipynb
    ├──── 📄 MLPClassifier.ipynb    
    ├──── ⚙️ make.bat / Makefile        
    ├──── 📄 Model_Applications.ipynb
    ├──── 📄 NeuralNetworks.ipynb
    ├──── 📄 Preprocessing.ipynb
    ├──── 📄 RandomForest.ipynb   
    ├────  ℹ️ README.md
    ├──── 📄 competition_webscrapper.py  
    ├──── 📄 requirements.txt  
    └──── ⚙️ test.conf 
```

* 📂**data**: Contains the dataset used in the notebooks.
* 📂**mlflow**: Contains tests with the MLflow library.
* ⚙️**make.bat / Makefile**: Contains the commands to run the mlflow server.
* ❗**Model_Applications.ipynb**: Contains the a summary of the task.
* **Preprocessing.ipynb**: Contains the exploratory data analysis and the preprocessing of the dataset.
* **competition_webscrapper.py**: Contains the code to get the ranking of the competition.

> [!IMPORTANT]  
> Model_Applications.ipynb is the first notebook you should look at


## Explanation of the datasets

| File name                      | Description                          |
|--------------------------------|--------------------------------------|
| df_encoded.csv                 | Dataset without features with random values (employment_industry) and encoded with OrdinalEncoder. **This is our first and main dataset**                   |
| df_encoded_all.csv             | All features with OrdinalEncoder          |
| df_encoded_no_outliers.csv     | All features with OrdinalEncoder and outliers removed with DBSCAN       |
| df_no_null.csv                 | Original dataset without missing values, no encoding              |
| submission_format.csv          | Summision format example                    |
| test_encoded.csv               | Same as `df_encoded`          |
| test_encoded_all.csv           | Same as `df_encoded_all`|
| test_no_null.csv               | Same as `df_no_null`    |
| test_set_features.csv          | Dataset to upload to competition |
| training_set_features.csv      | Original dataset |
| training_set_labels.csv        | Original dataset |
