# Progress Task 2

## Task Description

The objective is to build a predictive model that estimates the likelihood of individuals receiving two different vaccines: the H1N1 flu vaccine and the seasonal flu vaccine.

All the information about the task is on the following site: [competition website](https://www.drivendata.org/competitions/66/flu-shot-learning/page/210/)

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
    ├──── 📄 KNN.ipynb
    ├──── 📄 MLPClassifier.ipynb    
    ├──── ⚙️ make.bat / Makefile        
    ├──── 📄 Model_Applications.ipynb
    ├──── 📄 Preprocessing.ipynb
    ├──── 📄 RandomForest.ipynb   
    ├────  ℹ️ README.md  
    ├──── 📄 requirements.txt  
    └──── ⚙️ test.conf 
```

* 📂**data**: Contains the dataset used in the notebooks.
* 📂**mlflow**: Contains tests with the MLflow library.
* ⚙️**make.bat / Makefile**: Contains the commands to run the mlflow server.
* **KNN.ipynb**: Contains the implementation of the K-Nearest Neighbors algorithm.
* **MLPClassifier.ipynb**: Contains the implementation of the Multi-layer Perceptron Classifier algorithm.
* ❗**Model_Applications.ipynb**: Contains the a summary of the task.
* **Preprocessing.ipynb**: Contains the exploratory data analysis and the preprocessing of the dataset.
* **RandomForest.ipynb**: Contains the implementation of the Random Forest algorithm.

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
