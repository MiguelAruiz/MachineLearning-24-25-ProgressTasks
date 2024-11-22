# Progress Task 1

## Task Description

Using a dataset of wine characteristics, the objective of this task is to select an apropiate regression and classification model to predict the quality of wine. 

## Installation

To install the required libraries, run the following command:

```bash
pip install -r requirements.txt
```

> [!NOTE]
> There's a problem with this version of pygam that only works on Linux distributions.

## Structure

```
📂 .    
    ├──── 📄 Classification_models.ipynb    
    ├──── 📄 Preprocessing.ipynb  
    ├──── 📄 README.md  
    ├──── 📄 Regression_models.ipynb  
    ├──── 📂 img_classification  
    │      ├── 🖼️ DecisionTree.png  
    │      ├── 🖼️ GaussianNB.png  
    │      ├── 🖼️ MultinomialNB.png  
    │      ├── 🖼️ RandomForest.png  
    │      └── 🖼️ kNN.png  
    ├──── 📂 img_regression  
    │      ├── 🖼️ Multilinear Lasso.png  
    │      ├── 🖼️ Multilinear Ridge.png  
    │      ├── 🖼️ Polynomial Regression.png  
    │      └── 🖼️ Random Forest.png  
    └──── 📄 requirements.txt  
```

* **Classification_models.ipynb**: Jupyter notebook with the classification models.
* **Preprocessing.ipynb**: Jupyter notebook with the preprocessing of the dataset.
* **Regression_models.ipynb**: Jupyter notebook with the regression models.
* **img_classification**: Folder with the images of the confussion matrix for each classification model.
* **img_regression**: Folder with the images of the plots for the regression models.


## Some notes about the execution of the notebooks

The file `Preprocessing.ipynb` contains explanations and the exploratory data analysis of the dataset. Its execution does not influence the other two notebooks.

The file `Regression_models.ipynb` contains an execution with the Generative Additive Model (GAMs), the current version of the library thorws an error when trying to execute the codeblocks. Take this into account when executing the whole notebook.