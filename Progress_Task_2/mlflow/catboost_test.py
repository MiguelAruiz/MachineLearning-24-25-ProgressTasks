import optuna
from catboost import CatBoostClassifier
import mlflow
import mlflow.data
import mlflow.data.pandas_dataset
from mlflow.models import infer_signature
from mlflow.data.pandas_dataset import PandasDataset
from sklearn.model_selection import StratifiedKFold

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from create_dataset import Dataset
from sklearn.metrics import roc_auc_score, accuracy_score

#####################################################################################3
data = Dataset()
X, y, output = data.original_dataset()

categorical_columns = list(X.select_dtypes(include=['object']).columns)
X[categorical_columns] = X[categorical_columns].astype('category')
output[categorical_columns] = output[categorical_columns].astype('category')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
h1_train = y_train["h1n1_vaccine"]
seasonal_train = y_train["seasonal_vaccine"]
h1_test = y_test["h1n1_vaccine"]    
seasonal_test = y_test["seasonal_vaccine"]

def objective_vaccine(trial, target):

    param_dist_random = {
        'iterations': trial.suggest_int('iterations', 500, 5000),
        'depth': trial.suggest_int('depth', 4, 16),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.2, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
    }

    if target == 'h1n1_vaccine': #change target
        target_data = h1_train
    elif target == 'seasonal_vaccine':
        target_data = seasonal_train
    else:
        raise ValueError(f"Unknown target column: {target}")

    categorical_columns = list(X_train.select_dtypes(include=['category']).columns)

    model = CatBoostClassifier(
        eval_metric='AUC',        
        cat_features=categorical_columns,
        train_dir='catboost_whole',  
        early_stopping_rounds=5, verbose=0, **param_dist_random         
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True)

    auc_scores = []  
    X_train.reset_index(drop=True, inplace=True)
    target_data.reset_index(drop=True, inplace=True)

    for train_index, val_index in cv.split(X_train, target_data):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = target_data.iloc[train_index], target_data.iloc[val_index]

        model.fit(X_train_fold, y_train_fold, eval_set=(X_val_fold, y_val_fold), use_best_model=True)

        auc = model.get_best_score()['validation']['AUC']
        auc_scores.append(auc)

    mean_auc = np.mean(auc_scores)

    return float(mean_auc)


if __name__ == "__main__":

    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    experiment_name = "Catboost optuna"
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    print("Experiment created")

    ################ Start Optuna Seasonal vaccine ################
    storage = "sqlite:///optuna_study.db"
    study_s = optuna.create_study(study_name="seasonal_TT", direction="maximize", storage=storage, load_if_exists=True) 
    study_s.optimize(lambda trial: objective_vaccine(trial, target='seasonal_vaccine'), n_trials=50, show_progress_bar=True, n_jobs=1)

    print("Seasonal done")
    print(study_s.best_params)
    print(study_s.best_value)

    ################ Start Optuna h1n1 vaccine ################
    study_h1 = optuna.create_study(study_name="h1n1_TT", direction="maximize", storage=storage, load_if_exists=True)
    study_h1.optimize(lambda trial: objective_vaccine(trial, target='h1n1_vaccine'), n_trials=50, show_progress_bar=True, n_jobs = 1)  

    print("H1N1 done")
    print(study_h1.best_params)
    print(study_h1.best_value)

    ################ Log to MLFlow ################

    mlflow.start_run()
    mlflow.log_dict(study_h1.best_params,"h1_best.json")
    mlflow.log_dict(study_s.best_params,"s_best.json")
        
    mlflow.log_metric("h1_best_auc", float(study_h1.best_value))
    mlflow.log_metric("s_best_auc", float(study_s.best_value))
    mlflow.log_metric("roc_auc_train", float((study_h1.best_value + study_s.best_value)/2))
    # extract predictions


    ###### Train on best parameters ######
    model0 = CatBoostClassifier(
        eval_metric='AUC',        
        cat_features=categorical_columns,
        train_dir='catboost_whole',  
        early_stopping_rounds=5, verbose=0, **study_h1.best_params         
    )
    model1 = CatBoostClassifier(
        eval_metric='AUC',        
        cat_features=categorical_columns,
        train_dir='catboost_whole',  
        early_stopping_rounds=5, verbose=0, **study_s.best_params         
    )

    model0.fit(X_train, h1_train, early_stopping_rounds=5, verbose=100)
    model1.fit(X_train, seasonal_train, early_stopping_rounds=5, verbose=100)

    # signature0 = infer_signature(X_train, model0.predict(X_train))
    # signature1 = infer_signature(X_train, model1.predict(X_train))
    mlflow.catboost.log_model(model0, "model0" )
    mlflow.catboost.log_model(model1, "model1")
    print("model saved")
    y_pred0 = model0.predict(X_test)
    y_pred1 = model1.predict(X_test)
    y_pred_prob0 = model0.predict_proba(X_test)
    y_pred_prob1 = model1.predict_proba(X_test)
    y_pred = np.column_stack([y_pred0, y_pred1])
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred, average="macro")
    mlflow.log_metric("accuracy", float(accuracy))
    mlflow.log_metric("roc_auc", float(roc_auc))

    ################ Predictions ################
    predictions0 = model0.predict_proba(output)
    predictions1 = model1.predict_proba(output)

    h1n1_probs = predictions0[:, 1]  
    seasonal_probs = predictions1[:, 1] 

    predict = pd.DataFrame({
        "respondent_id": output.index,
        "h1n1_vaccine": h1n1_probs,
        "seasonal_vaccine": seasonal_probs
    })
    predict.set_index("respondent_id", inplace=True)
    predict.to_csv("predictions.csv") 
    mlflow.log_artifact("predictions.csv")
    print("predictions saved")
    mlflow.end_run()
