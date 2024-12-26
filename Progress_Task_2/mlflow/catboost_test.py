from turtle import st
from optuna.integration.mlflow import MLflowCallback
from catboost import CatBoostClassifier
import mlflow
import mlflow.data
import mlflow.data.pandas_dataset
from mlflow.models import infer_signature
from mlflow.data.pandas_dataset import PandasDataset
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV
import optuna
from sklearn.model_selection import StratifiedKFold

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from create_dataset import Dataset
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier


data = Dataset()
X, y, output = data.all_onehot()
# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
h1_train = y_train["h1n1_vaccine"]
seasonal_train = y_train["seasonal_vaccine"]
h1_test = y_test["h1n1_vaccine"]    
seasonal_test = y_test["seasonal_vaccine"]

def objective_vaccine(trial, target):
    param_dist_random = {
        'iterations': trial.suggest_int('iterations', 100, 3000),
        'depth': trial.suggest_int('depth', 4, 16),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.2, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
        'loss_function': trial.suggest_categorical('loss_function', ['Logloss', 'CrossEntropy']),
    }

    # Definir el objetivo según la columna
    if target == 'h1n1_vaccine':
        target_data = h1_train
    elif target == 'seasonal_vaccine':
        target_data = seasonal_train
    else:
        raise ValueError(f"Unknown target column: {target}")
    
    model = CatBoostClassifier(
        eval_metric='AUC',        
        cat_features=[],
        train_dir='catboost_info',
        early_stopping_rounds=50, verbose=0, **param_dist_random         
    )
    # Inicializar StratifiedKFold para validación cruzada
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    auc_scores = []  # Para almacenar AUC de cada pliegue
    X_train.reset_index(drop=True, inplace=True)
    target_data.reset_index(drop=True, inplace=True)

    for train_index, val_index in cv.split(X_train, target_data):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = target_data.iloc[train_index], target_data.iloc[val_index]

        # Entrenar el modelo en cada pliegue
        model.fit(X_train_fold, y_train_fold, eval_set=(X_val_fold, y_val_fold), use_best_model=True)

        # Obtener las predicciones para el conjunto de validación
        y_pred_prob = model.predict_proba(X_val_fold)[:, 1]

        # Calcular AUC para el pliegue
        auc = roc_auc_score(y_val_fold, y_pred_prob)
        auc_scores.append(auc)

    # Promediar las AUC de todos los pliegues
    mean_auc = np.mean(auc_scores)

    return float(mean_auc)

# Si tienes columnas categóricas, define cuáles son:
# categorical_features_indices = [X.columns.get_loc(col) for col in X.select_dtypes('category').columns]
categorical_features_indices = []
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
experiment_name = "Catboost optuna"
if not mlflow.get_experiment_by_name(experiment_name):
    mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)
print("Experimento creado")

# mlflow_callback = MLflowCallback(tracking_uri=None, metric_name="auc", create_experiment=False, mlflow_kwargs={"experiment_name": experiment_name})
study_s = optuna.create_study(directions=["maximize"])
study_s.optimize(lambda trial: objective_vaccine(trial, target='seasonal_vaccine'), n_trials=100, show_progress_bar=True)

print("Seasonal done")
print(study_s.best_params)
print(study_s.best_value)

study_h1 = optuna.create_study(directions=["maximize"])
study_h1.optimize(lambda trial: objective_vaccine(trial, target='h1n1_vaccine'), n_trials=100, show_progress_bar=True)  

print("H1N1 done")
print(study_h1.best_params)
print(study_h1.best_value)

mlflow.start_run()
mlflow.log_dict(study_h1.best_params,"h1_best.json")
mlflow.log_dict(study_s.best_params,
                    "s_best.json")
    
mlflow.log_metric("h1_best_auc", float(study_h1.best_value))
mlflow.log_metric("s_best_auc", float(study_s.best_value))
mlflow.log_metric("roc_auc", float((study_h1.best_value + study_s.best_value)/2))

model0 = CatBoostClassifier(**study_h1.best_params)
model1 = CatBoostClassifier(**study_s.best_params)

model0.fit(X_train, h1_train, early_stopping_rounds=20, verbose=100)
model1.fit(X_train, seasonal_train, early_stopping_rounds=20, verbose=100)

signature0 = infer_signature(X_train, model0.predict(X_train))
signature1 = infer_signature(X_train, model1.predict(X_train))
mlflow.catboost.log_model(model0, "model0", signature=signature0)
mlflow.catboost.log_model(model1, "model1", signature=signature1)
print("model saved")
predictions0 = model0.predict_proba(output)
predictions1 = model1.predict_proba(output)

h1n1_probs = predictions0[:, 1]  # Probabilidades de clase positiva para h1n1_vaccine
seasonal_probs = predictions1[:, 1]  # Probabilidades de clase positiva para seasonal_vaccine

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


# with mlflow.start_run():

#     # model = CatBoostClassifier(
#     #     eval_metric='AUC',        
#     #     cat_features=categorical_features_indices,
#     #     task_type="GPU",  
#     #     early_stopping_rounds=20, verbose=100         
#     # )
#     # result = model.randomized_search(X_train, h1_train, param_dist_random, n_iter=100, cv=5, stratified=True, search_by_train_test_split=True)
#     # # model0.fit(
#     # #     X_train, h1_train,
#     # #     eval_set=(X_test, h1_test),  
#     # #     use_best_model=True        
#     # # )
#     print("====================\n H1N1 \n====================")
#     model0 = CatBoostClassifier(**result["params"])
#     y_pred = model0.predict(X_test)

#     roc_auc0 = roc_auc_score(h1_test, y_pred)
#     accuracy0 = accuracy_score(h1_test, y_pred)

#     model = CatBoostClassifier(
#         eval_metric='AUC',        
#         cat_features=categorical_features_indices,
#         task_type="GPU",      
#         early_stopping_rounds=20, verbose=100         

#     )
#     result = model.randomized_search(X_train, seasonal_train, param_dist_random, n_iter=100, cv=5, stratified=True, search_by_train_test_split=True)
#     print("====================\n SEASONAL \n====================")
#     model1 = CatBoostClassifier(**result["params"])
#     # model1.fit(
#     #     X_train, seasonal_train,
#     #     eval_set=(X_test, seasonal_test),  # Datos para validación
#     #     use_best_model=True         # Detener si no mejora
#     # )
#     y_pred = model1.predict(X_test)

#     roc_auc1 = roc_auc_score(seasonal_test, y_pred)
#     accuracy1 = accuracy_score(seasonal_test, y_pred)
#     print(f"roc_auc0: {roc_auc0}")
#     print(f"roc_auc1: {roc_auc1}")

#     print(f"accuracy0: {accuracy0}")
#     print(f"accuracy1: {accuracy1}")

#     print("==== WHOLE ====")
#     roc_auc = (roc_auc0 + roc_auc1)/2
#     print(f"roc_auc: {roc_auc}")

        
#     mlflow.log_dict(model1.best_estimator_.get_params(),
#                     "model1_best_params.json")
    
#     mlflow.log_metric("roc_auc", float(roc_auc))
#     mlflow.log_metric("roc_auc0", float(roc_auc0))
#     mlflow.log_metric("roc_auc1", float(roc_auc1))
#     mlflow.log_metric("accuracy0", float(accuracy0))
#     mlflow.log_metric("accuracy1", float(accuracy1))

#     mlflow.set_tag("dataset", "Onehot + ordinal with all")

#     signature0 = infer_signature(X_train, model0.predict(X_train))
#     signature1 = infer_signature(X_train, model1.predict(X_train))
#     mlflow.catboost.log_model(model0.best_estimator_, "model0", signature=signature0)
#     mlflow.catboost.log_model(model1.best_estimator_, "model1", signature=signature1)

#     print("model saved")
#     predictions0 = model0.predict_proba(output)
#     predictions1 = model1.predict_proba(output)
    
#     h1n1_probs = predictions0[:, 1]  # Probabilidades de clase positiva para h1n1_vaccine
#     seasonal_probs = predictions1[:, 1]  # Probabilidades de clase positiva para seasonal_vaccine

#     predict = pd.DataFrame({
#         "respondent_id": output.index,
#         "h1n1_vaccine": h1n1_probs,
#         "seasonal_vaccine": seasonal_probs
#     })
#     predict.set_index("respondent_id", inplace=True)
#     predict.to_csv("predictions.csv") 
#     mlflow.log_artifact("predictions.csv")
#     print("predictions saved")
