import mlflow
import mlflow.data
import mlflow.data.pandas_dataset
from mlflow.models import infer_signature
from mlflow.data.pandas_dataset import PandasDataset
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV

import pandas as pd

from sklearn.model_selection import train_test_split
from create_dataset import Dataset
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import logging
logging.basicConfig(level=logging.INFO)

def hyperparameters(model_to_train):
    param_dist_random = {
                'estimator__n_estimators': randint(50, 200),
                'estimator__max_depth': [None, 10, 20, 30],
                'estimator__min_samples_split': randint(2, 11),
                'estimator__min_samples_leaf': randint(1, 5),
                'estimator__criterion': ['gini', 'entropy']
    }
    model = RandomizedSearchCV(estimator=model_to_train, param_distributions=param_dist_random,
                                    n_iter=100, cv=5, n_jobs=-1, verbose=2,
                                    scoring='roc_auc')
    logging.info("Optimizing hyperparameters")
    return model


def play_model(model, model_name, X, y, output):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    with mlflow.start_run():
        model = hyperparameters(model)


        model.fit(X_train, y_train)
        logging.info(f"Model trained")
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")
        logging.info("Model evaluated")
        
        pd_train = pd.concat([X_train, y_train], axis=1)
        pd_dataset = mlflow.data.pandas_dataset.from_pandas(pd_train, 
                                                            source = "df_encoded.csv", name="whole dataset and correlation")
        
        mlflow.log_params(model.best_estimator_.get_params())
        mlflow.log_input(pd_dataset, "training")

        mlflow.log_metric("roc_auc", float(roc_auc))
        mlflow.log_metric("accuracy", float(accuracy))
        mlflow.log_metric("f1", float(f1))

        mlflow.set_tag("Objective", "Compare multiple models with dataset with correlation")

        signature = infer_signature(X_train, model.predict(X_train))

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train,
            registered_model_name=model_name,
        )
        logging.info("model saved")
        predictions = model.predict_proba(output)
        
        h1n1_probs = predictions[0][:, 1]  # Probabilidades de clase positiva para h1n1_vaccine
        seasonal_probs = predictions[1][:, 1]  # Probabilidades de clase positiva para seasonal_vaccine

        predict = pd.DataFrame({
            "respondent_id": output.index,
            "h1n1_vaccine": h1n1_probs,
            "seasonal_vaccine": seasonal_probs
        })
        predict.set_index("respondent_id", inplace=True)
        predict.to_csv("predictions.csv") 
        mlflow.log_artifact("predictions.csv")
        logging.info("predictions saved")


def main():
    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    experiment_name = "RandomForest + whole dataset and correlation"
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(experiment_name)
    # Create a new MLflow Experiment
    mlflow.set_experiment(experiment_name)
    logging.info("fetching data")
    data = Dataset()
    X, y, output = data.with_correlation()
    logging.info("Data fetched")
    # Split the data
    # models = {'RandomForest_no_opt': MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42), n_jobs=-1), 
    #           'RandomForest_si_opt': MultiOutputClassifier(RandomForestClassifier(random_state=42), n_jobs=-1)}
    models = {'RandomForest': MultiOutputClassifier(RandomForestClassifier(warm_start = False), n_jobs=-1)}

    for model_name, model in models.items():
        logging.info(f"Starting run with {model_name}")
        play_model(model, model_name, X, y, output)

if __name__ == "__main__":
    main()
