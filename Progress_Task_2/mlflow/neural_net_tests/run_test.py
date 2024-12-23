import logging
from typing import Any, Callable
from scikeras.wrappers import KerasClassifier
from mlflow.models import infer_signature
import keras as kr
import mlflow
import mlflow.data.pandas_dataset
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from create_dataset import Dataset


def play_model(model, model_name, X, y, output):
    """
    ## play_model
    Run the mlflow test with the model, and data. This version is for a normal
    keras model

    Args:
        model: The keras model tested here.
        model_name (str): The name of the model
        X: Training data
        y: Training data
        output: Test data
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    with mlflow.start_run():
        model.fit(X_train, y_train)
        logging.info(f"Model trained")
        model.compute_metrics(X_test, y_test, model.predict(X_test))
        accuracy = model.get_metrics_result()["accuracy"]
        roc_auc = model.get_metrics_result()["auc"]
        logging.info("Model evaluated")

        pd_train = pd.concat([X_train, y_train], axis=1)
        pd_dataset = mlflow.data.pandas_dataset.from_pandas(
            pd_train, source="df_encoded.csv", name="whole dataset and correlation"
        )

        # mlflow.log_params(model.get_params())
        mlflow.log_input(pd_dataset, "training")

        mlflow.log_metric("roc_auc", float(roc_auc))
        mlflow.log_metric("accuracy", float(accuracy))

        mlflow.set_tag(
            "Objective", "Compare multiple models with dataset with correlation"
        )

        signature = infer_signature(X_train, model.predict(X_train))

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train,
            registered_model_name=model_name,
        )
        logging.info("model saved")
        predictions: np.ndarray = model.predict(output)
        predictions = predictions.transpose()

        h1n1_probs = predictions[0][
            :
        ]  # Probabilidades de clase positiva para h1n1_vaccine
        seasonal_probs = predictions[1][
            :
        ]  # Probabilidades de clase positiva para seasonal_vaccine

        predict = pd.DataFrame(
            {
                "respondent_id": output.index,
                "h1n1_vaccine": h1n1_probs,
                "seasonal_vaccine": seasonal_probs,
            }
        )
        predict.set_index("respondent_id", inplace=True)
        predict.to_csv(f"predictions_{model_name}.csv")
        mlflow.log_artifact(f"predictions_{model_name}.csv")
        logging.info("predictions saved")


def play_model_search(model, model_name, X, y, output, param_d):
    """
    ## play_model
    Run the mlflow test with the model, and data. This version is for random
    search of a keras model

    Args:
        model: The keras model tested here.
        model_name (str): The name of the model
        X: Training data
        y: Training data
        output: Test data
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    with mlflow.start_run():
        param_d |= {
            "input_shape": [X.shape[1]],
            "output_shape": [y.shape[1]],
        }

        model = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_d,
            n_iter=5,
            cv=5,
            n_jobs=9,
            verbose=0,
            scoring="roc_auc",
        )

        model.fit(X_train, y_train)
        logging.info(f"Model trained")
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred, average="macro")
        logging.info("Model evaluated")

        pd_train = pd.concat([X_train, y_train], axis=1)
        pd_dataset = mlflow.data.pandas_dataset.from_pandas(
            pd_train, source="df_encoded.csv", name="whole dataset and correlation"
        )

        mlflow.log_params(model.best_estimator_.get_params())
        mlflow.log_input(pd_dataset, "training")

        mlflow.log_metric("roc_auc", float(roc_auc))
        mlflow.log_metric("accuracy", float(accuracy))

        mlflow.set_tag(
            "Objective", "Compare multiple models with dataset with correlation"
        )

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

        predictions = predictions.transpose()
        h1n1_probs = predictions[0][:]
        # Probabilidades de clase positiva para h1n1_vaccine
        seasonal_probs = predictions[1][:]
        # Probabilidades de clase positiva para seasonal_vaccine

        predict = pd.DataFrame(
            {
                "respondent_id": output.index,
                "h1n1_vaccine": h1n1_probs,
                "seasonal_vaccine": seasonal_probs,
            }
        )
        predict.set_index("respondent_id", inplace=True)
        predict.to_csv(f"predictions_{model_name}.csv")
        mlflow.log_artifact(f"predictions_{model_name}.csv")
        logging.info("predictions saved")


def run_test(
    d: Dataset,
    NAME: str,
    gen_model: Callable[[int, int], kr.Model | KerasClassifier],
    PARAMS: dict[str, Any] | None = None,
):
    """
    ## run_test
    Used for running different types of tests for mlflow. It is intended for use
    with keras models. It should be run automatically by __init__.py for all
    test.py files created.

    Args:
        d (Dataset): Dataset class. Contains data to be used for the test
        NAME (str): model name
        gen_model (Callable[[int,int],kr.Model | KerasClassifier]): Model
        generation model. Uses the dataset shape in order to automatically
        generate the defined model
        PARAMS (dict[str,Any] | None, optional): Only used when evaluating
        RandomSearch, GridSearch, or similar. Defines the range of
        hyperparameters used for optimizing. Defaults to None.
    """
    X, y = d.with_correlation()
    m: kr.Model = gen_model(len(X.columns), len(y.columns))
    if not NAME.endswith("si_opt"):
        play_model(m, NAME, X, y, d.test)
    else:
        play_model_search(m, NAME, X, y, d.test, PARAMS)
