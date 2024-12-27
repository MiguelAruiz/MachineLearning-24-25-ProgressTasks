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

def extract_keras_params(model:kr.Model, compile_params=None, fit_params=None):
    """
    ## extract_keras_params
    Extracts relevant parameters from a keras model, optimizer and compile
    parameters, and fit parameters.

    Args:
        model: The keras model
        compile_params: The compile parameters (although they are in another class, it would be good to save them)
        fit_params: The fit parameters

    Returns:
        dict: The extracted parameters
    """
    params = {}

    params["num_layers"] = len(model.layers)
    for i, layer in enumerate(model.layers):
        layer_config = layer.get_config()
        params[f"layer_{i}_type"] = layer.__class__.__name__
        params[f"layer_{i}_units"] = layer_config.get("units", None)
        params[f"layer_{i}_activation"] = layer_config.get("activation", None)
        params[f"layer_{i}_dropout_rate"] = layer_config.get("rate", None)

    if compile_params is None:
        optimizer_config = model.optimizer.get_config()
        params.update({
            "optimizer": optimizer_config["name"],
            "learning_rate": optimizer_config.get("learning_rate"),
            "loss_function": model.loss,
        })
    else:
        params.update(compile_params)

    # Fit params
    if fit_params is not None:
        params.update(fit_params)

    return params


def play_model(model:kr.Model, model_name,fit_params, X, y, output):
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
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    early_stopping = kr.callbacks.EarlyStopping( # to improve the performance of the model
    monitor="val_auc", 
    patience=10, 
    restore_best_weights=True
    )
    with mlflow.start_run():
        model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                  callbacks=[early_stopping], **fit_params)
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
        

        params = extract_keras_params(model, fit_params=fit_params)
        mlflow.log_params(params)

        mlflow.log_metric("roc_auc", float(roc_auc))
        mlflow.log_metric("accuracy", float(accuracy))

        mlflow.set_tag(
            "Objective", "Compare multiple models with dataset with correlation"
        )

        signature = infer_signature(X_train, model.predict(X_train))

        model_info = mlflow.keras.log_model(
            model,
            artifact_path="model",
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


def run_test(
    d: Dataset,
    NAME: str,
    gen_model: Callable[[int, int], kr.Model | KerasClassifier],
    FIT_PARAMS: dict[str,Any],
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
    play_model(m, NAME,FIT_PARAMS, X, y, d.test)
