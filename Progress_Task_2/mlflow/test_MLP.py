'''
# Test
- Name: test.py
- Description: This script is used to train a model and log the results in MLflow. It uses the Dataset class from create_dataset.py to fetch the data.
- Author: Elena Ballesteros Morallón (E-Mail: Elena.Ballesteros2@alu.uclm.es GH:@elena-17)
'''

######################################## IMPORTS ########################################
import mlflow
import mlflow.data
import mlflow.data.pandas_dataset
from mlflow.models import infer_signature
from mlflow.data.pandas_dataset import PandasDataset
from scipy.stats import randint
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import pandas as pd

from sklearn.model_selection import train_test_split
from create_dataset import Dataset
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
import logging
import configparser
import os
from scipy.stats import uniform
import matplotlib.pyplot as plt
#########################################################################################


################################## Global Configurations ################################
# Load configuration file
CONFIG_FILE_PATH = "mlflow/test.conf"

config = configparser.ConfigParser()
config.read(CONFIG_FILE_PATH)

CONFIG_SECTION_MLFLOW = "mlflow"
CONFIG_SECTION_NAMES = "names"
CONFIG_SECTION_DATA = "data"
CONFIG_PARAM_MLFLOW_ADDRESS = "mflow_address"
CONFIG_PARAM_MFLWOW_PORT = "mlflow_port"
CONFIG_PARAM_EXPERIMENT_NAME = "MLPClassifier + Whole dataset"
MLFLOW_LOCATION = config[CONFIG_SECTION_MLFLOW][CONFIG_PARAM_MLFLOW_ADDRESS] + config[CONFIG_SECTION_MLFLOW][CONFIG_PARAM_MFLWOW_PORT]
OPTIMIZED_SUFFIX = config[CONFIG_SECTION_NAMES]["model_optimized"]
ROC_AUC_NAME = config[CONFIG_SECTION_NAMES]["parameter_roc_auc"]
ACCURACY_NAME = config[CONFIG_SECTION_NAMES]["parameter_accuracy"]
OUTPUT_FILE_PATH = config[CONFIG_SECTION_DATA]["output_path"]

# Load logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S' 
)

#########################################################################################

# TODO: This method only works for RandomForestClassifier in Grid Search. Maybe it should work with any model
def hyperparameters(model_to_train):
    '''
    ## hyperparameters
    
    Initialize the hyperparameters for a Random Forest model and creates it.
    
    :param model_to_train: The model to train.
    
    :return model: The model, now built with fine-tuned hyperparameters.
    
    '''
    
    tuner_logger = logging.Logger("[Tuner]")
    
    param_dist = {
    'estimator__hidden_layer_sizes': [
        (50,), (100,), (50, 25), (100, 50), (150,100,50), (100, 50, 25)
    ],
    'estimator__activation': ['tanh', 'relu', 'logistic'],
    'estimator__learning_rate_init': [0.0001, 0.001, 0.01],
    'estimator__alpha': [0.0001, 0.001, 0.01],
}

    tuner_logger.info("Hyperparameters optimized. Building model...")
    model = RandomizedSearchCV(estimator=model_to_train, param_distributions=param_dist, n_iter=100, cv=5, n_jobs=-1, verbose=2,
                                    scoring= ROC_AUC_NAME)
    tuner_logger.info("Model built successfully!")
    
    return model


def play_model(model, model_name : str, X : pd.DataFrame, y : pd.DataFrame, output : pd.DataFrame):
    
    '''
    ## play_model
    
    performs a Machine Learning run using the model passed as parameter and some input data.
    
    :param Any model: The model to train.
    :param str model_name: The name of the model.
    :param pd.DataFrame X: The input data.
    :param pd.DataFrame y: The target data.
    :param pd.DataFrame output: The data used to test the model.
    
    :return None: This method saves the predictions for the testing in a file within the `mlflow` directory.
    '''
    
    run_logger = logging.Logger("[Run]")
        
    # Split the dataset into train and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # All this codeblock englobes the part of the run tracked by MLflow (Anything outside this block won't be tracked)
    with mlflow.start_run():
        
        ################################## Initial logs ##################################
        
        run_logger.info(f"========== Starting initial MLflow logging for model {model_name} =========")
        
        # Log the "presentation card" of the model. What is it trying to achieve?
        mlflow.set_tag("Objective", "Compare multiple models with dataset with correlation")
        
        # Log the input data of the run: features and targets used for training the model.
        pd_train = pd.concat([X_train, y_train], axis=1)
        pd_dataset = mlflow.data.pandas_dataset.from_pandas(pd_train, 
                                                            source = "df_encoded.csv", name="whole dataset and correlation")
        mlflow.log_input(pd_dataset, "training")
               
        # Tune the hyperparameters of the model (if needed. Only for optimized models) and log them
        if model_name.endswith(OPTIMIZED_SUFFIX):
            run_logger.info(f"Model {model_name} is optimized. Tuning hyperparameters...")
            model = hyperparameters(model)


        ########################## Training, testing and evaluation ######################
        
        # Train the model
        run_logger.info(f"Training model {model_name}...")
        model.fit(X_train, y_train)

        mlflow.log_params(model.best_estimator_.get_params())
        signature = infer_signature(X_train, model.predict(X_train))    # Log the signature to MLFlow      
        run_logger.info(f"Model {model_name} trained successfully!")
       
        # Predict the test data
        run_logger.info(f"Predicting test data with trained model {model_name}...")
        y_pred = model.predict(X_test)
        run_logger.info(f"Test data predictions finished!")

        # Evaluate the model (get the metrics)
        run_logger.info(f"Evaluating model {model_name}...")
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred, average="macro")
        run_logger.info(f"Model {model_name} evaluated successfully!\nAccuracy: {accuracy}\nROC AUC: {roc_auc}")


        ################################## Result logs ##################################
        
        # Log the model's metrics and information to MLflow
        mlflow.log_metric(ROC_AUC_NAME, float(roc_auc))
        mlflow.log_metric(ACCURACY_NAME, float(accuracy))
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train,
            registered_model_name=model_name,
        )
        
        run_logger.info(f"Model {model_name} logged successfully on MLflow.") # I infer the model_info was for this...
        
        # Predict probabilities for the output data
        predictions = model.predict_proba(output)
        
        h1n1_probs = predictions[0][:, 1]  # Probabilidades de clase positiva para h1n1_vaccine
        seasonal_probs = predictions[1][:, 1]  # Probabilidades de clase positiva para seasonal_vaccine

        predict = pd.DataFrame({
            "respondent_id": output.index,
            "h1n1_vaccine": h1n1_probs,
            "seasonal_vaccine": seasonal_probs
        })
        
        # The predictions are indexed by their value of respondent_id
        predict.set_index("respondent_id", inplace=True)
        
        ################################## Final logs ##################################
        
        # Store the predictions in a file and log them to MLflow
        predict.to_csv(OUTPUT_FILE_PATH) 
        mlflow.log_artifact(OUTPUT_FILE_PATH)
        run_logger.info("predictions saved")


def main():
    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri=MLFLOW_LOCATION)
    experiment_name = CONFIG_PARAM_EXPERIMENT_NAME
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(experiment_name)
    # Create a new MLflow Experiment
    mlflow.set_experiment(experiment_name)
    logging.info("fetching data")
    data = Dataset()
    X, y = data.with_correlation()
    output = data.test
    logging.info("Data fetched")
    models = {
        f"{config[CONFIG_SECTION_NAMES]['mlp_name']}_adam_constant{OPTIMIZED_SUFFIX}": MultiOutputClassifier(MLPClassifier(early_stopping=True, solver='adam', learning_rate='constant', max_iter=2000), n_jobs=-1),
        }

    for model_name, model in models.items():
        logging.info(f"Starting run with {model_name}")
        play_model(model, model_name, X, y, output)

if __name__ == "__main__":
    main()
