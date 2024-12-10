"""First test
This test consists of a simple feedforward nn with 3 layers and a relu
activation function with a small negative slope of 0.2
"""

import logging
from typing import Callable
import keras as kr
from sklearn.model_selection import train_test_split
from create_dataset import (
    Dataset,
)  # FIXME this is just temporarily borrowing the dataset code from elena, this should be cleaner

NAME: str = "nn_1"


def gen_model(n_features: int, n_targets: int) -> kr.Model:
    relu_slope = lambda x: kr.activations.relu(x, negative_slope=0.2)
    nn_input = kr.Input(shape=(n_features,))
    x = kr.layers.Dense(15, activation=relu_slope)(nn_input)
    x = kr.layers.Dense(10, activation=relu_slope)(x)
    x = kr.layers.Dense(7, activation=relu_slope)(x)
    nn_output = kr.layers.Dense(n_targets, activation="softmax")(x)
    nn_model = kr.Model(inputs=nn_input, outputs=nn_output)
    nn_model.compile(
        optimizer=kr.optimizers.RMSprop(),  # Optimizer # type: ignore
        # Loss function to minimize
        loss=kr.losses.MeanSquaredError(),
        # List of metrics to monitor
        metrics=["accuracy", "auc"],
    )
    return nn_model


# def test(X,y,model):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#     n_feat=len(X.columns)
#     n_targ=len(y.columns)
#     # model=gen_model(n_feat,n_targ)
#     model.fit(,,epochs=10)
#     logging.info("Model Trained") # TODO include model name
