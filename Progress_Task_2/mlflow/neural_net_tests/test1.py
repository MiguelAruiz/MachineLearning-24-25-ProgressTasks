"""First test
This test consists of a simple feedforward nn with 3 layers and a relu
activation function with a small negative slope of 0.2
"""
import keras as kr

NAME: str = "nn_1"


def gen_model(n_features: int, n_targets: int) -> kr.Model:
    """
    ## gen_model
    Model generation function.

    Args:
        n_features (int): Used to determine the shape of the keras model at the input layer
        n_targets (int): Used to determine the shape of the model at the output layer

    Returns:
        kr.Model: The keras model, in this case a neural network with 3 layers,
        with 15,10 and 7 neurons in the layers
    """
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
