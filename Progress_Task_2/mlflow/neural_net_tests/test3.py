"""Third test
Create different neural network models
"""
import keras as kr


NAME: str = "nn_3"


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
    # relu_slope = lambda x: kr.activations.relu(x, negative_slope=0.2)
    nn_input = kr.Input(shape=(n_features,))
    x = kr.layers.Dense(512, activation='relu')(nn_input) #what ever other input and activation function
    x = kr.layers.Dense(256, activation='relu', kernel_regularizer=kr.regularizers.l2(0.01))(x)
    x = kr.layers.Dropout(0.3)(x)
    x = kr.layers.Dense(128, activation='relu')(x)
    x = kr.layers.Dense(64, activation='sigmoid')(x)
    x = kr.layers.Dropout(0.2)(x)
    x = kr.layers.Dense(32, activation='relu')(x)
    nn_output = kr.layers.Dense(n_targets, activation="sigmoid")(x) # SIGMOID better than SOFTMAX

    nn_model = kr.Model(inputs=nn_input, outputs=nn_output)
    nn_model.compile(
        optimizer=kr.optimizers.Adam(learning_rate=0.001),  
        # Loss function to minimize
        loss=kr.losses.BinaryCrossentropy(), # THIS ONE DO NOT CHANGE
        # List of metrics to monitor
        metrics=["accuracy", kr.metrics.AUC(name="auc", multi_label=True)],
    )
    return nn_model
