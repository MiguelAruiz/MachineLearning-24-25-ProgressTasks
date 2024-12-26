import keras as kr
from scikeras.wrappers import KerasClassifier
from scipy.stats import uniform,randint

NAME = "nn_2_si_opt"

PARAMS =  {
    # learning algorithm parameters
    # activation
    # "act": [relu_slope, "sigmoid"],
    # numbers of layers
    # # dropout and regularisation
    # "dropout": [0, 0.1, 0.2, 0.3],
    # "l1": [0, 0.01, 0.003, 0.001, 0.0001],
    # "l2": [0, 0.01, 0.003, 0.001, 0.0001],
    # "rs_num": [0.2,0.3,0.4],
    # neurons in each layer
    "shape" : [
        (50,),
        (50, 25),
        (100, 50, 20),
        (100, 50, 25),
        (100, 50, 25, 10),
        (200, 100, 50, 25),
        (200, 100, 50, 25, 10)
    ],
    "act": ["relu","tanh"],
    "learn_rate": uniform(0.0001,0.01),
    # "alpha": uniform(0.0001,0.01),
    # "max_iter": randint(200,1000),
}
FIT_PARAMS= {
    "epochs": 100,
    "verbose": 0,
}


def create_model(
    shape,
    act,
    # rs_num,
    learn_rate,
    # alpha,
    # max_iter,
    input_shape,
    output_shape,
):
    """This is a model generating function so that we can search over neural net
    parameters and architecture"""
    # act = lambda x: kr.activations.relu(x, negative_slope=rs_num)
    model = kr.models.Sequential()

    # for the firt layer we need to specify the input dimensions
    first = True

    for i in shape:
        if first:
            model.add(kr.layers.Dense(i,input_dim=input_shape,activation=act))
        else:
            model.add(kr.layers.Dense(i,activation=act))

    model.add(kr.layers.Dense(output_shape, activation="softmax"))
    model.compile(
        loss=kr.losses.MeanSquaredError(),
        metrics=["accuracy", "auc"],
        optimizer=kr.optimizers.Adam(learning_rate=learn_rate),
    )
    return model


def gen_model(n_features: int, n_targets: int) -> KerasClassifier:
    """
    ## gen_model
    Model generation function. It uses the create_model function, alongside a
    KerasClassifier, in order to be able to create lots of instances of the
    neural network when performing the random search.

    Args:
        n_features (int): Used to determine the shape of the keras model at the input layer
        n_targets (int): Used to determine the shape of the model at the output layer

    Returns:
        kr.Model: The keras model, in this case KerasClassifier, from sklearn, to be used in the future for random search
    """
    return KerasClassifier(
        build_fn=create_model,
        # act=relu_slope,
        # rs_num=0.2,
        # dropout=0,
        shape = (100,),
        act="relu",
        # rs_num = 0.2,
        learn_rate=0.001,
        # alpha=0.001,
        # max_iter=400,
        input_shape=1000,
        output_shape=20,
    )
