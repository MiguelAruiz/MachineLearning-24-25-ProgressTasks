import keras as kr
from scikeras.wrappers import KerasClassifier

NAME = "nn_2_si_opt"

PARAMS = {
    # learning algorithm parameters
    # activation
    # "act": [relu_slope, "sigmoid"],
    # numbers of layers
    "rs_num": [0.2,0.3,0.4],
    "nl1": [1, 2, 3],
    "nl2": [1, 2, 3],
    "nl3": [1, 2, 3],
    # neurons in each layer
    "nn1": [14,15,16,17],
    "nn2": [10,11,12],
    "nn3": [6,7,8],
    # dropout and regularisation
    "dropout": [0, 0.1, 0.2, 0.3],
    "l1": [0, 0.01, 0.003, 0.001, 0.0001],
    "l2": [0, 0.01, 0.003, 0.001, 0.0001],
    # # learning algorithm parameters
    # # activation
    # "act": [relu_slope, "sigmoid"],
    # # numbers of layers
    # "nl1": [1],
    # "nl2": [1],
    # "nl3": [1],
    # # neurons in each layer
    # "nn1": [15],
    # "nn2": [10],
    # "nn3": [7],
    # # dropout and regularisation
    # "dropout": [0, 0.1, 0.2, 0.3],
    # "l1": [0, 0.01, 0.003, 0.001, 0.0001],
    # "l2": [0, 0.01, 0.003, 0.001, 0.0001],
}


def create_model(
    nl1,
    nl2,
    nl3,
    nn1,
    nn2,
    nn3,
    l1,
    l2,
    # act,
    rs_num,
    dropout,
    input_shape,
    output_shape,
    meta,
):
    """This is a model generating function so that we can search over neural net
    parameters and architecture"""
    act = lambda x: kr.activations.relu(x, negative_slope=rs_num)
    
    reg = kr.regularizers.l1_l2(l1=l1, l2=l2)

    model = kr.models.Sequential()

    # for the firt layer we need to specify the input dimensions
    first = True

    for i in range(nl1):
        if first:
            model.add(kr.layers.Dense(nn1, input_dim=input_shape, activation=act))
            first = False
        else:
            model.add(kr.layers.Dense(nn1, activation=act, kernel_regularizer=reg))
        if dropout != 0:
            model.add(kr.layers.Dropout(dropout))

    # If nl1==0, this if first will be triggered, which is why it is there.
    for i in range(nl2):
        if first:
            model.add(kr.layers.Dense(nn2, input_dim=input_shape, activation=act))
            first = False
        else:
            model.add(kr.layers.Dense(nn2, activation=act))
        if dropout != 0:
            model.add(kr.layers.Dropout(dropout))

    for i in range(nl3):
        if first:
            model.add(kr.layers.Dense(nn3, input_dim=input_shape, activation=act))
            first = False
        else:
            model.add(kr.layers.Dense(nn3, activation=act))
        if dropout != 0:
            model.add(kr.layers.Dropout(dropout))

    model.add(kr.layers.Dense(output_shape, activation="softmax"))
    model.compile(
        loss=kr.losses.MeanSquaredError(),
        metrics=["accuracy", "auc"],
    )
    return model


def gen_model(n_features: int, n_targets: int) -> KerasClassifier:
    return KerasClassifier(
        build_fn=create_model,
        nl1=1,
        nl2=1,
        nl3=1,
        nn1=17,
        nn2=12,
        nn3=8,
        l1=0.01,
        l2=0.01,
        # act=relu_slope,
        rs_num = 0.2,
        dropout=0,
        input_shape=1000,
        output_shape=20,
    )
