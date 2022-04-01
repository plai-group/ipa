from torch import nn


def get_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "elu":
        return nn.ELU()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "tanh":
        return nn.Tanh()
    else:
        raise ValueError(f"Unexpected activation name ({name})")


def _mlp(dim_list, activation="relu", dropout=None):
    """ Given a list of dimensions, retuns an MLP with the given dimensions
        and nonlinearities between layers, except the output layer.

    Args:
        dim_list (list): List of layer dimensions (length > 1)
        activation (str): Activation function for all the hidden layers.
                          should one of "relu", "elu", "tanh", "sigmoid"
        dropout (int): If not None, adds a dropout layer with "p=dropout"
                       after each hidden layer
    """
    assert len(dim_list) > 1
    layers = []
    ## Hidden layers ##
    for i in range(1, len(dim_list) - 1):
        layers.extend([nn.Linear(dim_list[i-1], dim_list[i]),
                       get_activation(activation)])
        if dropout is not None:
            layers.append(nn.Dropout(p=dropout))
    ## Output layer ##
    layers.append(nn.Linear(dim_list[-2], dim_list[-1]))
    return nn.Sequential(*layers)