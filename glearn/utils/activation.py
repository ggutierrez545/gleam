import numpy as np
import copy


def activation(values, func='', derivative=False):
    """Primary function to handle neuron activations.

    Parameters
    ----------
    values : :obj:`float`, :obj:`int`, :obj:`ndarray`
        Inputted value.
    func : str
        Keyword conveying type of activation function to use.
    derivative : bool
        Whether or not to use derivative form of activation function.

    Returns
    -------
    float
        If `val` is `float` or `int`.
    :obj:`ndarray:
        If `val` is :obj:`ndarray`.

    Raises
    ------
    KeyError
        If `func` input is unsupported.

    """
    val = copy.deepcopy(values)
    val[val < 1e-16] = 0
    if func == 'sigmoid':
        if derivative:
            return (np.exp(-val)) / ((1 + np.exp(-val)) ** 2)
        else:
            return 1 / (1 + np.exp(-val))

    elif func == 'relu':
        if derivative:
            val[val <= 0] = 0
            val[val > 0] = 1
        else:
            val[val < 0] = 0
        return val

    elif func == 'softmax':
        sm = np.exp(val) / np.sum(np.exp(val), axis=0)
        if derivative:
            sm[np.argmax(val)] -= 1
            return sm
        else:
            return sm

    else:
        raise KeyError(f"Unrecognized activation function: {func}")


def loss(prediction, truth, loss_type=''):

    if loss_type == 'mean-squared':
        return prediction - truth
    elif loss_type == 'cross-entropy':
        return -np.log(np.max(prediction))
    elif loss_type == 'avg-cross-entropy':
        return -(1 / len(prediction)) * np.sum([i / np.log(prediction) for i in prediction])


