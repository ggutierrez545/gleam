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
    val[val < 1e-8] = 0
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
        # d is meant for numerical stability, keeps vals in val close
        # to zero.
        d = -np.max(val)
        exp_sum = np.sum(np.exp(val + d), axis=0)
        softmax = np.exp(val + d) / exp_sum
        if derivative:
            jacobian = softmax @ -softmax.reshape(1, -1)
            diag = softmax - softmax**2
            np.fill_diagonal(jacobian, diag)
            return jacobian
        else:
            return softmax

    else:
        raise KeyError(f"Unrecognized activation function: {func}")


def loss(prediction, truth, loss_type=''):

    true = np.argmax(truth)

    if loss_type == 'mean-squared':
        return prediction - truth

    elif loss_type == 'cross-entropy':
        return -truth / prediction


