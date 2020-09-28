import numpy as np
import copy
from ..config import gpu_bool
import glearn.utils.arithmetic as arith


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
    if func == 'sigmoid':
        exps = arith.exp(-val, gpu_bool)
        if derivative:
            denom = arith.add(1, arith.power(exps, 2, gpu_bool), gpu_bool)
            return arith.div(exps, denom, gpu_bool)
        else:
            return arith.div(1, arith.add(1, exps, gpu_bool), gpu_bool)

    elif func == 'relu':
        if derivative:
            val = arith.lst_arr_set(val, 0, 0)
            val = arith.grt_arr_set(val, 0, 1)
        else:
            val = arith.lst_arr_set(val, 0, 0)
        return val

    elif func == 'softmax':
        # d is meant for numerical stability, keeps vals in val close
        # to zero.
        d = -arith.max_val(val)
        exps = arith.exp(arith.add(d, val))
        exp_sum = arith.z_sum(exps)
        softmax = arith.div(exps, exp_sum)
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


