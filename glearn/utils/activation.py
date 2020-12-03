import copy
import glearn.utils.gfuncs as gf
from ..config import gpu_bool


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
        exps = gf.exp(-val, gpu_bool)
        if derivative:
            denom = gf.add(1, gf.power(exps, 2, gpu_bool), gpu_bool)
            return gf.div(exps, denom, gpu_bool)
        else:
            return gf.div(1, gf.add(1, exps, gpu_bool), gpu_bool)

    elif func == 'relu':
        if derivative:
            val = gf.lst_arr_set(val, 0, 0)
            val = gf.grt_arr_set(val, 0, 1)
        else:
            val = gf.lst_arr_set(val, 0, 0)
        return val

    elif func == 'softmax':
        # d is meant for numerical stability, keeps vals in val close
        # to zero.
        d = -gf.max_val(val)
        exps = gf.exp(gf.add(d, val))
        exp_sum = gf.z_sum(exps)
        softmax = gf.div(exps, exp_sum)
        if derivative:
            jacobian = gf.dot(softmax, -gf.reshape(softmax, (1, -1)))
            diag = gf.sub(softmax, gf.power(softmax, 2))
            gf.fill_diagonal(jacobian, diag)
            return jacobian
        else:
            return softmax

    else:
        raise KeyError(f"Unrecognized activation function: {func}")


def loss(prediction, truth, loss_type=''):

    if loss_type == 'mean-squared':
        return gf.sub(prediction, truth)

    elif loss_type == 'cross-entropy':
        return gf.div(-truth, prediction)


