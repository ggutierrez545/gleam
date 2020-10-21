import numpy as np
from numba import jit
from .decorators import gpu_conditional

kwargs = {'target': 'gpu', 'nopython': True, 'parallel': True}


@gpu_conditional
@jit(**kwargs)
def dot(*args):
    return np.matmul(*args[:2])


@gpu_conditional
@jit(**kwargs)
def div(*args):
    return np.divide(*args[:2])


@gpu_conditional
@jit(**kwargs)
def add(*args):
    return np.add(*args[:2])


@gpu_conditional
@jit(**kwargs)
def sub(*args):
    return np.subtract(*args[:2])


@gpu_conditional
@jit(**kwargs)
def power(*args):
    return np.power(*args[:2])


@gpu_conditional
@jit(**kwargs)
def root(*args):
    return np.power(args[0], 1/args[1])


@gpu_conditional
@jit(**kwargs)
def exp(*args):
    return np.exp(args[0])


@gpu_conditional
@jit(**kwargs)
def grt_arr_set(array, cond, update):
    array[array > cond] = update
    return array


@gpu_conditional
@jit(**kwargs)
def grteq_arr_set(array, cond, update):
    array[array >= cond] = update
    return array


@gpu_conditional
@jit(**kwargs)
def lst_arr_set(array, cond, update):
    array[array < cond] = update
    return array


@gpu_conditional
@jit(**kwargs)
def lsteq_arr_set(array, cond, update):
    array[array <= cond] = update
    return array


@gpu_conditional
@jit(**kwargs)
def max_val(array):
    return np.max(array)


@gpu_conditional
@jit(**kwargs)
def z_sum(array):
    return np.sum(array, axis=0)


@gpu_conditional
@jit(**kwargs)
def reshape(array, shape):
    return array.reshape(shape)


@gpu_conditional
@jit(**kwargs)
def fill_diagonal(array, diag):
    return np.fill_diagonal(array, diag)


@gpu_conditional
@jit(**kwargs)
def argmax(array):
    return np.argmax(array)