import numpy as np
from numba import cuda
from .decorators import gpu_conditional


@gpu_conditional
@cuda.jit
def mult(*args):
    return args[0] * args[1]


@gpu_conditional
@cuda.jit
def dot(*args):
    return np.matmul(*args)


@gpu_conditional
@cuda.jit
def div(*args):
    return np.divide(*args)


@gpu_conditional
@cuda.jit
def add(*args):
    return np.add(*args)


@gpu_conditional
@cuda.jit
def sub(*args):
    return np.subtract(*args)


@gpu_conditional
@cuda.jit
def power(*args):
    return np.power(*args)


@gpu_conditional
@cuda.jit
def root(*args):
    return np.power(args[0], 1/args[1])


@gpu_conditional
@cuda.jit
def exp(*args):
    return np.exp(args[0])


@gpu_conditional
@cuda.jit
def grt_arr_set(array, cond, update):
    array[array > cond] = update
    return array


@gpu_conditional
@cuda.jit
def grteq_arr_set(array, cond, update):
    array[array >= cond] = update
    return array


@gpu_conditional
@cuda.jit
def lst_arr_set(array, cond, update):
    array[array < cond] = update
    return array


@gpu_conditional
@cuda.jit
def lsteq_arr_set(array, cond, update):
    array[array <= cond] = update
    return array


@gpu_conditional
@cuda.jit
def max_val(array):
    return np.max(array)


@gpu_conditional
@cuda.jit
def z_sum(array):
    return np.sum(array, axis=0)


@gpu_conditional
@cuda.jit
def reshape(array, shape):
    return array.reshape(shape)


@gpu_conditional
@cuda.jit
def fill_diagonal(array, diag):
    return np.fill_diagonal(array, diag)


@gpu_conditional
@cuda.jit
def argmax(array):
    return np.argmax(array)


@gpu_conditional
@cuda.jit
def transpose(array):
    return array.T


@gpu_conditional
@cuda.jit
def zeros(size):
    return np.zeros(size)
