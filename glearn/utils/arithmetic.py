import numpy as np
from numba import jit
from .decorators import gpu_conditional


@gpu_conditional
@jit(target='gpu', nopython=True, parallel=True)
def dot(*args):
    return np.matmul(*args[:2])


@gpu_conditional
@jit(target='gpu', nopython=True, parallel=True)
def div(*args):
    return np.divide(*args[:2])


@gpu_conditional
@jit(target='gpu', nopython=True, parallel=True)
def add(*args):
    return np.add(*args[:2])


@gpu_conditional
@jit(target='gpu', nopython=True, parallel=True)
def sub(*args):
    return np.subtract(*args[:2])


@gpu_conditional
@jit(target='gpu', nopython=True, parallel=True)
def power(*args):
    return np.power(*args[:2])


@gpu_conditional
@jit(target='gpu', nopython=True, parallel=True)
def root(*args):
    return np.power(args[0], 1/args[1])


@gpu_conditional
@jit(target='gpu', nopython=True, parallel=True)
def exp(*args):
    return np.exp(args[0])


@gpu_conditional
@jit(target='gpu', nopython=True, parallel=True)
def grt_arr_set(array, cond, update):
    array[array > cond] = update
    return array


@gpu_conditional
@jit(target='gpu', nopython=True, parallel=True)
def grteq_arr_set(array, cond, update):
    array[array >= cond] = update
    return array


@gpu_conditional
@jit(target='gpu', nopython=True, parallel=True)
def lst_arr_set(array, cond, update):
    array[array < cond] = update
    return array


@gpu_conditional
@jit(target='gpu', nopython=True, parallel=True)
def lsteq_arr_set(array, cond, update):
    array[array <= cond] = update
    return array


@gpu_conditional
@jit(target='gpu', nopython=True, parallel=True)
def max_val(array):
    return np.max(array)


@gpu_conditional
@jit(target='gpu', nopython=True, parallel=True)
def z_sum(array):
    return np.sum(array, axis=0)



