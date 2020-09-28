from functools import wraps
from glearn.config import gpu_bool


def gpu_conditional(func):
    @wraps(func)
    def cond_wrapper(*args):
        if gpu_bool:
            return func(*args)
        else:
            return func.__wrapped__(*args)
    return cond_wrapper

