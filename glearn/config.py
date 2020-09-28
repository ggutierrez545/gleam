from numba import cuda

global gpu_bool
gpu_bool = False


def _set_gpu_compiler():
    try:
        cuda.select_device(0)
        global gpu_bool
        gpu_bool = True
    except cuda.cudadrv.error.CudaSupportError:
        raise OSError("CUDA driver library cannot be found")
    except:
        raise AssertionError("Something else happened...")
