import numpy as np
import numba
from numba import jit


@jit(nopython=True)
def myfunc_gradient(x, h):
    """
    x: 1-D array with periodic boundary condition x[0] = x[-1].
    """
    grad_array = np.zeros_like(x)
    grad_array[1:-1] = (x[2:]-x[:-2]) / (2*h)
    grad_array[0] = (x[1]-x[-2]) / (2*h)
    grad_array[-1] = grad_array[0]
    return grad_array
