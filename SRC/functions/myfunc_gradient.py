import numpy as np




def myfunc_gradient(x: np.ndarray, h : np.ndarray) -> np.ndarray:
    """
    x: 1D array with periodic boundary condition x[0] = x[-1].
    """
    grad_array = np.empty(x.shape, dtype=x.dtype)

    grad_array[1:-1] = (x[2:]-x[:-2]) / (2*h)
    grad_array[0] = (x[1]-x[-2]) / (2*h)
    grad_array[-1] = grad_array[0]
    return grad_array
