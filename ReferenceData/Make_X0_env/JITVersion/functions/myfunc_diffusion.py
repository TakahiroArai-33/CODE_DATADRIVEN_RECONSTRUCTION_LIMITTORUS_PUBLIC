import numpy as np
import numba
from numba import jit


@jit(nopython=True)
def myfunc_diffusion(x, h):
    """
    x: 1-D array with periodic boundary condition x[0] = x[-1].
    """
    N = x.size
    diffusion = np.empty(N)
    for i in range(1, N - 1):
        diffusion[i] = (x[i + 1] - 2. * x[i] + x[i - 1]) / (h ** 2)
    # Periodic boundary condition.
    diffusion[0] = (x[1] - 2. * x[0] + x[-2]) / (h ** 2)
    diffusion[-1] = diffusion[0]
    return diffusion
