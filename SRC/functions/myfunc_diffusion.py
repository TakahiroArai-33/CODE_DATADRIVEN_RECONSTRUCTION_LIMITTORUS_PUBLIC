import numpy as np




def myfunc_diffusion(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    x: 1D array with periodic boundary condition x[0] = x[-1].
    """
    N = x.size
    diffusion = np.empty(N, dtype=x.dtype)
    
    # Vectorized diffusion term at interior points.
    diffusion[1:-1] = (x[2:] - 2. * x[1:-1] + x[:-2]) / (h ** 2)
    # Periodic boundary condition.
    diffusion[0] = (x[1] - 2. * x[0] + x[-2]) / (h ** 2)
    diffusion[-1] = diffusion[0]
    return diffusion
