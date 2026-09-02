#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import cupy as cp

def myfunc_interp_cupy(x_new: float, xp:np.ndarray, Yp:cp.ndarray) -> cp.ndarray:
    """Linearly interpolate 2D data ``Yp`` along its ``xp`` axis.

    ``xp`` has shape ``(n,)`` and ``Yp`` has shape ``(n, m)``.
    """
    L = Yp.shape[1]
    Y_interp = np.empty(L)
    
    # Locate the interpolation interval.
    idx = np.searchsorted(xp, x_new)

    # Clamp values outside the sampled range.
    if idx == 0:
        return Yp[0]
    elif idx == len(xp):
        return Yp[-1]
    
    # Return an exact sample without interpolation.
    if xp[idx] == x_new:
        return Yp[idx]
    
    # Linear interpolation.
    x0, x1 = xp[idx - 1], xp[idx]
    y0, y1 = Yp[idx - 1], Yp[idx]
    t = (x_new - x0) / (x1 - x0)
    Y_interp = y0 * (1 - t) + y1 * t
    return Y_interp
