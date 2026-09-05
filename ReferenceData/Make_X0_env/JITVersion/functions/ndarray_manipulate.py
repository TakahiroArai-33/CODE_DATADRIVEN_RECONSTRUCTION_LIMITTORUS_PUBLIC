#!/usr/bin/env python3
# coding: utf-8

from scipy.interpolate import interp1d
import numpy as np


def myfunc_roll(X, rx, Phi, L):
    """ 
    Translate X(chi - Phi, t) by Phi to obtain X(chi).

    Parameters:
        X: 2d-ndarray
        rx: 1d-ndarray
        Phi: float
    """
    _f = interp1d(rx, X, axis=1)

    def f(_rx):
        rx_roll = np.mod(_rx-Phi, L)
        rx_roll[rx_roll>=L/2.] -= L
        return _f(rx_roll)

    return f(rx)
