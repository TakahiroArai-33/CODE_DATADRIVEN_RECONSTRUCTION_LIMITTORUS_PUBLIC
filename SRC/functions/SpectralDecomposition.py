import numpy as np
# from scipy import integrate
import sys
from typing import Literal


def calculate_Hk(X, rx, j=1):
    r"""
    According to [Eq.(86) in Kawamura, Physica D], spectral decomposition, Hk, is calculated.
    input: 
        X: 2d-ndarray [time, spatial grid]
        rx: locations of spatial grids [1d-ndarray]
        k: wave number
    in out study, the equation is 
        Hj := \int_{-L/2}^{L/2} dx X(rx, t) exp[-i j (2pi/L)x]
           = L*\int_0^1 ds X(s, t) exp[-i j 2pi s]
    """
    L = np.ptp(rx)
    f = X * np.exp(-1j*j*(2*np.pi/L)*rx)
    return np.trapz(f, rx, axis=1)




def calculate_A(X, rx):
    """
    input:
        X: 2d-ndarray [time, spatial grid]
            
        rx: spatial grids [1d-ndarray]
    key = "c_eq_0"
        Eq.(92) in Kawamura, Physica D. The method of numerical calculation is the same as Hk_{-1}.
        In (89), we assume that H_{-1} with Phi=0 is realnumber, while A=(H_{-1}*exp(i pi Phi)) in (92) is complex number. 
    
    """
    return calculate_Hk(X, rx, j=-1)


    
        
