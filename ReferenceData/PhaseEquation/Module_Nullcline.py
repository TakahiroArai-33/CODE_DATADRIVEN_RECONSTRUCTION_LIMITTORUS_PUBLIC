#!/usr/bin/env python3
# coding: utf-8


# %% 
import numpy as np
from scipy.interpolate import interpn, RegularGridInterpolator
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Literal
from scipy.optimize import fsolve, root, bisect
from scipy.interpolate import interp1d



def Gamma_findroot(i, x_range=[-10, 10], xdiv=2**12, 
                    key: Literal["s", "t"] = "s"):
    delta_phi = dict_coupling["delta_phi"]
    delta_theta = dict_coupling["delta_theta"]
    if key == "s":
            Gamma = dict_coupling["Gamma_s"]
    elif key == "t":
        Gamma = dict_coupling["Gamma_t"]

    # One-dimensional Gamma slice at fixed delta_theta.
    delta_theta_val = delta_theta[i]
    prefunc = interp1d(delta_phi, Gamma[i], kind="linear")
    f = lambda x: prefunc(np.mod(x, L))
    
    # Locate intervals that contain zero crossings.
    x = np.linspace(x_range[0], x_range[1], xdiv)
    y = f(x)
    x1, x2 = x[:-1], x[1:]
    # y1, y2 = np.sign(y[:-1]), np.sign(y[1:])
    y1, y2 = y[:-1], y[1:]
    arg = np.where(y1 * y2 < 0)[0]  # Adjacent values with opposite signs.
    xp1, xp2 = x1[arg], x2[arg]
    yp1, yp2 = y1[arg], y2[arg]

    # Estimate each root by linear interpolation.
    t = np.abs(yp1) / (np.abs(yp1) + np.abs(yp2))
    sol_x = xp1 + (xp2 - xp1) * t
    xp = sol_x
    yp = delta_theta_val

    # Disabled because applying modulo here introduces discontinuities.
    # Map delta_phi from [0, L) to [-L/2, L/2).
    # xp = np.where(np.mod(xp, L) > L/2,
    #                 np.mod(xp, L) - L, np.mod(xp, L))
                    
    # Map delta_theta from [0, 2*pi) to [-pi, pi).
    # yp = np.where(np.mod(yp, 2*np.pi) > np.pi, 
    #                         np.mod(yp, 2*np.pi) - 2*np.pi, np.mod(yp, 2*np.pi))
    return xp, yp


# %% 

def main_routine():
    # Load the mutual coupling functions.
    datas = np.load("./MutualCouplingFunction_8193.npz")
    Gamma_s = datas["Gamma_s"] #* mutual coupling for spatial 2d-ndarray [temporal, spatial]
    Gamma_t = datas["Gamma_t"] #* temporal coupling for temporal 2d-ndarray [temporal, spatial]
    delta_theta = datas["delta_theta"] #* 0~2π
    delta_phi = datas["delta_phi"] #* 0~L

    global dict_coupling  
    dict_coupling = {
        "Gamma_s": Gamma_s, #* [temporal, spatial]
        "Gamma_t": Gamma_t, #* [temporal, spatial]
        "delta_theta": delta_theta,
        "delta_phi": delta_phi
    }


    # T = np.ptp(delta_theta) #* 2.0*np.pi
    global L 
    L = np.ptp(delta_phi) #* L

    # Interpolants on [0, L] x [0, 2*pi], evaluated as
    # [delta_phi, delta_theta].
    pre_gs = RegularGridInterpolator((delta_phi, delta_theta), Gamma_s.T, method="linear" )
    pre_gt = RegularGridInterpolator((delta_phi, delta_theta), Gamma_t.T, method="linear" )


    # The following nullclines use delta_phi in [0, L) and
    # delta_theta in [0, 2*pi).

    # Spatial nullcline near the in-phase state.
    null_s1 = {"delta_phi": [], "delta_theta": []}
    for i in range(dict_coupling["delta_theta"].size):
        xp, yp = Gamma_findroot(i, x_range=[-20, 20], xdiv=2**14, key="s")
        null_s1["delta_phi"].append(xp)
        null_s1["delta_theta"].append(yp)

    # Spatial nullcline near the antiphase state.
    null_s2 = {"delta_phi": [], "delta_theta": []}
    for i in range(dict_coupling["delta_theta"].size):
        xp, yp = Gamma_findroot(i, x_range=[150, 100], xdiv=2**14, key="s")
        null_s2["delta_phi"].append(xp)
        null_s2["delta_theta"].append(yp)


    # Temporal nullcline near the in-phase state.
    null_t1 = {"delta_phi": [], "delta_theta": []}
    for i in range(dict_coupling["delta_theta"].size):
        xp, yp = Gamma_findroot(i, x_range=[-30, 30], xdiv=2**14, key="t")
        null_t1["delta_phi"].append(xp)
        null_t1["delta_theta"].append(yp)

    # Temporal nullcline near the antiphase state.
    null_t2 = {"delta_phi": [], "delta_theta": []}
    for i in range(dict_coupling["delta_theta"].size):
        xp, yp = Gamma_findroot(i, x_range=[150, 100], xdiv=2**14, key="t")
        null_t2["delta_phi"].append(xp)
        null_t2["delta_theta"].append(yp)        

    
    # Flatten the nullclines without branches.
    null_s1["delta_phi"] = np.array(null_s1["delta_phi"]).ravel()
    null_s1["delta_theta"] = np.array(null_s1["delta_theta"]).ravel()
    null_s2["delta_phi"] = np.array(null_s2["delta_phi"]).ravel()
    null_s2["delta_theta"] = np.array(null_s2["delta_theta"]).ravel()
    null_t2["delta_phi"] = np.array(null_t2["delta_phi"]).ravel()
    null_t2["delta_theta"] = np.array(null_t2["delta_theta"]).ravel()

    # Split null_t1, assuming three branches near delta_theta = 0.
    null_1_before_startsplit = {"delta_phi": [], "delta_theta": []}
    null_1_after_endsplit = {"delta_phi": [], "delta_theta": []}
    null_t1_1 = {"delta_phi": [], "delta_theta": []}
    null_t1_2 = {"delta_phi": [], "delta_theta": []}
    null_t1_3 = {"delta_phi": [], "delta_theta": []}


    for i in range(dict_coupling["delta_theta"].size):
        theta_val = null_t1["delta_theta"][i]
        phi_val = null_t1["delta_phi"][i]
        if phi_val.size == 1:
            if theta_val < np.pi:
                null_1_before_startsplit["delta_theta"].append(theta_val)
                null_1_before_startsplit["delta_phi"].append(phi_val[0])
            else:
                null_1_after_endsplit["delta_theta"].append(theta_val)
                null_1_after_endsplit["delta_phi"].append(phi_val[0])

        elif phi_val.size ==3:
            null_t1_1["delta_theta"].append(theta_val)
            null_t1_1["delta_phi"].append(phi_val[1])  # Center branch.
            null_t1_2["delta_theta"].append(theta_val)
            null_t1_2["delta_phi"].append(phi_val[0])  # Left branch.
            null_t1_3["delta_theta"].append(theta_val)
            null_t1_3["delta_phi"].append(phi_val[2])  # Right branch.
        else:
            print("Unexpected number of branches:", null_t1["delta_phi"][i].size)

    return null_s1, null_s2, null_t1_1, null_t1_2, null_t1_3, null_1_before_startsplit, null_1_after_endsplit, null_t2

# %%
