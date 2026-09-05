#!/usr/bin/env python
# coding: utf-8


"""
The class for simulation of GrayScott Model.
The simulation is use JIT for fast calculation.
"""


import numpy as np
import sys
import numba
from numba import jit


#* myfunc_gradient 
sys.path.append("functions")
from myfunc_gradient import myfunc_gradient
from myfunc_diffusion import myfunc_diffusion



def test_c(class_instance):
    if hasattr(class_instance, 'c'):
        pass 
    else: 
        print()
        print("Notice!!, instance does not have member `c`")
        print("Terminate Program")
        sys.exit()



@jit(nopython=True)
def F_jit(u, v, f, k):
    """
    JIT optimized function for F(u, v)
    input: 
        u, v: 1d-ndarray
        f, k: float or np.float
    """
    Fu = u * u * v - (f + k) * u
    Fv = -u * u * v + f * (1.0 - v)
    return Fu, Fv



@jit(nopython=True)
def diffusion_jit(u, v, D, h):
    """
    JIT optimized diffusion calculation
    input: 
        u, v: 1d-ndarray
        D: 2d-ndarray
        h: float or np.float
    """
    Du, Dv = D[0,0], D[1,1]
    D_ux = Du * myfunc_diffusion(u, h)
    D_vx = Dv * myfunc_diffusion(v, h)
    return D_ux, D_vx


# @jit(nopython=True)
# def J_jit(u, v, f, k):
#     """
#     Jacobi matrix
#         input:
#             u, v: 1d-ndarray
#             f, k: flaot
#         output:
#             return: 3-dimensional tensor
#             Ex. mat[:,:,rx]: jacobi matrix at gridpoint [ry-th, rx-th]
#     """
#     N = u.size
#     mat = np.empty((2,2,N))
#     mat[0,0,:] = 2.0 * u * v - (f + k)
#     mat[0,1,:] = u * u
#     mat[1,0,:] = -2.0 * u * v
#     mat[1,1,:] = -u * u - f
#     return mat




@jit(nopython=True)
def dxdt_jit(u, v, f, k, D, h):
    """
    JIT optimized time evolution function
    input: 
        u, v: 1d-ndarray
        D: 2d-ndarray
        f, k, h: float or np.float
    """
    Fu, Fv = F_jit(u, v, f, k)
    D_u, D_v = diffusion_jit(u, v, D, h)
    du = Fu + D_u
    dv = Fv + D_v
    return du, dv



@jit(nopython=True)
def pde_jit(U_save, V_save, t_save, tspan, step, f, k, D, h):
    """
    JIT optimized PDE solver (Heun method) with step-wise saving

    Parameters:
    U_save, V_save : 2d-ndarray
        Time evolution storage (recorded every `step` steps)
    t_save : 1d-ndarray
        Time values corresponding to recorded steps
    tspan : 1d-ndarray
    step : int
        Save results every `step` iterations
    f, k : float or np.float
    D : 2d-ndarray
    h : float or np.float

    Returns:
    U_save, V_save : 2d-ndarray
    t_save : 1d-ndarray
    """

    N = U_save.shape[1]  # Grid size
    t_size = len(tspan)

    # Working arrays.
    U = np.copy(U_save[0])
    V = np.copy(V_save[0])

    save_index = 1  # Next index in U_save, V_save, and t_save.

    for i in range(1, t_size):
        dt = tspan[i] - tspan[i - 1]

        # First step (Heun method)
        dF1_u, dF1_v = dxdt_jit(U, V, f, k, D, h)
        x2_u = U + dt * dF1_u
        x2_v = V + dt * dF1_v

        # Second step
        dF2_u, dF2_v = dxdt_jit(x2_u, x2_v, f, k, D, h)

        # Final update (Heun method)
        U += dt * 0.5 * (dF1_u + dF2_u)
        V += dt * 0.5 * (dF1_v + dF2_v)

        # Record every ``step`` iterations.
        if i % step == 0:
            U_save[save_index] = U
            V_save[save_index] = V
            t_save[save_index] = tspan[i]
            save_index += 1

    return U_save, V_save, t_save



@jit(nopython=True)
def dxdt_jit_with_chi(u, v, f, k, D, h, c):
    """
    JIT optimized time evolution function
    input: 
        u, v: 1d-ndarray
        D: 2d-ndarray
        f, k, h, c: float or np.float
    """
    Fu, Fv = F_jit(u, v, f, k)
    D_u, D_v = diffusion_jit(u, v, D, h)
    grad_u = c * myfunc_gradient(u, h)
    grad_v = c * myfunc_gradient(v, h)
    du = Fu + D_u + grad_u
    dv = Fv + D_v + grad_v
    return du, dv



@jit(nopython=True)
def pde_jit_with_chi(U_save, V_save, t_save, tspan, step, f, k, D, h, c):
    """
    JIT optimized PDE solver (Heun method) with step-wise saving

    Parameters:
    U_save, V_save : 2d-ndarray
        Time evolution storage (recorded every `step` steps)
    t_save : 1d-ndarray
        Time values corresponding to recorded steps
    tspan : 1d-ndarray
    step : int
        Save results every `step` iterations
    f, k : float or np.float
    D : 2d-ndarray
    h : float or np.float
    c: float or np.float

    Returns:
    U_save, V_save : 2d-ndarray
    t_save : 1d-ndarray
    """

    N = U_save.shape[1]  # Grid size
    t_size = len(tspan)

    # Working arrays.
    U = np.copy(U_save[0])
    V = np.copy(V_save[0])

    save_index = 1  # Next index in U_save, V_save, and t_save.

    for i in range(1, t_size):
        dt = tspan[i] - tspan[i - 1]

        # First step (Heun method)
        dF1_u, dF1_v = dxdt_jit_with_chi(U, V, f, k, D, h, c)
        x2_u = U + dt * dF1_u
        x2_v = V + dt * dF1_v

        # Second step
        dF2_u, dF2_v = dxdt_jit_with_chi(x2_u, x2_v, f, k, D, h, c)

        # Final update (Heun method)
        U += dt * 0.5 * (dF1_u + dF2_u)
        V += dt * 0.5 * (dF1_v + dF2_v)

        # Record every ``step`` iterations.
        if i % step == 0:
            U_save[save_index] = U
            V_save[save_index] = V
            t_save[save_index] = tspan[i]
            save_index += 1

    return U_save, V_save, t_save










class GrayScottModel_JIT():
    def __init__(self, grayscottmodel_class):

        self.L = grayscottmodel_class.L
        self.gridnum = grayscottmodel_class.gridnum
        self.d = grayscottmodel_class.d  # Also included in D.
        self.f = grayscottmodel_class.f
        self.k = grayscottmodel_class.k
        self.h = grayscottmodel_class.h
        self.rx = grayscottmodel_class.rx
        self.D = grayscottmodel_class.D

        if hasattr(grayscottmodel_class, 'c'):
            self.c = grayscottmodel_class.c
        else: 
            pass


    def pde(self, init_x, tspan, step=int(1)):
        """ 
        Solve PDE with step-wise saving (every `step` time steps)

        Parameters:
        init_x : list of 1d-ndarray
            [u, v] 
        tspan : 1d-ndarray
        step : int, optional
            Save results every `step` iterations (default=10)

        Returns:
        U_save, V_save : 2d-ndarray
            Time evolution recorded every `step` steps
        t_save : 1d-ndarray
            Time values corresponding to recorded steps
        """

        N = self.gridnum
        t_size = len(tspan)

        if step == 1:
            save_steps = t_size
        elif step >=2:
            save_steps = (t_size - 1) // step + 1  # Number of saved states.
        else:
            print("Error: step must be positive integer")
            sys.exit()

        # Arrays holding results sampled every ``step`` iterations.
        U_save = np.zeros([save_steps, N])
        V_save = np.zeros([save_steps, N])
        t_save = np.zeros(save_steps)

        # Store the initial condition.
        U_save[0] = init_x[0]
        V_save[0] = init_x[1]
        t_save[0] = tspan[0]

        return pde_jit(U_save, V_save, t_save, tspan, step, self.f, self.k, self.D, self.h)




    def pde_with_chi(self, init_x, tspan, step=int(1)):
        """ 
        Solve PDE with step-wise saving (every `step` time steps)

        Parameters:
        init_x : list of 1d-ndarray
            [u, v] 
        tspan : 1d-ndarray
        step : int, optional
            Save results every `step` iterations (default=10)

        Returns:
        U_save, V_save : 2d-ndarray
            Time evolution recorded every `step` steps
        t_save : 1d-ndarray
            Time values corresponding to recorded steps
        """

        # Require the translation speed c.
        test_c(self)

        N = self.gridnum
        t_size = len(tspan)

        if step == 1:
            save_steps = t_size
        elif step >=2:
            save_steps = (t_size - 1) // step + 1  # Number of saved states.
        else:
            print("Error: step must be positive integer")
            sys.exit()

        # Arrays holding results sampled every ``step`` iterations.
        U_save = np.zeros([save_steps, N])
        V_save = np.zeros([save_steps, N])
        t_save = np.zeros(save_steps)

        # Store the initial condition.
        U_save[0] = init_x[0]
        V_save[0] = init_x[1]
        t_save[0] = tspan[0]

        return pde_jit_with_chi(U_save, V_save, t_save, tspan, step, self.f, self.k, self.D, self.h, self.c)



