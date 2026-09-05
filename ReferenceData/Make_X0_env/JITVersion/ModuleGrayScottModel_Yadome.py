#!/usr/bin/env python
# coding: utf-8


 
"""
This module contains the Gray-Scott model class and its methods.
The FitzHugh-Nagumo or Gray-Scott model is a reaction-diffusion model that describes the time evolution of two reacting and diffusing species.
The model is given by the following system of partial differential equations:

        ut = ∆u + u^2v - (f + k)u,
        vt = d*∆v - u^2v + f (1 - v).

    The parameter values are set to F = 0.018,k = 0.052, for which the kinetics part of Eq. (2) is excitable (Fig. 2). Periodic boundary conditions are used for Eq. (2), and the system size L = 250 (-250~250) is sufficiently larger than the typical size of a pulse solution. The spatial grid size is 512 and the time increment is 0.1.
                    [Yadome PRE (2011)]

    where u and v are the concentrations of the two species, D_u and D_v are the diffusion coefficients, F is the feed rate, and k is the kill rate.
    Here, we consider the model in a 1D spatioal domein, and the frequency of spatioal phase is zero.

    or 

    ut = u - u3 - v + ∆u,
    vt = ϵ(u - a1v - a0) + δ∆v,
                    [Hagberg Nonlinearity (1994), Eq (1-1)]
    According to Fig.15 and its caption
            epsilon = 0.03  delta = 2.5  a1 = 2.0  a0 = -0.01, L=600
    According to Fig.14 and its caption
            epsilon = 0.03  delta = 2.5  a1 = 2.0  a0 = -0.1, L=200
            For a unit diffusion coefficient, choose dt to satisfy the
            stability condition delta_t / delta_x^2 < 0.5.

    The effect of the front bifurcation on the behavior of a single domain structure is demonstrated in Figs. 15a and 15b corresponding to ǫ > ǫc(δ) (Ising fronts) and ǫ < ǫc(δ) (Bloch fronts), respectively. In both figures the initial conditions consist of two fronts bounding a wide up-state domain and propagating toward one another. In 15a the two fronts set in a stable oscillatory motion, whereas in 15b they rebound from one another and leave the system in a uniform up state.

    

### Memo for the program

"The following methods and members will be included in order to pass them to the `Adjoint` class in functions/SolveAdjointEquation.py."
    method: self.F, self.J, self.dxdt 
    member: self.D, self.gridnum. self.h,
This member will be added later, before passing it to the `Adjoint` class."
    member: self.c 
"""


# %%

import numpy as np
from numpy.random import random, normal, seed
from scipy.interpolate import interp1d
import copy, time, sys
import matplotlib.pyplot as plt
import os
import gc
from typing import Literal
import numba
from numba import jit



#* myfunc_gradient 
sys.path.append("functions")
from myfunc_gradient import myfunc_gradient



def mul_uv(uv, const):
    """
    uv: list[u, v]
    const: float
    """
    new_u = const*uv[0]
    new_v = const*uv[1]
    return [new_u, new_v]


def add_uv(uv1, uv2, const1=1.0, const2=1.0):
    """ 
    uv1, uv2: list[u, v]
    const: float    
    """
    new_u = const1*uv1[0] + const2*uv2[0]
    new_v = const1*uv1[1] + const2*uv2[1]
    return [new_u, new_v]


# @jit(nopython=True)
# def myfunc_gradient(x, h):
#     """
#     x: 1-D array with periodic boundary condition x[0] = x[-1].
#     """
#     grad_array = np.empty_like(x)
#     grad_array[1:-1] = (x[2:]-x[:-2]) / (2*h)
#     grad_array[0] = (x[1]-x[-2]) / (2*h)
#     grad_array[-1] = grad_array[0]
#     return grad_array



def test_c(class_instance):
    if hasattr(class_instance, 'c'):
        pass 
    else: 
        print()
        print("Notice!!, instance does not have member `c`")
        print("Terminate Program")
        sys.exit()




class GrayScottModel():
    """ 
    Equation (2) in Yadome PRE 2011
        ut = ∆u + u^2v - (F + k)u,
        vt = d*∆vxx - u^2v + F (1 - v).

    The parameter values are set to F = 0.018,k = 0.052, for which the kinetics part of Eq. (2) is excitable (Fig. 2). Periodic boundary conditions are used for Eq. (2), and the system size L = 250 is sufficiently larger than the typical size of a pulse solution. The spatial grid size is 512 and the time increment is 0.1.
                    [Yadome PRE (2011)]

    where u and v are the concentrations of the two species, D_u and D_v are the diffusion coefficients, F is the feed rate, and k is the kill rate.
    Here, we consider the model in a 1D spatioal domein, and the frequency of spatioal phase is zero.
    """
    def __init__(self, paramdict=None, display=True):

        if paramdict is not None:
            self.L = paramdict["L"]
            self.gridnum = paramdict["gridnum"]
            self.d = paramdict["d"]
            self.f = paramdict["f"]
            self.k = paramdict["k"]
            self.A = paramdict["A"]

        else:
            # Spatial-grid parameters.
            self.L = 250
            self.gridnum = int(2**10+1)
            self.d = 1.9  # 4.07, 1.9 etc.
            self.f = 0.018
            self.k = 0.052
            self.A = [0.00, 0.00]



        self.h = self.L/(self.gridnum-1)  # Grid spacing.
        # self.rx = np.linspace(0.0, self.L, num=self.gridnum)
        self.rx = np.linspace(-self.L/2, self.L/2, num=self.gridnum) #*
        self.D = np.array([[1.0, 0.0],
                            [0.0, self.d]])
        
        # if self.boundary == "Neumann":
        #     self.arg1, self.arg2 = 1, -2
        # elif self.boundary == "periodic":
        #     self.arg1, self.arg2 = -2, 1

        if display == True:
            print("model setting:")
            print("L: {:0.4f}".format(self.L))
            print("h: {:0.4f}".format(self.h))
            print("gridnum: {:d}".format(self.gridnum))
            print("d: {:0.4f}".format(self.d))
            print("f: {:0.4f}".format(self.f))
            print("k: {:0.4f}".format(self.k))
            print("D:\n[[{:0.4f}, {:0.4f}]\n[{:0.4f}, {:0.4f}]]".\
                format(self.D[0,0], self.D[0,1], self.D[1,0], self.D[1,1]))
            print("A: [{:0.4f}, {:0.4f}]".format(self.A[0], self.A[1]))


    def initialize_x0(self, center=0.0):
        """
        Construct the initial condition.
        """
        width = 20
        def periodic_tanh(rx, k=5, center=0):
            L = self.L
            shifted_rx = rx - center
            # The coefficient k has a positive sign here.
            return np.tanh(k * np.sin(2 * np.pi * shifted_rx / L))
        
        # u = 0.4*np.exp(-(self.rx-center)**2 / (2*width**2))
        # v = 0.5*np.tanh(100*(self.rx-center))+0.5

        #* u 
        # Wrap x periodically with period L.
        periodic_x = np.mod(self.rx - center + self.L / 2, self.L) - self.L / 2
        # Set a width-wide interval centered at ``center`` to one.
        u = 1.0 * (np.abs(periodic_x) <= width / 2)

        #* v
        v = 0.5 * periodic_tanh(self.rx, k=5, center=center) + 0.5
        return [u, v]



    def spde(self, init_x, tspan):
        """ 
        Solve the stochastic PDE with Heun's method.

        Parameters:
            init_x: initial condition (list of 1d-ndarray), 
                list[u,v]
                u size:[grid(rx)]
                v size:[grid(rx)]
            tspan:
        Returns:
            [U, V] (2-dimension Array)  size: [tspan, rx]
        """
        N = self.gridnum
        U = np.zeros([tspan.size, N])
        V = np.zeros([tspan.size, N])
        # set initial condition
        U[0] = init_x[0]; V[0] = init_x[1]
        x1 = copy.deepcopy(init_x)

        # start loop (stochastic)
        if np.any(self.A):
            # print("A is non-zero matrix: solve SPDE.")
            for i in range(1, tspan.size):
                dt = tspan[i] - tspan[i-1]; 
                # dW
                eta = normal(loc=0.0, scale=1.0, size=N) * np.sqrt(dt)
                dW = np.outer(self.A, eta) # size: [2, N]
                dW[:,-1] = dW[:,0] # periodic boundary condition
                dW = [dW[0,:], dW[1,:]]
                ## 1st step
                dF1 = mul_uv(self.dxdt(None, x1), dt)
                ## 2nd step
                x2 = add_uv(add_uv(x1, dF1), dW)
                dF2 = mul_uv(self.dxdt(None, x2), dt)
                _x = add_uv(x1, add_uv(dF1, dF2, const1=0.5, const2=0.5))
                new_x = add_uv(_x, dW)
                U[i] = new_x[0]; V[i] = new_x[1];
                x1 = new_x
            return [U, V]
        
        # start loop (deterministic)
        else:
            # print("A is zero matrix: solve (deterministic) PDE.")
            for i in range(1, tspan.size):
                dt = tspan[i] - tspan[i-1]; 
                ## 1st step
                dF1 = mul_uv(self.dxdt(None, x1), dt)
                ## 2nd step
                x2 = add_uv(x1, dF1)
                dF2 = mul_uv(self.dxdt(None, x2), dt)
                new_x = add_uv(x1, add_uv(dF1, dF2, const1=0.5, const2=0.5))
                U[i] = new_x[0]; V[i] = new_x[1];
                x1 = new_x
            return [U, V]



    def dxdt(self, t, x):
        """
        x: [u, v]
        """
        u, v = x
        N = self.gridnum
        # Fu, Fv:
        Fu, Fv = self.F(None, x)
        # Diffusion term
        D_u, D_v = self.diffusion(x)
        # Time derivative of each component.
        du = Fu + D_u
        dv = Fv + D_v
        return [du, dv]


    def diffusion(self, x):    
        """
        input:
            x: [u,v]
                Diffusion term with periodic boundary conditions.
        return: [D_u, D_v]
            diffusion for each component [u, v]
        """
        u, v = x
        Du, Dv = self.D[0,0], self.D[1,1] #* Du = 1.0, Dv = d
        # diffusion: x-axis
        D_ux = Du * \
            np.diff(u, n=2, prepend=u[-2], append=u[1]) / self.h**2
        D_vx = Dv * \
            np.diff(v, n=2, prepend=v[-2], append=v[1]) / self.h**2

        # D_ux = Du * \
        #     np.diff(u, n=2, prepend=u[1], append=u[-2]) / self.h**2
        # D_vx = Dv * \
        #     np.diff(v, n=2, prepend=v[1], append=v[-2]) / self.h**2
        return [D_ux, D_vx]


    def F(self, t, x):
        """
        x : list [u, v]
        """
        # ut = ∆u + u^2v - (f + k)u,
        # vt = d*∆v - u^2v + f (1 - v).
        u, v = x
        Fu = u*u*v - (self.f+self.k)*u
        Fv = -u*u*v + self.f*(1.-v)
        return [Fu, Fv]
    

    def J(self, t, x):
        """
        Jacobi matrix
            input:
                t: time (float)
                x: list [u, v]
                    each size: [gridnum, gridnum]
            output:
                return: 3-dimensional tensor
                Ex. mat[:,:,rx]: jacobi matrix at gridpoint [ry-th, rx-th]
        """
        u, v = x

        mat = np.empty([2,2,self.gridnum])
        mat[0,0,:] = 2.0 * u * v - (self.f + self.k)
        mat[0,1,:] = u * u
        mat[1,0,:] = -2.0 * u * v
        mat[1,1,:] = -u * u - self.f
        return mat
    


    def spde_with_chi(self, init_x, tspan):
        """ 
        Solve the stochastic PDE for X(chi, tau) with Heun's method.

        Parameters:
            init_chi: initial condition (list of 1d-ndarray), 
                list[u,v]
                u size:[grid(rx)]
                v size:[grid(rx)]
            tspan:
        Returns:
            [U, V] (2-dimension Array)  size: [tspan, rx]
            (rx is transfromed to chi [x → chi])
        """
        
        #* confirm that the instance has member `c`
        test_c(self)

        N = self.gridnum
        U = np.zeros([tspan.size, N])
        V = np.zeros([tspan.size, N])
        # set initial condition
        U[0] = init_x[0]; V[0] = init_x[1]
        x1 = copy.deepcopy(init_x)

        # start loop (stochastic)
        if np.any(self.A):
            # print("A is non-zero matrix: solve SPDE.")
            for i in range(1, tspan.size):
                dt = tspan[i] - tspan[i-1]; 
                # dW
                eta = normal(loc=0.0, scale=1.0, size=N) * np.sqrt(dt)
                dW = np.outer(self.A, eta) # size: [2, N]
                dW[:,-1] = dW[:,0] # periodic boundary condition
                dW = [dW[0,:], dW[1,:]]
                ## 1st step
                dF1 = mul_uv(self.dxdt_with_chi(None, x1), dt)
                ## 2nd step
                x2 = add_uv(add_uv(x1, dF1), dW)
                dF2 = mul_uv(self.dxdt_with_chi(None, x2), dt)
                _x = add_uv(x1, add_uv(dF1, dF2, const1=0.5, const2=0.5))
                new_x = add_uv(_x, dW)
                U[i] = new_x[0]; V[i] = new_x[1];
                x1 = new_x
            return [U, V]
        
        # start loop (deterministic)
        else:
            # print("A is zero matrix: solve (deterministic) PDE.")
            for i in range(1, tspan.size):
                dt = tspan[i] - tspan[i-1]; 
                ## 1st step
                dF1 = mul_uv(self.dxdt_with_chi(None, x1), dt)
                ## 2nd step
                x2 = add_uv(x1, dF1)
                dF2 = mul_uv(self.dxdt_with_chi(None, x2), dt)
                new_x = add_uv(x1, add_uv(dF1, dF2, const1=0.5, const2=0.5))
                U[i] = new_x[0]; V[i] = new_x[1];
                x1 = new_x
            return [U, V]


    
    def dxdt_with_chi(self, t, x):
        """
        x: [u, v]
            u, v: 1d-ndarray
        """
        u, v = x
        N = self.gridnum
        # Fu, Fv:
        Fu, Fv = self.F(None, x)
        # Diffusion term
        D_u, D_v = self.diffusion(x)
        # graident X
        grad_u = self.c * myfunc_gradient(u, self.h)
        grad_v = self.c * myfunc_gradient(v, self.h)
        # Time derivative of each component.
        du = Fu + D_u + grad_u
        dv = Fv + D_v + grad_v
        return [du, dv]




# %%
