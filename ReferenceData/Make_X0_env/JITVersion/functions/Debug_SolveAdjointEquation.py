#!/usr/bin/env python3
# coding: utf-8

# In[]

import copy, sys, time
from copy import deepcopy
import numpy as np
import numpy.linalg as LA
import tqdm 
from typing import Literal
from numpy.random import random, normal, seed
from scipy import interpolate

import sys
from time_deco import log_execution_time


#* myfunc_gradient
sys.path.append("../functions")
from myfunc_gradient import myfunc_gradient

from SolveAdjointEquation import Adjoint



class Debug(Adjoint):

    # def __init__(self, pde_class):
    #     super().__init__(pde_class)

    # Temporal central difference along the limit cycle.
    def central_difference(self, X, tspan):
        """ 
        X: list of 2d-ndarray [tspan-grid, rx-grid]
            X0(x, t) in R^1 corresponding to the limit cycle.
        tspan: Uniform samples from t0 to t0 + T, where T is the period.
        """
        omega = 2.0*np.pi/np.ptp(tspan)
        h = np.abs(tspan[1] - tspan[0])
        list1 = []
        for _X in X:
            dXdt = np.empty_like(_X)
            for i in range(1, tspan.size-1):
                dX = _X[i+1,:] - _X[i-1,:]
                dXdt[i] = dX / (2.0*h)
            #* i = 0, -1 (t=t0, t0+T)
            dXdt[0] = (_X[1]-_X[-2])/(2.*h)
            dXdt[-1] = dXdt[0]
            ##
            list1.append(dXdt/omega)
        return list1

        
    def right_Ut(self, X, tspan):
        return super().right_Ut(X, tspan)
    
    def right_Us(self, X, tspan):
        return super().right_Us(X, tspan)


    # @log_execution_time
    def debug_routine(self, X, tspan, 
                     init_condition_Us, 
                     init_condition_Ut):
        """
        Solve the adjoint equation for one cycle without normalization or
        orthogonalization.
        """
        
        dim = len(X)

        #* frequency of oscillation
        period = tspan[-1]-tspan[0]
        self.omega = 2.0*np.pi/period

        #* period of translation
        # self.c = self.c

        
        if len(init_condition_Us) == len(X):
            us_0 = init_condition_Us

            for us in init_condition_Us:
                if not(us.size == self.rx.size):
                    print("init_condition_Us has wrong size")
                    sys.exit()

        else: 
            print("init_condition_Us has invalid values") 
            sys.exit()

        if len(init_condition_Ut) == len(X):
            ut_0 = init_condition_Ut

            for ut in init_condition_Ut:
                if not(ut.size == self.rx.size):
                    print("init_condition_Ut has wrong size")
                    sys.exit()

        else: 
            print("init_condition_Us has invalid values") 
            sys.exit()

        #* main of the routine.
        new_Us = self._solve_adjoint_for_one_cycle(us_0, X, tspan)
        new_Ut = self._solve_adjoint_for_one_cycle(ut_0, X, tspan)
        return new_Us, new_Ut
    


    def normalization(self, Us, Ut, X):
        return super()._normalization(Us, Ut, X, scale=1.0)
    
    def orthogonalization(self, Us, Ut, X):
        return super()._orthogonalization(Us, Ut, X)
    

    def project_to_right_Us(self, left_Up, X, tspan):
        """ 
        left_Up: list of 2d-ndarray, each size is [rx-grid, t-grid]
        """
        _, _, right_Us = self.right_Us(X, tspan)
        stackfunc = lambda X, i: np.stack([x_[i] for x_ in X], axis=0)

        array = []
        for i in range(tspan.size):
            A = stackfunc(left_Up, i)
            B = stackfunc(right_Us, i)
            A_dot_B = np.trapz(np.sum(A*B, axis=0), self.rx)
            array.append(A_dot_B)
        return np.array(array)


    def project_to_right_Ut(self, left_Up, X, tspan):
        _, _, right_Ut = self.right_Ut(X, tspan)
        stackfunc = lambda X, i: np.stack([x_[i] for x_ in X], axis=0)

        array = []
        for i in range(tspan.size):
            A = stackfunc(left_Up, i)
            B = stackfunc(right_Ut, i)
            A_dot_B = np.trapz(np.sum(A*B, axis=0), self.rx)
            array.append(A_dot_B)
        return np.array(array)


# %%
