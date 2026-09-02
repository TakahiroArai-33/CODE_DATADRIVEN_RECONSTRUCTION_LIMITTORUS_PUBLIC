#!/usr/bin/env python
# coding: utf-8


# %%

import numpy as np
import cupy as cp
import gc, os, sys
from typing import Literal
from functions.time_deco import log_execution_time
# from Module_euler import euler_nosave, euler_save
# from Module_heun import heun_nosave, heun_save



class GrayScottModel():
    """
    Equation (2) in Yadome PRE 2011
        ut = ∆u + u^2v - (F + k)u,
        vt = d*∆vxx - u^2v + F (1 - v).

    The parameter values are set to F = 0.018,k = 0.052, for which the kinetics part of Eq. (2) is excitable (Fig. 2). Periodic boundary conditions are used for Eq. (2), and the system size L = 250 is sufficiently larger than the typical size of a pulse solution. The spatial grid size is 512 and the time increment is 0.1.
                    [Yadome PRE (2011)]

    where u and v are the concentrations of the two species, 
    D_u and D_v are the diffusion coefficients, 
    F is the feed rate, and k is the kill rate.
    """

    def __init__(self, paramdict=None, display=True):

        self.gridnum = int(2**10+1)
        self.L = np.float64(250)
        self.d = cp.float64(1.9)
        self.f = cp.float64(0.018)
        self.k = cp.float64(0.052)

        # Override default parameters.
        if type(paramdict) is dict:
                if 'L' in paramdict:
                    self.L = np.float64(paramdict['L'])
                if 'gridnum' in paramdict:
                    self.gridnum = int(paramdict['gridnum'])
                if 'd' in paramdict:
                    self.d = cp.float64(paramdict['d'])
                if 'f' in paramdict:
                    self.f = cp.float64(paramdict['f'])
                if 'k' in paramdict:
                    self.k = cp.float64(paramdict['k'])
        else:
            pass


        self.h = cp.float64(self.L/(self.gridnum-1))
        self.rx = np.linspace(-self.L/2, self.L/2, num=self.gridnum)
        self.D = cp.array([[1.0, 0.0],
                           [0.0, self.d]])
        self.params = (self.f, self.k)
        


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



    def initialize_x0(self, center:float = 0.0) -> tuple[cp.ndarray, cp.ndarray]:
        """Construct a localized periodic initial condition."""
        width = 20
        def periodic_tanh(rx: np.ndarray, k:float=5.0, center:float=0.0) -> np.ndarray:
            L = self.L
            shifted_rx = rx - center
            return np.tanh(k * np.sin(2 * np.pi * shifted_rx / L))

        periodic_x = np.mod(self.rx - center + self.L / 2, self.L) - self.L / 2
        u = 1.0 * (np.abs(periodic_x) <= width / 2)

        v = 0.5 * periodic_tanh(self.rx, k=5.0, center=center) + 0.5

        u = cp.asarray(u)
        v = cp.asarray(v)

        return u, v



# Legacy Euler solvers retained for reference.

#     def pde_nosave(self, init_x: tuple[cp.ndarray, cp.ndarray], dt: float, iteration: int,
#            c:float=float(0.0)) -> tuple[cp.ndarray, cp.ndarray]:
#         """ 
#         Solve PDE with step-wise saving (every `step` time steps)

#         Parameters:
#         init_x : tuple of 1d-ndarray [u, v] 
#         tspan : 1d-ndarray
#         step : int, optional
#             Save results every `step` iterations (default=10)

#         Returns:
#         U_save, V_save : 2d-ndarray
#             Time evolution recorded every `step` steps
#         t_save : 1d-ndarray
#             Time values corresponding to recorded steps
#         """

#         u0, v0 = init_x
#         D = self.D
#         h = self.h
#         params = self.params
#         c = cp.float64(c)

#         u0 = cp.asarray(u0)
#         v0 = cp.asarray(v0)
        
#         final_state = euler_nosave(u0, v0, D, h, c, params, dt, iteration)
#         return final_state



#     def pde_save(self, init_x: tuple[np.ndarray, np.ndarray], 
#                   dt:float, iteration:int, save_step:int=int(1),
#                   c:float=0.0, ts:float=0.0) \
#         -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
#         """ 
#         Solve PDE with step-wise saving (every `step` time steps)

#         Parameters:
#         init_x : tuple of 1d-ndarray [u, v] 
#         tspan : 1d-ndarray
#         step : int, optional
#             Save results every `step` iterations (default=10)

#         Returns:
#         U_save, V_save : 2d-ndarray
#             Time evolution recorded every `step` steps
#         t_save : 1d-ndarray
#             Time values corresponding to recorded steps
#         """

#         u0, v0 = init_x
#         u0 = cp.asarray(u0)
#         v0 = cp.asarray(v0)

#         D = self.D
#         h = self.h
#         params = self.params
#         c = cp.float64(c)

#         t_size = iteration+1

#         if save_step == 1:
#             L = t_size
#         elif save_step >=2:
#             L = (t_size - 1) // save_step + 1
#         else:
#             print("Error: step must be positive integer")
#             sys.exit()
#         N = self.gridnum

#         # Allocate arrays for stepwise output.
#         U_save = cp.zeros([L, N])
#         V_save = cp.zeros([L, N])
#         t_save = cp.zeros(L)

#         # Store the initial condition.
#         U_save[0] = cp.copy(u0)
#         V_save[0] = cp.copy(v0)
#         t_save[0] = ts

#         return euler_save(U_save, V_save, t_save,
#                D, h, c, params,
#                dt, iteration, save_step)

    
#     # Legacy Heun solvers.

#     def pde_heun_nosave(self, init_x: tuple[cp.ndarray, cp.ndarray], dt: float, iteration: int,
#            c:float=float(0.0)) -> tuple[cp.ndarray, cp.ndarray]:
#         """ 
#         Solve PDE with step-wise saving (every `step` time steps)

#         Parameters:
#         init_x : tuple of 1d-ndarray [u, v] 
#         tspan : 1d-ndarray
#         step : int, optional
#             Save results every `step` iterations (default=10)

#         Returns:
#         U_save, V_save : 2d-ndarray
#             Time evolution recorded every `step` steps
#         t_save : 1d-ndarray
#             Time values corresponding to recorded steps
#         """

#         u0, v0 = init_x
#         D = self.D
#         h = self.h
#         params = self.params
#         c = cp.float64(c)

#         u0 = cp.asarray(u0)
#         v0 = cp.asarray(v0)
        
#         final_state = heun_nosave(u0, v0, D, h, c, params, dt, iteration)
#         return final_state



#     def pde_heun_save(self, init_x: tuple[np.ndarray, np.ndarray], 
#                   dt:float, iteration:int, save_step:int=int(1),
#                   c:float=0.0, ts:float=0.0) \
#         -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
#         """ 
#         Solve PDE with step-wise saving (every `step` time steps)

#         Parameters:
#         init_x : tuple of 1d-ndarray [u, v] 
#         tspan : 1d-ndarray
#         step : int, optional
#             Save results every `step` iterations (default=10)

#         Returns:
#         U_save, V_save : 2d-ndarray
#             Time evolution recorded every `step` steps
#         t_save : 1d-ndarray
#             Time values corresponding to recorded steps
#         """

#         u0, v0 = init_x
#         u0 = cp.asarray(u0)
#         v0 = cp.asarray(v0)

#         D = self.D
#         h = self.h
#         params = self.params
#         c = cp.float64(c)

#         t_size = iteration+1

#         if save_step == 1:
#             L = t_size
#         elif save_step >=2:
#             L = (t_size - 1) // save_step + 1
#         else:
#             print("Error: step must be positive integer")
#             sys.exit()
#         N = self.gridnum

#         # Allocate arrays for stepwise output.
#         U_save = cp.zeros([L, N])
#         V_save = cp.zeros([L, N])
#         t_save = cp.zeros(L)

#         # Store the initial condition.
#         U_save[0] = cp.copy(u0)
#         V_save[0] = cp.copy(v0)
#         t_save[0] = ts

#         return heun_save(U_save, V_save, t_save,
#                D, h, c, params,
#                dt, iteration, save_step)




    


# if __name__ == '__main__':
#     from functions.time_deco import log_execution_time
#     import sys

#     # Redirect standard output to test_output.txt.
#     original_stdout = sys.stdout
#     sys.stdout = open('test_output.txt', 'w')
    

#     # DT = 0.001
#     # SAVE_STEP = int(5000)
#     DT = 0.01
#     SAVE_STEP = int(100)
#     ITERATION = int(20)
#     LENGTH = float(5000)
#     SECTION = 7.0

#     paramdict = {
#         "L": 250,
#         "gridnum": int(2**10+1)
#         # "gridnum": int(2**13+1)
#     }




#     @log_execution_time
#     def test1():
#         C_VELOCITY = 0.0

#         GSM = GrayScottModel(paramdict=paramdict)
#         x0 = GSM.initialize_x0(center=0.0)
#         for i in range(ITERATION):
#             if i < ITERATION-1:
#                 u, v = GSM.pde_nosave(init_x=x0, dt=DT, iteration=int(LENGTH/DT), c=C_VELOCITY)
#                 x0 = (u,v)

#             elif i == ITERATION-1:
#                 Usave, Vsave, tsave = GSM.pde_save(init_x=x0, 
#                                     dt=DT, iteration=int(LENGTH/DT),
#                                     save_step=SAVE_STEP, 
#                                     ts=i*LENGTH, c=C_VELOCITY)
#             else:
#                 print("Error: Invalid iteration")
#                 sys.exit()

#         np.savez("test1.npz", Usave=Usave, Vsave=Vsave, 
#                  tsave=tsave, rx=GSM.rx)
#         print("save test1.npz")
        


#     @log_execution_time
#     def test2():
#         C_VELOCITY = 0.043105018691330634

#         GSM = GrayScottModel(paramdict=paramdict)
#         x0 = GSM.initialize_x0(center=0.0)
#         for i in range(ITERATION):
#             if i < ITERATION-1:
#                 u, v = GSM.pde_nosave(init_x=x0, dt=DT, iteration=int(LENGTH/DT), c=C_VELOCITY)
#                 x0 = (u,v)

#             elif i == ITERATION-1:
#                 Usave, Vsave, tsave = GSM.pde_save(init_x=x0, 
#                                     dt=DT, iteration=int(LENGTH/DT),
#                                     save_step=SAVE_STEP, 
#                                     ts=i*LENGTH, c=C_VELOCITY)
#             else:
#                 print("Error: Invalid iteration")
#                 sys.exit()

#         np.savez("test2.npz", Usave=Usave, Vsave=Vsave, 
#                  tsave=tsave, rx=GSM.rx)
#         print("save test2.npz")



#     @log_execution_time
#     def test3():
#         C_VELOCITY = 0.0

#         GSM = GrayScottModel(paramdict=paramdict)
#         x0 = GSM.initialize_x0(center=0.0)
#         for i in range(ITERATION):
#             if i < ITERATION-1:
#                 u, v = GSM.pde_heun_nosave(init_x=x0, dt=DT, iteration=int(LENGTH/DT), c=C_VELOCITY)
#                 x0 = (u,v)

#             elif i == ITERATION-1:
#                 Usave, Vsave, tsave = GSM.pde_heun_save(init_x=x0, 
#                                     dt=DT, iteration=int(LENGTH/DT),
#                                     save_step=SAVE_STEP, 
#                                     ts=i*LENGTH, c=C_VELOCITY)
#             else:
#                 print("Error: Invalid iteration")
#                 sys.exit()

#         np.savez("test3.npz", Usave=Usave, Vsave=Vsave, 
#                  tsave=tsave, rx=GSM.rx)
#         print("save test3.npz")
        


#     @log_execution_time
#     def test4():
#         C_VELOCITY = 0.043105018691330634

#         GSM = GrayScottModel(paramdict=paramdict)
#         x0 = GSM.initialize_x0(center=0.0)
#         for i in range(ITERATION):
#             if i < ITERATION-1:
#                 u, v = GSM.pde_heun_nosave(init_x=x0, dt=DT, iteration=int(LENGTH/DT), c=C_VELOCITY)
#                 x0 = (u,v)

#             elif i == ITERATION-1:
#                 Usave, Vsave, tsave = GSM.pde_heun_save(init_x=x0, 
#                                     dt=DT, iteration=int(LENGTH/DT),
#                                     save_step=SAVE_STEP, 
#                                     ts=i*LENGTH, c=C_VELOCITY)
#             else:
#                 print("Error: Invalid iteration")
#                 sys.exit()

#         np.savez("test4.npz", Usave=Usave, Vsave=Vsave, 
#                  tsave=tsave, rx=GSM.rx)
#         print("save test4.npz")

#     try:
#         test1()
#         test2()
#         test3()
#         test4()

#     finally:
#         # Restore standard output.
#         sys.stdout.close()
#         sys.stdout = original_stdout
