#!/usr/bin/env python3
# coding: utf-8

"""
Calculate the limit-cycle trajectory of the FHN diffusion systems 1 and 2
preparation:
    Calculate the limit-cycle trajectory by 1_trajectory.py,
    and make ./UV1.npz, ./UV2.npz data. (rename ./UV.npz)
output:
    * ./QuQv1.npz, ./QuQv2.npz: spatiotemporal dynamics of phase sensitivity function.

To obtain the Us*, Ut*, solve the adjoint equation with orthonormal constraint.
Zs(x-Phi, Theta) = -Us*
Zt(x-Phi, Theta) = Ut*
"""


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



def split_and_squeeze(X, n, axis=0):
    # Split X into n arrays.
    result = np.split(X, n, axis=axis)
    # Remove singleton dimensions from each part.
    result_squeezed = [np.squeeze(part) for part in result]
    return result_squeezed


def mul_uv(uv, const):
    """
    uv: list[u, v, ...] or nest 
        u, v, ...: 1d-ndarray
    const: float
    """
    n = len(uv)
    new = const*np.concatenate(uv)
    # new_u = const*uv[0]
    # new_v = const*uv[1]
    return np.split(new, n)


def add_uv(uv1, uv2, const1=1.0, const2=1.0):
    """ 
    uv1, uv2: list[u, v, ...] or nest
        u, v, ...: 1d-ndarray
    const: float    
    """
    n = len(uv1)
    new = const1*np.concatenate(uv1) + const2*np.concatenate(uv2)
    # new_u = const1*uv1[0] + const2*uv2[0]
    # new_v = const1*uv1[1] + const2*uv2[1]
    return np.split(new, n)




def is_diagonal(matrix):
    # Extract the diagonal.
    diag = np.diagonal(matrix)
    # Test whether every off-diagonal element is zero.
    matrix_no_diag = matrix - np.diag(diag)
    return np.all(matrix_no_diag == 0)



class Adjoint():
    """
    cf. Nakao,(2014),PRX
    """  

    def __init__(self, pde_class):
        """ 
        adjoint class for 1D-FHN reaction diffusion system 
        """
        ## func
        self.F = pde_class.F
        self.J = pde_class.J
        self.D = pde_class.D

        ##
        self.N = pde_class.gridnum
        self.h = pde_class.h
        self.rx = pde_class.rx
        ##

        if hasattr(pde_class, 'c'):
            self.c = pde_class.c
        else: 
            print()
            print("Notice!!, instance does not have member `c`")
            print("Terminate Program")
            sys.exit()
        
        if self.c == 0:
            print()
            print("c = 0: use `pde_class.dxdt` in self._right_Ut")
            self.dxdt = pde_class.dxdt

        else:
            print()
            print("c neq 0: use `pde_class.dxdt_with_chi` in self._right_Ut")
            self.dxdt = pde_class.dxdt_with_chi


        if not(is_diagonal(self.D)):
            print()
            print("Warning!!: `Adjoint` class does not support non-diagonal `D`")
        
        else:
            print()
            print("D is diagonal.")


        



    def JtU(self, x, u):
        """
        Ref. Eq. (A) in the adjoint-equation derivation.
        right hand side, second term

        input: 
        x: state variable, (list of 1d-ndarray),
            i.e., (u, v)
        u: left eigen function (vector) (list of 1d-ndarray) 
            i.e., (Uu, Uv) 
        """
        # number of component of list `x`, i.e., dim = 2 (u,v)
        dim = len(x)

        # Jt[:,:,rx] Jacobian matrix at rx-th gridpoint
        tensor_Jt = self.J(None, x).transpose([1,0,2])
        tensor_U = np.stack(u, axis=0)
        ##
        JtU = np.zeros([dim, self.N])

        # J*Q; axis 0,1: component (i.e. u, v); axis2: spatial grid
        for i in range(dim):
            for j in range(dim):
                JtU[i,:] += tensor_Jt[i,j,:] * tensor_U[j,:]
        return split_and_squeeze(JtU, dim, axis=0)



    def U_diffusion(self, u):
        """
        Ref. Eq. (A) in the adjoint-equation derivation.
        right hand side, first term

        input: 
        u: left eigen function (vector) (list of 1d-ndarray) 
            i.e., (Uu, Uv) 
        """
        D = np.diag(self.D)  # Equivalent to D.T for supported diagonal D.
        dim = len(u)

        # diffusion: x-axis
        diffusion = []
        for i in range(dim):
            _d = D[i]; _u = u[i];
            diffusion.append(
                _d * np.diff(_u, n=2, 
                             prepend=_u[-2], append=_u[1]) / self.h**2
            )
        return diffusion


    
    def U_gradient(self, u):
        """
        Ref. Eq. (A) in the adjoint-equation derivation.
        right hand side, third term

        input: 
        u: left eigen function (vector) (list of 1d-ndarray) 
            i.e., (Uu, Uv)
                Uu & Uv: 1d-ndarray
        """
        dim = len(u)

        gradient = []
        for i in range(dim):
            _u = u[i]
            # Use the periodic finite-difference implementation.
            grad_array = myfunc_gradient(_u, self.h)
            gradient.append( self.c*grad_array )    #* c*grad(U)
        return gradient



    def dudt(self, t, x, up):
        r"""
        Ref. Eq. (B) in the adjoint-equation derivation.
        \frac{\partial}{\partial t}\bm{U}_p^*(x-\Phi, t)
        = -D^{\dagger} \nabla^2 \bm{U}_p^*(x-\Phi, t)
        -J^{\dagger}(t) \bm{U}_p^* (x-\Phi, t)
        + c\frac{\partial}{\partial x} \bm{U}_p^*(x-\Phi, t). 

        each term in right hand side is return by following function:
            * self.U_diffusion
            * self.JtU
            * self.dudx

        input: 
        x: state variable, (list of 1d-ndarray),
            i.e., (u, v)
        up: us or ut
            us: left eigen function for spatial phase (vector) (list of 1d-ndarray) 
            ut: left eigen function for temporal phase
                i.e., us, = (us_u, us_v, ...),   ut =(ut_u, ut_v, ...)
                
        return: 
            list of 1d-ndarray, representing du/dt at each spatial grid point.
        """
        # jacobian
        JtUp = self.JtU(x, up)
        # Diffusion term
        Dp  = self.U_diffusion(up)
        # third term
        grad_Up = self.U_gradient(up)    #* c*grad(U)

        # du/dt
        dUp_dt = []
        dim = len(x)
        for i in range(dim):
            dUp_dt.append(- JtUp[i] - Dp[i] + grad_Up[i])
        # [Up_u, Up_v, ...]; each elements, Up_u,..., is 1d-ndarray
        return dUp_dt


    def _right_Us(self, x):
        """ 
        input: 
            x: (list of 1d-ndarrays)
            [A 2d-ndarray can be used instead of list of 1d-ndarray]
        """
        right_Us = []
        for _x in x:
            # Use the periodic finite-difference implementation.
            right_Us.append(myfunc_gradient(_x, self.h))
        return right_Us


    def _right_Ut(self, x):
        """
        input:
            x: list of 1d-ndarrays
            [A 2d-ndarray can be used instead of list of 1d-ndarray]
        """

        right_Ut = []
        #* if self.c is 0, self.dxdt is class_pde.dxdt
        #* elif self.c is not 0, self.dxdt is class_pde.dxdt_with_chi
        for _dxdt in self.dxdt(None, x):
            right_Ut.append(_dxdt/self.omega)
        return right_Ut



    def _calculate_delta(self, Us, Ut, X):
        """
        calculate delta function. 
        input:
            Us: [Us_a, Us_b, ...]  (list of 2d-ndarray)
                Us_a: 2d-ndarray, i.e., Us_a has [t-grid, rx-grid]
            Ut: [Ut_a, Ut_b, ...]  (list of 2d-ndarray)
            X: [a, b, ...]  (list of 2d-ndarray)
                a: 2d-ndarray [t-grid, rx-grid]
        return:
            delta: 2d-ndarray size:[t-grid, 4]
                delta[:,0]: delta_{ss}
                delta[:,1]: delta_{st}
                delta[:,2]: delta_{ts}
                delta[:,3]: delta_{tt}
        """
        dim = len(X)
        L = X[0].shape[0]
        delta = np.zeros([L,4])
    
        ## lambda func
        f = lambda X, i: [x_[i] for x_ in X]
        g = lambda X, i: np.stack([x_[i] for x_ in X], axis=0) # get ith-row in each 2d-ndarray included in X.
        inner_prod = lambda A, B: np.trapz(np.sum(A*B, axis=0), x=self.rx)

        ## calculate delta 
        for i in range(L):
            # left and right eigenfunction of Us, Ut [Us*, Ut*].
            x = f(X, i); 
            left_Us_i = g(Us, i); left_Ut_i = g(Ut, i)
            right_Us_i = np.stack(self._right_Us(x), axis=0)
            right_Ut_i = np.stack(self._right_Ut(x), axis=0)
            ## 
            delta_ss = inner_prod(left_Us_i, right_Us_i)
            delta_st = inner_prod(left_Us_i, right_Ut_i)
            delta_ts = inner_prod(left_Ut_i, right_Us_i)
            delta_tt = inner_prod(left_Ut_i, right_Ut_i)

            delta[i,0] = delta_ss
            delta[i,1] = delta_st
            delta[i,2] = delta_ts
            delta[i,3] = delta_tt
        return delta



    def _solve_adjoint_for_one_cycle(self, up0, X, tspan):
        """
        Integrate Up = Us or Ut backward for one cycle and return it in
        forward-time order.

        Parameters:
            up0: initial-state (in backward time evolution) of Up
                [Up_a0, Up_b0, ...], [list of 1d-ndarray]
                Up_a0: 1d-ndarray, i.e., Ua(t=0(=T), rx),
            X: [a, b, ...] [list of 2d-ndarray]
                a: 2d-ndarray [t-grid, rx-grid]
            tspan: time grids of X [1d-ndarray]
        Returns:
            Up: [Up_a, Up_b,...] [list of 2d-ndarray]
        """

        dim = len(X)
        Up = [np.zeros(x.shape) for x in X]
        for i in range(dim):
            Up[i][-1] = up0[i]

        # Heun
        g = lambda X, i: [x_[::-1][i] for x_ in X]
        for i in range(tspan.size-1):
            dt = tspan[::-1][i+1] - tspan[::-1][i]
            x1 = g(X, i)
            u1 = g(Up, i)
            k1 = mul_uv(self.dudt(None, x1, u1), dt)
            ##
            x2 = g(X, i+1)
            u2 = add_uv(u1, k1)
            k2 = mul_uv(self.dudt(None, x2, u2), dt)
            ##
            u_new = add_uv(u1, add_uv(k1, k2, const1=0.5, const2=0.5))
            for j in range(dim):
                _Up = Up[j]; _u_new = u_new[j]
                _Up[::-1][i+1] = _u_new
        return Up



    def _orthogonalization(self, Us, Ut, X):
        """
        orthogonalize left Us and Ut against right Ut and Us, respectively.
        input:
            Us: [Us_a, Us_b, ...]  (list of 2d-ndarray)
                Us_a: 2d-ndarray, i.e., Us_a has [t-grid, rx-grid]
            Ut: [Ut_a, Ut_b, ...]  (list of 2d-ndarray)
            X: [a, b, ...]  (list of 2d-ndarray)
                a: 2d-ndarray [t-grid, rx-grid]
        return 
            new_Us: new_Us has the same format as Us.
            new_Ut: new_Ut has the same format as Us.
        """
        dim = len(X)
        L = X[0].shape[0]
        f = lambda X, i: [x_[i] for x_ in X]
        g = lambda X, i: np.stack([x_[i] for x_ in X], axis=0)

        #* orthogonalize left Us against right Ut.
        new_Us = [np.zeros_like(U) for U in Us]
        for i in range(L):
            # orthogonalize A against B
            x = f(X, i);
            A = g(Us, i); B = np.stack(self._right_Ut(x), axis=0)

            #[[A1(0), A1(1), ... A1(L)],     [[B1(0), B1(1), ... B1(L)],
            # [A2(0), A2(1), ... A2(L)],     [B2(0), B2(1), ... B2(L)],
            #  ...]                          ...]

            A_dot_B = np.trapz(np.sum(A*B, axis=0), x=self.rx)
            B_dot_B = np.trapz(np.sum(B*B, axis=0), x=self.rx)
            A = A - (A_dot_B / B_dot_B) * B
            A = split_and_squeeze(A, dim, axis=0)

            # assignment
            for j in range(dim):
                # assign A[j] to ith row of jth nd-array in new_Us
                new_Us[j][i] = A[j]


        #* orthogonalize left Ut against right Us.
        new_Ut = [np.zeros_like(U) for U in Ut]
        for i in range(L):
            # orthogonalize A against B
            x = f(X, i); 
            A = g(Ut, i); B = np.stack(self._right_Us(x), axis=0)

            #[[A1(0), A1(1), ... A1(L)],     [[B1(0), B1(1), ... B1(L)],
            # [A2(0), A2(1), ... A2(L)],     [B2(0), B2(1), ... B2(L)],
            #  ...]                          ...]
            
            A_dot_B = np.trapz(np.sum(A*B, axis=0), x=self.rx)
            B_dot_B = np.trapz(np.sum(B*B, axis=0), x=self.rx)
            A = A - (A_dot_B / B_dot_B) * B
            A = split_and_squeeze(A, dim, axis=0)

            # assignment
            for j in range(dim):
                # assign A[j] to ith row of jth nd-array in new_Us
                new_Ut[j][i] = A[j]

        return new_Us, new_Ut



    def _normalization(self, Us, Ut, X, scale=1.0):
        dim = len(X)
        L = X[0].shape[0]
        f = lambda X, i: [x_[i] for x_ in X]
        g = lambda X, i: np.stack([x_[i] for x_ in X], axis=0)
        
        #* normalize Us 
        new_Us = [np.zeros_like(U) for U in Us]
        for i in range(L):
            x = f(X, i);
            A = g(Us, i); B = np.stack(self._right_Us(x), axis=0)

            #[[A1(0), A1(1), ... A1(L)],     [[B1(0), B1(1), ... B1(L)],
            # [A2(0), A2(1), ... A2(L)],     [B2(0), B2(1), ... B2(L)],
            #  ...]                          ...]

            A_dot_B = np.trapz(np.sum(A*B, axis=0), x=self.rx)
            A /= A_dot_B #* scale to 1.0
            A *= scale #* scale to `scale`
            A = split_and_squeeze(A, dim, axis=0)
            for j in range(dim):
                # assign A[j] to ith row of jth nd-array in new_Us
                new_Us[j][i] = A[j]

    
        #* normalize Ut
        new_Ut = [np.zeros_like(U) for U in Ut]
        for i in range(L):
            x = f(X, i); 
            A = g(Ut, i); B = np.stack(self._right_Ut(x), axis=0)

            #[[A1(0), A1(1), ... A1(L)],     [[B1(0), B1(1), ... B1(L)],
            # [A2(0), A2(1), ... A2(L)],     [B2(0), B2(1), ... B2(L)],
            #  ...]                          ...]

            A_dot_B = np.trapz(np.sum(A*B, axis=0), x=self.rx)
            A /= A_dot_B #* scale to 1.0
            A *= scale #* scale to `scale`
            A = split_and_squeeze(A, dim, axis=0)
            for j in range(dim):
                # assign A[j] to ith row of jth nd-array in new_Us
                new_Ut[j][i] = A[j]

        return new_Us, new_Ut
    



    @log_execution_time
    def main_routine(self, X, tspan, 
                     init_condition_Us, 
                     init_condition_Ut,
                     eps1=1.0e-5, eps2=1.0e-9,
                     scale_size = 1.0,
                     option_display=5, ):
        r"""
        Ref. Eqs. (38, 39, 44), Kawamura, Physica D (2015).
        X: [a, b, ...] (list-2d array)
            limit-torus at Phi=const(0) 
            a(t, r) = a[t-grid, rx-grid]
            b(t, r) = b[t-grid, rx-grid]
        tspan: 1d-array tspan[t-grid]
        init_condition_Us, init_condition_Ut: list of 1d-ndarray
            [len(init_condition_Us) and len(init_condition_Ut) is len(X).]
        scale: Normalization magnitude for left Us and Ut. During iteration,
            normalize to scale*\delta_{qp}; normalize returned values to
            1.0*\delta_{qp}.
        option_display: Errors will be displayed after every `option_display` updates.
        """
        info_dict = {"count":[],
                     "Us_error":[],
                     "Ut_error":[],
                     "delta":[]}
        
        dim = len(X)

        #* frequency of oscillation
        period = tspan[-1]-tspan[0]
        self.omega = 2.0*np.pi/period

        #* period of translation
        # self.c = self.c

        #* Symmetric initial conditions. [ref. Eqs(104,105) in Kawamura (2015)]
        #* c = 0 case:
            #*  Us(-x,t;Phi) = -Us(x,t;Phi)  [Us is left]
            #*  Ut(-x,t;Phi) = Ut(x,t;Phi)   [Ut is left]
        
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

        # us_0 = [normal(loc=0., scale=1.0, size=_x[0].shape) for _x in X]
        # ut_0 = [normal(loc=0., scale=1.0, size=_x[0].shape) for _x in X] 

        #* calculate for the first cycle
        ref_new_Us = self._solve_adjoint_for_one_cycle(us_0, X, tspan)
        ref_new_Ut = self._solve_adjoint_for_one_cycle(ut_0, X, tspan)
        _Us, _Ut = self._orthogonalization(ref_new_Us, ref_new_Ut, X)
        new_Us, new_Ut = self._normalization(_Us, _Ut, X, scale=scale_size)

        # Track convergence before normalization and orthogonalization.
        info_dict["count"].append(int(0))
        info_dict["Us_error"].append([np.nan for _ in range(dim)])
        info_dict["Ut_error"].append([np.nan for _ in range(dim)])

        # Track the Kronecker-delta constraints.
        info_dict["delta"].append(
            self._calculate_delta(ref_new_Us, ref_new_Ut, X)
            )
        

        #* repetition (after the second cycle)
        count = int(1)  # Number of updates.
        retrieve_init = lambda X : [x_[0] for x_ in X] 
        while (1):
            #* old <- new
            ref_old_Us, ref_old_Ut = deepcopy(ref_new_Us), deepcopy(ref_new_Ut)
            old_Us, old_Ut = deepcopy(new_Us), deepcopy(new_Ut)
            us_0, ut_0 = retrieve_init(old_Us), retrieve_init(old_Ut)

            #* update
            ref_new_Us = self._solve_adjoint_for_one_cycle(us_0, X, tspan)
            ref_new_Ut = self._solve_adjoint_for_one_cycle(ut_0, X, tspan)
            _Us, _Ut = self._orthogonalization(ref_new_Us, ref_new_Ut, X)
            new_Us, new_Ut = self._normalization(_Us, _Ut, X, scale=scale_size)

            # Convergence before normalization and orthogonalization.
            error_Us_list, error_Ut_list = [],[]
            for j in range(dim):
                # Maximum error over time for each component of Us and Ut.
                error_Usj = LA.norm(ref_new_Us[j]/scale_size \
                                    - ref_old_Us[j]/scale_size, axis=1).max()
                error_Utj = LA.norm(ref_new_Ut[j]/scale_size \
                                    - ref_old_Ut[j]/scale_size, axis=1).max()
                error_Us_list.append(error_Usj)
                error_Ut_list.append(error_Utj)
            info_dict["count"].append(count)
            info_dict["Us_error"].append(error_Us_list)
            info_dict["Ut_error"].append(error_Ut_list)

            # Divide by scale_size to restore target values of zero or one.
            info_dict["delta"].append(
                self._calculate_delta(ref_new_Us, ref_new_Ut, X) / scale_size
                )

            # Difference between consecutive delta-function states.
            old_delta = info_dict["delta"][int(count-1)]
            new_delta = info_dict["delta"][int(count)]
            inf_norm = np.max(np.abs(new_delta-old_delta), axis=0) 
            ##
            relative_error_ss = inf_norm[0]
            relative_error_st = inf_norm[1]
            relative_error_ts = inf_norm[2]
            relative_error_tt = inf_norm[3]

            #* print
            if (count % option_display) == 0:
                print("count: {:d} ".format(count))
                ##
                string = "Us error:"
                for j in range(dim):
                    string += " dim{:d}: {:0.4e},".format(j+1, error_Us_list[j])
                print(string)
                ##
                string = "Ut error:"
                for j in range(dim):
                    string += " dim{:d}: {:0.4e},".format(j+1, error_Ut_list[j])
                print(string)
                ##
                str1 = "delta_ss: {:0.4e},  ".format(relative_error_ss)
                str2 = "delta_st: {:0.4e},  ".format(relative_error_st)
                str3 = "delta_ts: {:0.4e},  ".format(relative_error_ts)
                str4 = "delta_tt: {:0.4e},  ".format(relative_error_tt)
                print(str1+str2+str3+str4)

            #* Termination based on convergence criteria (inf_norm of delta function.)
            # if np.all(np.array(error_Us_list) < eps) &\
            #      np.all(np.array(error_Ut_list) < eps):
            #     break
            flag_ss = (relative_error_ss <= eps2)
            flag_st = (relative_error_st <= eps2)
            flag_ts = (relative_error_ts <= eps2)
            flag_tt = (relative_error_tt <= eps2)
            flag_Us = np.all(np.array(error_Us_list) <= eps1)
            flag_Ut = np.all(np.array(error_Ut_list) <= eps1)
            if flag_ss and flag_st and flag_ts and flag_tt and flag_Us and flag_Ut:
                break

            #* add count
            count += 1

        ##
        info_dict["count"] = np.array(info_dict["count"])
        info_dict["Us_error"] = np.array(info_dict["Us_error"])
        info_dict["Ut_error"] = np.array(info_dict["Ut_error"])
        info_dict["delta"] = np.stack(info_dict["delta"], axis=0)

        #* scale to 1.0
        new_Us, new_Ut = self._normalization(new_Us, new_Ut, X, scale=1.0)
        
        return new_Us, new_Ut, info_dict
    

    def right_Us(self, X, tspan):
        """
        return right eigenfunction Us
        input: 
            X: list of 2d-ndarray [t-grid, rx-grid]
            [X1(x,t), X2(x,t),...]
            tspan: 1d-ndarray (0~T)
        return 
            self.rx: 1d-ndarray
            theta: transform tspan to theta(0~2pi).
            Us: list of 2d-ndarray
        """
        dim = len(X)
        L = X[0].shape[0]
        f = lambda X, i: [x_[i] for x_ in X]
        # g = lambda X, i: np.stack([x_[i] for x_ in X], axis=0)
        ##
        Us = [np.zeros_like(Y) for Y in X]
        for i in range(L): 
            x = f(X, i)
            A = self._right_Us(x)
            for j in range(dim):
                Us[j][i] = A[j]

        theta = 2.0*np.pi* (tspan-tspan[0])/np.ptp(tspan)

        # Return coordinates and values in x, y, z(x, y) form.
        return self.rx, theta, Us


    def right_Ut(self, X, tspan):
        """ 
        return right eigenfunction Ut
        input: 
            X: list of 2d-ndarray [t-grid, rx-grid]
            [X1(x,t), X2(x,t),...]
            tspan: 1d-ndarray (0~T)
        return 
            self.rx: 1d-ndarray
            theta: transform tspan to theta(0~2pi).
            Us: list of 2d-ndarray
        """
        dim = len(X)
        L = X[0].shape[0]
        f = lambda X, i: [x_[i] for x_ in X]
        # g = lambda X, i: np.stack([x_[i] for x_ in X], axis=0)
        ##
        Ut = [np.zeros_like(Y) for Y in X]
        for i in range(L): 
            x = f(X, i)
            A = self._right_Ut(x)
            for j in range(dim):
                Ut[j][i] = A[j]

        theta = 2.0*np.pi* (tspan-tspan[0])/np.ptp(tspan)

        # Return coordinates and values in x, y, z(x, y) form.
        return self.rx, theta, Ut




# %%
