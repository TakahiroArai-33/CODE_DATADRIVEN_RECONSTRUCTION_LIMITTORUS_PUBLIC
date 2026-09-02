#!/usr/bin/env python3
# coding: utf-8


"""Run Bayesian inference on the GPU."""

import numpy as np
import sys, os
import pickle
from numpy import linalg as LA
import gc, os, re, sys, copy
from collections import defaultdict
from typing import Tuple, List, Callable, Literal






# Load paths and run parameters from environment variables.
import  Module_get_envname
SPDE_DIR = Module_get_envname.SPDE_DIR 
CONCATENATE_DIR = Module_get_envname.CONCATENATE_DIR
INTERIM_DIR = Module_get_envname.INTERIM_DIR
PHASE_DIR = Module_get_envname.PHASE_DIR
BAYES_DIR = Module_get_envname.BAYES_DIR
STDOUT_DIR = Module_get_envname.STDOUT_DIR
USE_DEBUG_FILES = Module_get_envname.USE_DEBUG_FILES
filename1 = Module_get_envname.filename1  # Concatenated SPDE data
filename2 = Module_get_envname.filename2  # Per-run intermediate data
filename3 = Module_get_envname.filename3  # Intermediate data shared by epsilon and sigma
filename4 = Module_get_envname.filename4  # Phase data
filename5 = Module_get_envname.filename5  # Bayesian-inference data
stdoutname = os.path.join(Module_get_envname.stdout_dir, "stdout_bayes.txt") #* stdout


init_a_idx = Module_get_envname.init_a_idx
init_b_idx = Module_get_envname.init_b_idx
eps_val = Module_get_envname.eps_val
sigma_val = Module_get_envname.sigma_val


# Import workflow helpers.
import Module_Bayes_GPU as Module_Bayes
BayesianEstimator = Module_Bayes.BayesianEstimator
OutputFunction = Module_Bayes.OutputFunction
pick_L = Module_Bayes.pick_L

import Module_get_filelist  
get_phase_file_list = Module_get_filelist.get_phase_file_list

import Module_savefunc
save_npz_atomic = Module_savefunc.save_npz_atomic

import Module_calphase_toolbox
get_common_time = Module_calphase_toolbox.get_common_time

import Module_ArrangeData_GPU as Module_ArrangeData
ArrangeData = Module_ArrangeData.ArrangeData

# Half the spatial domain length.
L = Module_Bayes.params["L"]
# Whether to store G and partial G as complex64 instead of complex128.
complex64 = bool(False)
# Maximum Fourier orders of the phase-coupling function.
Ms, Mt = int(20), int(5)
# Relative convergence tolerance.
rtol = 1e-5
# Maximum number of Bayesian updates.
max_iterations = 50

# Grid used for data resampling.
_x = L * np.linspace(-1., 1., 41)
_y = np.pi * np.linspace(-1., 1., 41)
# Samples retained per grid cell.
resample_in_grid = 250

# Number of GPU chunks used to limit memory consumption in einsum operations.
gpu_chunk = 50


# Estimate a diagonal phase-noise covariance matrix when NOISE_COV_ZERO=1.
noise_cov_zero = bool(int(os.environ.get("NOISE_COV_ZERO", 0)))

#* ---------------------------------------------------------




if __name__ == '__main__':

    # Redirect standard output to the workflow log.
    original_stdout = sys.stdout
    sys.stdout = open(stdoutname, 'a')

    try:
        # Load phase data sets with the same epsilon and sigma.

        if noise_cov_zero:
            print("Check: noise_cov_zero={:s}".format(str(noise_cov_zero)), flush=True)
            print("NOISE_COV_ZERO={:s}: estimating a diagonal phase-noise covariance matrix.".format(os.environ.get("NOISE_COV_ZERO", "0")), flush=True)
        else:
            print("Check: noise_cov_zero={:s}".format(str(noise_cov_zero)), flush=True)
            print("NOISE_COV_ZERO={:s}: estimating the full phase-noise covariance matrix.".format(os.environ.get("NOISE_COV_ZERO", "0")), flush=True)


        params = {"L": L, "complex64": complex64, "rtol": rtol, 
                  "gpu_chunk": gpu_chunk, "noise_cov_zero": noise_cov_zero}

        Phi1_list = []
        Phi2_list = []
        Theta1_list = []
        Theta2_list = []
        t1_list = []
        t2_list = []

        for phasefiledir in get_phase_file_list():
        # key: [Phi1, Phi2, Theta1, Theta2, t1, t2]
            print(f"Loading {phasefiledir} ...", flush=True)
            phase_data = np.load(phasefiledir)

            # Retain samples shared by t1 and t2.
            _Phi1 = phase_data['Phi1']
            _Phi2 = phase_data['Phi2']
            _Theta1 = phase_data['Theta1']
            _Theta2 = phase_data['Theta2']
            _t1 = phase_data['t1']
            _t2 = phase_data['t2']
            mask1, mask2 = get_common_time(_t1, _t2)
            Phi1_list.append(_Phi1[mask1])
            Phi2_list.append(_Phi2[mask2])
            Theta1_list.append(_Theta1[mask1])
            Theta2_list.append(_Theta2[mask2])
            t1_list.append(_t1[mask1])
            t2_list.append(_t2[mask2])

        phase_datas = {
            'Phi1': Phi1_list,
            'Phi2': Phi2_list,
            'Theta1': Theta1_list,
            'Theta2': Theta2_list,
            't': t1_list,
        }

        # Verify that the aligned time arrays match.
        for t1, t2 in zip(t1_list, t2_list):
            if not np.array_equal(t1, t2):
                print("Error: t1 and t2 do not match.")
        
        Delta_t = np.mean(np.diff(t1_list[0]))

        # Prepare samples for inference.
        print("Creating data for estimation...", flush=True)
        arrange_data_instance = ArrangeData()
        data_dict = arrange_data_instance.make_data(
            Phi1_list=Phi1_list,
            Phi2_list=Phi2_list,
            Theta1_list=Theta1_list,
            Theta2_list=Theta2_list,
            Delta_t=Delta_t,
        )


        xx, yy = np.meshgrid(_x, _y, indexing="xy")
        num = resample_in_grid * np.ones((xx.shape[0] - 1, xx.shape[1] - 1), dtype=int)

        # num[:, 0] = 0
        # num[:, -1] = 0

        thinned_data_dict = arrange_data_instance.thin_data(
                                            data=data_dict, xx=xx, yy=yy, num=num
                                        )


        # Infer the interaction from system 2 to system 1.
        print("Starting Bayesian Estimation (i,j)=(1,2)...", flush=True)
        estimator1 = Module_Bayes.BayesianEstimator(
            Ms=Ms, Mt=Mt, params=params
        )
        instance1, error1 = estimator1.map_method(
                                    Phi_i_ast=thinned_data_dict['Phi1_ast'],
                                    Phi_j_ast=thinned_data_dict['Phi2_ast'],
                                    Theta_i_ast=thinned_data_dict['Theta1_ast'],
                                    Theta_j_ast=thinned_data_dict['Theta2_ast'],
                                    dot_Phi_i=thinned_data_dict['dot_Phi1'],
                                    dot_Phi_j=thinned_data_dict['dot_Phi2'],
                                    dot_Theta_i=thinned_data_dict['dot_Theta1'],
                                    dot_Theta_j=thinned_data_dict['dot_Theta2'],
                                    Delta_t=Delta_t,
                                    num_iterations=max_iterations
                                  )
        error1 = np.array(error1)
        result1 = OutputFunction(instance1)



        # Infer the interaction from system 1 to system 2.
        print("Starting Bayesian Estimation (i,j)=(2,1)...", flush=True)
        estimator2 = Module_Bayes.BayesianEstimator(
            Ms=Ms, Mt=Mt, params=params
        )
        instance2, error2 = estimator2.map_method(
                                    Phi_i_ast=thinned_data_dict['Phi2_ast'],
                                    Phi_j_ast=thinned_data_dict['Phi1_ast'],
                                    Theta_i_ast=thinned_data_dict['Theta2_ast'],
                                    Theta_j_ast=thinned_data_dict['Theta1_ast'],
                                    dot_Phi_i=thinned_data_dict['dot_Phi2'],
                                    dot_Phi_j=thinned_data_dict['dot_Phi1'],
                                    dot_Theta_i=thinned_data_dict['dot_Theta2'],
                                    dot_Theta_j=thinned_data_dict['dot_Theta1'],
                                    Delta_t=Delta_t,
                                    num_iterations=max_iterations
                                    )
        error2 = np.array(error2)
        result2 = OutputFunction(instance2)



        # Evaluate the inferred phase equations.
        print("Outputting results...", flush=True)
        xx = np.linspace(-L, L, 2**10+1)
        yy = np.linspace(-np.pi, np.pi, 2**10+1)

        # Full phase equations.
        ZZ1_s_mean, ZZ1_t_mean, ZZ1_s_var, ZZ1_t_var =  result1.phase_equation(XX=xx, YY=yy, key="normal") 
        sqrt_E1 = result1.get_noise_covariance(colesky=True)
        a1, Sigma1 = result1.a, result1.Sigma

        ZZ2_s_mean, ZZ2_t_mean, ZZ2_s_var, ZZ2_t_var =  result2.phase_equation(XX=xx, YY=yy, key="normal")
        sqrt_E2 = result2.get_noise_covariance(colesky=True)
        a2, Sigma2 = result2.a, result2.Sigma


        # Constant terms.
        ZZ1_s_mean_const, ZZ1_t_mean_const, ZZ1_s_var_const, ZZ1_t_var_const =  result1.phase_equation(XX=xx, YY=yy, key="const") 
        ZZ2_s_mean_const, ZZ2_t_mean_const, ZZ2_s_var_const, ZZ2_t_var_const =  result2.phase_equation(XX=xx, YY=yy, key="const")

        # Nonconstant terms.
        ZZ1_s_mean_nonconst, ZZ1_t_mean_nonconst, ZZ1_s_var_nonconst, ZZ1_t_var_nonconst =  result1.phase_equation(XX=xx, YY=yy, key="nonconst") 
        ZZ2_s_mean_nonconst, ZZ2_t_mean_nonconst, ZZ2_s_var_nonconst, ZZ2_t_var_nonconst =  result2.phase_equation(XX=xx, YY=yy, key="nonconst")







        m_profile = np.stack(result1._m_profile, axis=0)  # Shape: (R, 2)

        result_dict = {'Delta_phi': xx, 
                 'Delta_theta': yy,
                 'ZZ1_s_mean': ZZ1_s_mean,
                 'ZZ1_t_mean': ZZ1_t_mean,
                 'ZZ1_s_var': ZZ1_s_var,
                 'ZZ1_t_var': ZZ1_t_var,
                 'sqrt_E1': sqrt_E1,
                 'ZZ2_s_mean': ZZ2_s_mean,
                 'ZZ2_t_mean': ZZ2_t_mean,
                 'ZZ2_s_var': ZZ2_s_var,
                 'ZZ2_t_var': ZZ2_t_var,
                 'sqrt_E2': sqrt_E2,
                 'param_a1': a1,
                 'param_Sigma1': Sigma1,
                 'param_a2': a2,
                 'param_Sigma2': Sigma2,
                 'm_profile': m_profile,
                # Fourier indices corresponding to rows in each parameter block.
                 'param.R': result1.R,
                 "error1": error1,  # Relative parameter-update errors
                "error2": error2,
                 }
        
        result_dict_const = copy.deepcopy(result_dict)
        result_dict_const.update({
            'ZZ1_s_mean': ZZ1_s_mean_const,
            'ZZ1_t_mean': ZZ1_t_mean_const,
            'ZZ1_s_var': ZZ1_s_var_const,
            'ZZ1_t_var': ZZ1_t_var_const,
            'ZZ2_s_mean': ZZ2_s_mean_const,
            'ZZ2_t_mean': ZZ2_t_mean_const,
            'ZZ2_s_var': ZZ2_s_var_const,
            'ZZ2_t_var': ZZ2_t_var_const,
        })

        result_dict_nonconst = copy.deepcopy(result_dict)
        result_dict_nonconst.update({
            'ZZ1_s_mean': ZZ1_s_mean_nonconst,
            'ZZ1_t_mean': ZZ1_t_mean_nonconst,
            'ZZ1_s_var': ZZ1_s_var_nonconst,
            'ZZ1_t_var': ZZ1_t_var_nonconst,
            'ZZ2_s_mean': ZZ2_s_mean_nonconst,
            'ZZ2_t_mean': ZZ2_t_mean_nonconst,
            'ZZ2_s_var': ZZ2_s_var_nonconst,
            'ZZ2_t_var': ZZ2_t_var_nonconst,
        })

        # Mark outputs that use a diagonal noise covariance matrix.
        if noise_cov_zero:
            filename5 = filename5.replace(".npz", "_covzero.npz")

        # Save results.
        save_npz_atomic(
            filename=filename5,
            base={},
            updates=result_dict,
            message=f"Saved Bayes results to {filename5}"
        )

        filename_const = filename5.replace(".npz", "_const.npz")
        save_npz_atomic(
            filename=filename_const,
            base={},
            updates=result_dict_const,
            message=f"Saved Bayes const-term results to {filename_const}"
        )

        filename_nonconst = filename5.replace(".npz", "_nonconst.npz")
        save_npz_atomic(
            filename=filename_nonconst,
            base={},
            updates=result_dict_nonconst,
            message=f"Saved Bayes non-const-term results to {filename_nonconst}"
        )


    finally:
        # Restore standard output.
        sys.stdout.close()
        sys.stdout = original_stdout
