#!/usr/bin/env python3
# coding: utf-8


"""Estimate the common translational speed c for each epsilon and sigma."""

import numpy as np
import sys, os
import pickle
from numpy import linalg as LA
import gc, os, re, sys
from collections import defaultdict
from typing import Tuple
from scipy.interpolate import interp1d
from functions.SpectralDecomposition import calculate_A
from functions.intersect_time import intersect_time
import numpy as np
from scipy.interpolate import PchipInterpolator



# Load workflow paths from environment variables.
import  Module_get_envname
SPDE_DIR = Module_get_envname.SPDE_DIR 
CONCATENATE_DIR = Module_get_envname.CONCATENATE_DIR
INTERIM_DIR = Module_get_envname.INTERIM_DIR
PHASE_DIR = Module_get_envname.PHASE_DIR
STDOUT_DIR = Module_get_envname.STDOUT_DIR
USE_DEBUG_FILES = Module_get_envname.USE_DEBUG_FILES

filename1 = Module_get_envname.filename1  # Concatenated SPDE data
filename2 = Module_get_envname.filename2  # Per-run intermediate data
filename3 = Module_get_envname.filename3  # Intermediate data shared by epsilon and sigma
filename4 = Module_get_envname.filename4  # Phase data
stdoutname = os.path.join(Module_get_envname.stdout_dir, "stdout_cal_c.txt") #* stdout


eps_val = Module_get_envname.eps_val
sigma_val = Module_get_envname.sigma_val


# Import workflow helpers.
import Module_calphase_toolbox
calculate_c = Module_calphase_toolbox.calculate_c
import Module_get_filelist  
get_concatfile_list = Module_get_filelist.get_concatfile_list
import Module_savefunc
save_npz_atomic = Module_savefunc.save_npz_atomic




if __name__ == '__main__':


    # Redirect standard output to the workflow log.
    original_stdout = sys.stdout
    sys.stdout = open(stdoutname, 'a')

    try:

        # Load concatenated data sets with the same epsilon and sigma.
        # keys: U1_bar, U2_bar, V1_bar, V2_bar, A1_tilde, A2_tilde, t, eps_t, rx

        concat_data_list = []
        for concatfiledir in get_concatfile_list():

            print(f"Loading {concatfiledir} ...", flush=True)

            #* key: [A1_tilde, A2_tilde, U1_bar, U2_bar, V1_bar, V2_bar, t, eps_t, rx]
            concat_data = np.load(concatfiledir)
            concat_data_list.append({"U1_bar": np.copy(concat_data["U1_bar"]),
                                    "U2_bar": np.copy(concat_data["U2_bar"]),
                                    "A1_tilde": np.copy(concat_data["A1_tilde"]),
                                    "A2_tilde": np.copy(concat_data["A2_tilde"]),
                                    "t": np.copy(concat_data["t"])})
            
            L = np.ptp(concat_data["rx"]) / 2.0  # Half the spatial domain length

            del concat_data
            gc.collect()

        # Pickle is required because the archive contains object arrays.
        common_interim_data = np.load(filename3, allow_pickle=True)

        params = {"L": L}

        # Estimate translational speeds.
        A_tilde1_list, A_tilde2_list = [], []
        U1_bar_list, U2_bar_list = [], []
        t_list = []
        for concat_data in concat_data_list:
            A_tilde1_list.append(concat_data["A1_tilde"])
            A_tilde2_list.append(concat_data["A2_tilde"])
            U1_bar_list.append(concat_data["U1_bar"])
            U2_bar_list.append(concat_data["U2_bar"])
            t_list.append(concat_data["t"])

        # Maximum-likelihood estimate for system 1.
        c_hat_1, sigma2_hat_1, residuals_1 = calculate_c(
                            A_tilde_list=A_tilde1_list,
                            x_list=U1_bar_list,
                            t_list=t_list,
                            params=params
                            )
        
        # Maximum-likelihood estimate for system 2.
        c_hat_2, sigma2_hat_2, residuals_2 = calculate_c(
                            A_tilde_list=A_tilde2_list,
                            x_list=U2_bar_list,
                            t_list=t_list,
                            params=params
                            )
        
        print("system1: ", flush=True)
        print(f"    → estimated c: {c_hat_1}, sigma^2: {sigma2_hat_1}", flush=True)
        print("Residual distribution:", flush=True)
        print("Mean: ", np.mean(np.concatenate(residuals_1)), flush=True)
        print("Variance: ", np.var(np.concatenate(residuals_1)), flush=True)

        print("system2: ", flush=True)
        print(f"    → estimated c: {c_hat_2}, sigma^2: {sigma2_hat_2}", flush=True)
        print("Residual distribution:", flush=True)
        print("Mean: ", np.mean(np.concatenate(residuals_2)), flush=True)
        print("Variance: ", np.var(np.concatenate(residuals_2)), flush=True)


        # Store estimates and residuals.
        params["c1"] = c_hat_1
        params["c2"] = c_hat_2
        params["sigma2_1"] = sigma2_hat_1
        params["sigma2_2"] = sigma2_hat_2
        params["residuals_1"] = np.concatenate(residuals_1)
        params["residuals_2"] = np.concatenate(residuals_2)

        if USE_DEBUG_FILES:
            # Report keys that will be overwritten.
            print("Keys in both common_interim_data and params:", 
                set(common_interim_data.keys()) & set(params.keys()), flush=True)
        
        # Merge estimates into the shared intermediate archive.
        common_dict = dict(common_interim_data.items())
        common_interim_data.close()

        save_npz_atomic(
            filename=filename3,
            base=common_dict,
            updates=params,
            message=f"Saved `c`, `sigma2`, and `residuals` to {filename3}"
        )


        if USE_DEBUG_FILES:
            # Save residual histograms.
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            xx = axes[0]
            yy = axes[1]
            
            # Use a histogram range of three standard deviations.
            title1 = (
                "$system1: \\hat{\\sigma}^{(1)}=" + f"{np.sqrt(sigma2_hat_1):.4e}"
                + ",\\ \\hat{{c}}^{{(1)}}=" + f"{c_hat_1:.4e}$\n"
                + "[hist] "
                + "$\\mathrm{{mean}}=" + f"{np.mean(params['residuals_1']):.3e}"
                + ",\\ \\mathrm{{sd}}=" + f"{np.std(params['residuals_1']):.3e}$"
            )
            xx.set_title(title1, fontsize=10)
    
            bins = np.linspace(-3*np.sqrt(sigma2_hat_1), 3*np.sqrt(sigma2_hat_1), 51)
            xx.hist(params["residuals_1"], bins=bins, alpha=1, label="system1")
            xx.axvline(0, color='k', linestyle='dashed', linewidth=1)

            title2 = (
                "$system2: \\hat{\\sigma}^{(2)}=" + f"{np.sqrt(sigma2_hat_2):.4e}"
                + ",\\ \\hat{{c}}^{{(2)}}=" + f"{c_hat_2:.4e}$\n"
                + "[hist] "
                + "$\\mathrm{{mean}}=" + f"{np.mean(params['residuals_2']):.3e}"
                + ",\\ \\mathrm{{sd}}=" + f"{np.std(params['residuals_2']):.3e}$"
            )
            yy.set_title(title2, fontsize=10)
            bins = np.linspace(-3*np.sqrt(sigma2_hat_2), 3*np.sqrt(sigma2_hat_2), 51)
            yy.hist(params["residuals_2"], bins=bins, alpha=1, label="system2")
            yy.axvline(0, color='k', linestyle='dashed', linewidth=1)

            xx.set_xlabel("residuals")
            xx.set_ylabel("num")
            yy.set_xlabel("residuals")
            yy.set_ylabel("num")

            plt.tight_layout()
            plt.savefig(f"./Results/Diagnostics/fig_residuals_hist_eps{eps_val}_sig{sigma_val}.png", dpi=200)
            plt.close()

    finally:
        # Restore standard output.
        sys.stdout.close()
        sys.stdout = original_stdout
