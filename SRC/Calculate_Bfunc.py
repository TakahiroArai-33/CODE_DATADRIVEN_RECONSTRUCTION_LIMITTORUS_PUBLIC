#!/usr/bin/env python3
# coding: utf-8


"""Fit the periodic B functions used to calculate Phi."""

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
stdoutname = os.path.join(Module_get_envname.stdout_dir, "stdout_cal_Bfunc.txt") #* stdout

eps_val = Module_get_envname.eps_val
sigma_val = Module_get_envname.sigma_val


# Import workflow helpers.
import Module_calphase_toolbox
fit_B_function = Module_calphase_toolbox.fit_B_function

import Module_get_filelist  
get_interim_file_list = Module_get_filelist.get_interim_file_list

import Module_savefunc
save_npz_atomic = Module_savefunc.save_npz_atomic



if __name__ == '__main__':

    # Redirect standard output to the workflow log.
    original_stdout = sys.stdout
    sys.stdout = open(stdoutname, 'a')
    try:


        # Load per-run intermediate data with the same epsilon and sigma.
        interim_data_list = []          
        for interimfiledir in get_interim_file_list():
        # key: [‘Theta1’, ‘Theta2’, ‘mask1’, ‘mask2’, ‘t’, ‘eps_t’, ‘U1_bar’, ‘U2_bar’, ‘V1_bar’, ‘V2_bar’, ‘A1_tilde’, ‘A2_tilde’, ‘rx’, ‘bx1’, ‘by1’, ‘bx2’, ‘by2’]
            print(f"Loading {interimfiledir} ...", flush=True)
            interim_data = np.load(interimfiledir)

            if len(interim_data["bx1"]) <= 0 or len(interim_data["bx2"]) <= 0:
                print(f"  Skipped due to empty bi data.", flush=True)
                continue
            
            interim_data_list.append({
                                    "bx1": interim_data["bx1"],  # Theta values for b_i(Theta)
                                    "by1": interim_data["by1"],  # b_i values
                                    "bx2": interim_data["bx2"],
                                    "by2": interim_data["by2"],
                                })
            del interim_data
            gc.collect()
            
        concat_func = lambda data, key : np.concatenate([d[key] for d in data])
        bx1_concat = concat_func(interim_data_list, "bx1")
        by1_concat = concat_func(interim_data_list, "by1")
        bx2_concat = concat_func(interim_data_list, "bx2")
        by2_concat = concat_func(interim_data_list, "by2")
        del interim_data_list
        gc.collect()

        # Estimate memory use before fitting.
        datasize1 = bx1_concat.nbytes  # bytes
        datasize2 = bx2_concat.nbytes  # bytes
        print(f"Data size for B1 function fitting: {datasize1/1e6} MB", flush=True)
        print(f"Data size for B2 function fitting: {datasize2/1e6} MB", flush=True)
        # Downsample arrays larger than 1 GB to approximately 500 MB.
        if datasize1 > (datasize1/1e6) >= 1000:
            s = int(np.ceil(datasize1 / 500e6))
            bx1_concat = bx1_concat[::s]
            by1_concat = by1_concat[::s]

        if datasize2 > (datasize2/1e6) >= 1000:
            s = int(np.ceil(datasize2 / 500e6))
            bx2_concat = bx2_concat[::s]
            by2_concat = by2_concat[::s]




        common_interim_data = np.load(filename3)


        # Fit periodic B functions.
        l2 = 0.0  # Regularization parameter
        M = 10  # Fourier order
        dict_B1 = fit_B_function(Theta=bx1_concat, B=by1_concat, M=M, l2=0.0)
        gc.collect()
        dict_B2 = fit_B_function(Theta=bx2_concat, B=by2_concat, M=M, l2=0.0)
        gc.collect()

        add_params = {   
            #* system1
            "Bx1": dict_B1["Bx"], "By1": dict_B1["By"],
            "alpha1": dict_B1["alpha"], "beta1": dict_B1["beta"], 
            "M1": dict_B1["M"], "offset1": dict_B1["offset"],
            #* system2
            "Bx2": dict_B2["Bx"], "By2": dict_B2["By"],
            "alpha2": dict_B2["alpha"], "beta2": dict_B2["beta"],
            "M2": dict_B2["M"], "offset2": dict_B2["offset"],
        }
        # B can be reconstructed by interpolating each (Bx, By) pair.
        print(common_interim_data.items())
        common_dict = dict(common_interim_data.items())
        common_interim_data.close()

        save_npz_atomic(
            filename=filename3,
            base=common_dict,
            updates=add_params,
            message="Save Bx, By, alpha, beta, M, offset to {filename3}".format(filename3=filename3)
        )

        if USE_DEBUG_FILES:
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(7, 3))
            ax = fig.add_subplot(1, 2, 1)
            ax.plot(dict_B1["Bx"], dict_B1["By"], color="red", label="fitted")
            ax.scatter(bx1_concat, by1_concat, s=0.5, alpha=1.0, color='royalblue', 
                        label="datas")
            ax.set_title("system 1")
            ax.set_xlabel(r"$\Theta$")
            ax.set_ylabel(r"$B^{(1)}(\Theta)$")
            ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
            ax.set_xticklabels([r'$0$', r'$0.5\pi$', 
                                r'$\pi$', r'$1.5\pi$', r'$2\pi$'])
            ax.legend(fontsize=10)

            bx = fig.add_subplot(1, 2, 2)
            bx.plot(dict_B2["Bx"], dict_B2["By"], color="red", label="fitted")
            bx.scatter(bx2_concat, by2_concat, s=0.5, alpha=1.0, color='royalblue', label="datas")
            bx.set_title("system 2")
            bx.set_xlabel(r"$\Theta$")
            bx.set_ylabel(r"$B^{(2)}(\Theta)$")
            bx.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
            bx.set_xticklabels([r'$0$', r'$0.5\pi$', 
                                r'$\pi$', r'$1.5\pi$', r'$2\pi$'])

            plt.tight_layout()
            os.makedirs("./Results/Diagnostics", exist_ok=True)

            fname = f"./Results/Diagnostics/fig_Bfunc_plot_eps{eps_val}_sig{sigma_val}.png"
            plt.savefig(fname, dpi=200)


    finally:
        # Restore standard output.
        sys.stdout.close()
        sys.stdout = original_stdout
