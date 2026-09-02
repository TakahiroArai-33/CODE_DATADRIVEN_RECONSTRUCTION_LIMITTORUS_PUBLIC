#!/usr/bin/env python3
# coding: utf-8


"""Calculate the b_i time series used to fit B."""

import numpy as np
import sys, os
import pickle
from numpy import linalg as LA
import gc, os, re, sys, time
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
stdoutname = os.path.join(Module_get_envname.stdout_dir, "stdout_cal_bi.txt") #* stdout


init_a_idx = Module_get_envname.init_a_idx
init_b_idx = Module_get_envname.init_b_idx
eps_val = Module_get_envname.eps_val
sigma_val = Module_get_envname.sigma_val


# Import workflow helpers.
import Module_calphase_toolbox
calculate_b = Module_calphase_toolbox.calculate_b
concatenate_theta_b = Module_calphase_toolbox.concatenate_theta_b
import Module_get_filelist  
get_concatfile_list = Module_get_filelist.get_concatfile_list
import Module_savefunc
save_npz_atomic = Module_savefunc.save_npz_atomic


if __name__ == '__main__':

    # Redirect standard output to the workflow log.
    original_stdout = sys.stdout
    sys.stdout = open(stdoutname, 'a', encoding='utf-8')

    print("Calculating bi time series...", flush=True)
    print("dataname(interim data):", filename2, flush=True)
    ts = time.time()

    try:
        indiv_interim_data = np.load(filename2)
        common_interim_data = np.load(filename3)


        # Read per-run intermediate data.
        mask1 = indiv_interim_data["mask1"]
        mask2 = indiv_interim_data["mask2"]
        Theta1 = indiv_interim_data["Theta1"]
        Theta2 = indiv_interim_data["Theta2"]
        # Apply phase-validity masks to the time and amplitude arrays.
        t1 = indiv_interim_data["t"][mask1]
        t2 = indiv_interim_data["t"][mask2]
        # eps_t1 = indiv_interim_data["eps_t"][mask1]
        # eps_t2 = indiv_interim_data["eps_t"][mask2]
        A1_tilde = indiv_interim_data["A1_tilde"][mask1]
        A2_tilde = indiv_interim_data["A2_tilde"][mask2]
        # U1_bar = indiv_interim_data["U1_bar"][mask1]
        # U2_bar = indiv_interim_data["U2_bar"][mask2]
        L = np.ptp(indiv_interim_data["rx"]) / 2.0  # Half the spatial domain length

        # Read shared intermediate data.
        c1 = common_interim_data["c1"]
        c2 = common_interim_data["c2"]
        
        
        # Calculate b_i(Theta).
        try:
            b_data1 = calculate_b(
                A_tilde=A1_tilde,
                Theta=Theta1,
                t=t1,
                params={"L": L, "c": c1},
            )
            flag1 = True

        except Exception as e:
            b_data1 = np.array(None)
            flag1 = False
            print(f"----------------------------", flush=True)
            print(f"Error occurred in calculate_b (b_data1):", flush=True)
            print(f"Error message: {e}", flush=True)
            print(f"Filename: {filename2}", flush=True)

        try:
            b_data2 = calculate_b(
                A_tilde=A2_tilde,
                Theta=Theta2,
                t=t2,
                params={"L": L, "c": c2},
            )   
            flag2 = True

        except Exception as e:
            b_data2 = np.array(None)
            flag2 = False
            print(f"----------------------------", flush=True)
            print(f"Error occurred in calculate_b (b_data2):", flush=True)
            print(f"Error message: {e}", flush=True)
            print(f"Filename: {filename2}", flush=True)


        # Concatenate successful b_i-versus-Theta results.
        if (flag1 is True) and (flag2 is True):
            bx1, by1 = concatenate_theta_b(b_data1)
            bx2, by2 = concatenate_theta_b(b_data2)

            # Report the mean and standard deviation across phase wraps.
            by1_means = np.array([b_data1[i]['y_mean'] for i in range(len(b_data1))])
            by2_means = np.array([b_data2[i]['y_mean'] for i in range(len(b_data2))])
            print(by1_means)
            aaa1 = np.mean(by1_means)
            bbb1 = np.std(by1_means)
            aaa2 = np.mean(by2_means)
            bbb2 = np.std(by2_means)
            print(f"[Calculate_bi] by1_mean: {aaa1}, by1_std: {bbb1}", flush=True)
            print(f"[Calculate_bi] by2_mean: {aaa2}, by2_std: {bbb2}", flush=True)
        
        if flag1 is False or flag2 is False:
            # Empty arrays cause this run to be skipped by Calculate_Bfunc.py.
            bx1 = np.array([])
            by1 = np.array([])
            bx2 = np.array([])
            by2 = np.array([])


        # Update the per-run intermediate data.
        add_params = {
            "bx1": bx1,
            "by1": by1,
            "bx2": bx2,
            "by2": by2,
        }

        target = dict(indiv_interim_data.items())
        indiv_interim_data.close()


        save_npz_atomic(
            filename=filename2,
            base=target,
            updates=add_params,
            message=f"Saved `b1` and `b2` to {filename2}"
            )


        # Save b_i-versus-Theta diagnostic plots in debug mode.
        if USE_DEBUG_FILES:
            import matplotlib.pyplot as plt



            os.makedirs("./Results/Diagnostics", exist_ok=True)


            fig = plt.figure(figsize=(7,3))
            
            ax = fig.add_subplot(1,2,1)
            ax.scatter(np.mod(bx1, 2*np.pi), 
                        by1, marker='o',
                        s=0.5)
            ax.set_title(r'$\{b^{(1)}_i (\Theta)\}_i$')
            ax.set_xlabel(r'$\Theta$')
            ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
            ax.set_xticklabels([r'$0$', r'$0.5\pi$', 
                                r'$\pi$', r'$1.5\pi$', r'$2\pi$'])

            bx = fig.add_subplot(1,2,2)
            bx.scatter(np.mod(bx2, 2*np.pi),
                        by2, marker='o',
                        s=0.5)
            bx.set_title(r'$\{b^{(2)}_i (\Theta)\}_i$')
            bx.set_xlabel(r'$\Theta$')
            bx.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
            bx.set_xticklabels([r'$0$', r'$0.5\pi$', 
                                r'$\pi$', r'$1.5\pi$', r'$2\pi$'])

            plt.tight_layout()
            fname = (
                f"./Results/Diagnostics/"
                f"fig_bi_plot_A{init_a_idx}_B{init_b_idx}_eps{eps_val}_sig{sigma_val}.png"
            )
            plt.savefig(fname, dpi=200)


    finally:
        te = time.time()
        print(f"Elapsed time: {te - ts:.2f} seconds.", flush=True)
        # Restore standard output.
        sys.stdout.close()
        sys.stdout = original_stdout
