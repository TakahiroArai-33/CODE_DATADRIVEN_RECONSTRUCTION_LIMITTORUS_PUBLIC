#!/usr/bin/env python3
# coding: utf-8


"""Calculate the spatial phase coordinate Phi."""

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
stdoutname = os.path.join(Module_get_envname.stdout_dir, "stdout_cal_Phi.txt") #* stdout


init_a_idx = Module_get_envname.init_a_idx
init_b_idx = Module_get_envname.init_b_idx
eps_val = Module_get_envname.eps_val
sigma_val = Module_get_envname.sigma_val


# Import workflow helpers.
import Module_calphase_toolbox
calculate_Phi = Module_calphase_toolbox.calculate_Phi
import Module_get_filelist  
get_concatfile_list = Module_get_filelist.get_concatfile_list
import Module_savefunc
save_npz_atomic = Module_savefunc.save_npz_atomic


if __name__ == '__main__':

    # Redirect standard output to the workflow log.
    original_stdout = sys.stdout
    sys.stdout = open(stdoutname, 'a', encoding='utf-8')

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
        Bx1 = common_interim_data["Bx1"]
        By1 = common_interim_data["By1"]
        Bx2 = common_interim_data["Bx2"]
        By2 = common_interim_data["By2"]
        
        params = {"L": L}
        B1func = interp1d(Bx1, By1, kind="linear")
        B2func = interp1d(Bx2, By2, kind="linear")

        # Calculate Phi.
        Phi1 = calculate_Phi(A_tilde=A1_tilde, Theta=Theta1,
                             Bfunc=B1func, params=params)
        Phi2 = calculate_Phi(A_tilde=A2_tilde, Theta=Theta2,
                             Bfunc=B2func, params=params)

        # Update the per-run intermediate data.
        add_params = {
            "Phi1": Phi1,
            "Phi2": Phi2,
        }

        target = dict(indiv_interim_data.items())
        indiv_interim_data.close()

        save_npz_atomic(
            filename=filename2,
            base=target,
            updates=add_params,
            message=f"Saved `Phi1` and `Phi2` to {filename2}"
            )
        

        # Save the completed phase data.
        phase_params = {
            "t1": t1,
            "t2": t2,
            "Phi1": Phi1,
            "Phi2": Phi2,
            "Theta1": Theta1,
            "Theta2": Theta2,
        }

        np.savez(filename4, **phase_params)
        print(f"Saved phase data to {filename4}")


    finally:
        # Restore standard output.
        sys.stdout.close()
        sys.stdout = original_stdout
