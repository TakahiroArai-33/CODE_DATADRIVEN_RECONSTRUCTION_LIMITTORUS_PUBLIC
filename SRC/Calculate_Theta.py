#!/usr/bin/env python3
# coding: utf-8


"""Calculate the Theta(t) time series for one simulation data set."""

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
stdoutname = os.path.join(Module_get_envname.stdout_dir, "stdout_cal_theta.txt") #* stdout


# Import phase-calculation helpers.
import Module_calphase_toolbox
find_crossing_time = Module_calphase_toolbox.find_crossing_time
calculate_Theta = Module_calphase_toolbox.calculate_Theta
calculate_Theta_Kralemann = Module_calphase_toolbox.calculate_Theta_Kralemann


# USE_KRALEMANN=1 selects the Kralemann method; 0 selects linear interpolation.
use_Kralemann_method = bool(int(os.environ.get("USE_KRALEMANN", 0)))



if __name__ == '__main__':

    # keys: U1_bar, U2_bar, V1_bar, V2_bar, A1_tilde, A2_tilde, t, eps_t, rx
    concat_data = np.load(filename1)
    indiv_interim_data = np.load(filename2)
    common_interim_data = np.load(filename3)

    # Redirect standard output to the workflow log.
    original_stdout = sys.stdout
    sys.stdout = open(stdoutname, 'a', encoding='utf-8')

    print("Calculating Theta(t) time series...", flush=True)
    print("dataname(interim data):", filename2, flush=True)
    ts = time.time()

    try:
        
        if use_Kralemann_method == False:
        
            print("USE_KRALEMANN={:s}: calculating Theta by linear interpolation.".format(str(os.environ.get("USE_KRALEMANN", 0))), flush=True)
            

            Theta1, mask1 = calculate_Theta(x=concat_data["U1_bar"],
                                            t=concat_data["t"], 
                                            return_mask=True)
            Theta2, mask2 = calculate_Theta(x=concat_data["U2_bar"],
                                            t=concat_data["t"], 
                                            return_mask=True)   
        
        elif use_Kralemann_method == True:

            print("USE_KRALEMANN={:s}: calculating Theta with the Kralemann method.".format(str(os.environ.get("USE_KRALEMANN", 0))), flush=True)

            Theta1, mask1 = calculate_Theta_Kralemann(x=concat_data["U1_bar"],
                                                    t=concat_data["t"])
            Theta2, mask2 = calculate_Theta_Kralemann(x=concat_data["U2_bar"],
                                                    t=concat_data["t"])
        

        # Update the per-run intermediate data.
        np.savez(filename2, 
                 Theta1=Theta1, 
                 Theta2=Theta2, 
                 mask1=mask1, 
                 mask2=mask2,
                 t=concat_data["t"],
                 eps_t=concat_data["eps_t"],
                 U1_bar=concat_data["U1_bar"],
                 U2_bar=concat_data["U2_bar"],
                 V1_bar=concat_data["V1_bar"],
                 V2_bar=concat_data["V2_bar"],
                 A1_tilde=concat_data["A1_tilde"],
                 A2_tilde=concat_data["A2_tilde"],
                 rx=concat_data["rx"],)
        
    finally:
        te = time.time()
        print(f"Elapsed time: {te - ts:.2f} seconds.", flush=True)
        
        # Restore standard output.
        sys.stdout.close()
        sys.stdout = original_stdout
