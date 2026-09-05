#!/usr/bin/env python3
# coding: utf-8


"""Initialize intermediate data files."""

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
stdoutname = os.path.join(Module_get_envname.stdout_dir, "stdout_prep.txt") #* stdout



if __name__ == '__main__':

    pass

    # Redirect standard output to the workflow log.
    original_stdout = sys.stdout
    sys.stdout = open(stdoutname, 'a')
    try:
        # Create placeholder intermediate files.
        np.savez(filename2, dummy=np.zeros([3,3]))
        np.savez(filename3, dummy=np.zeros([3,3]))

        print("make", filename2)
        print("make", filename3) 

    finally:
        # Restore standard output.
        sys.stdout.close()
        sys.stdout = original_stdout
