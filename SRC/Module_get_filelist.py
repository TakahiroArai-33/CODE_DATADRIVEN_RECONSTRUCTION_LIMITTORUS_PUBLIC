import os
import re

import Module_get_envname

SPDE_DIR = Module_get_envname.SPDE_DIR 
CONCATENATE_DIR = Module_get_envname.CONCATENATE_DIR
INTERIM_DIR = Module_get_envname.INTERIM_DIR
PHASE_DIR = Module_get_envname.PHASE_DIR
BAYES_DIR = Module_get_envname.BAYES_DIR
STDOUT_DIR = Module_get_envname.STDOUT_DIR
filename1 = Module_get_envname.filename1
filename2 = Module_get_envname.filename2
filename3 = Module_get_envname.filename3
filename4 = Module_get_envname.filename4
filename5 = Module_get_envname.filename5


USE_DEBUG_FILES = Module_get_envname.USE_DEBUG_FILES

init_a_idx = Module_get_envname.init_a_idx
init_b_idx = Module_get_envname.init_b_idx
eps_val = Module_get_envname.eps_val
sigma_val = Module_get_envname.sigma_val
SECTION = Module_get_envname.SECTION

# Extract epsilon and sigma from generated filenames.
PARAM_PATTERN = re.compile(r"_eps([0-9.eE+-]+)_sig([0-9.eE+-]+)(?=\.npz$)")

def get_concatfile_list(sigma: float = sigma_val, epsilon: float = eps_val) -> list:
    """Return concatenated NPZ paths matching ``sigma`` and ``epsilon``."""

    target_prefix = "debug_concatenated_state" if USE_DEBUG_FILES else "concatenated_state"
    filelist = []

    for name in os.listdir(CONCATENATE_DIR):
        if not (name.startswith(target_prefix) and name.endswith(".npz")):
            continue

        match = PARAM_PATTERN.search(name)
        if not match:
            continue

        matched_eps = float(match.group(1))
        matched_sigma = float(match.group(2))

        if matched_sigma == sigma and matched_eps == epsilon:
            full_path = os.path.join(CONCATENATE_DIR, name)
            relative_path = os.path.relpath(full_path, start=os.getcwd())
            filelist.append(relative_path)

    return filelist



def get_interim_file_list(sigma: float = sigma_val, epsilon: float = eps_val) -> list:
    """Return per-trajectory interim NPZ paths matching the parameters."""

    target_prefix = "debug_state" if USE_DEBUG_FILES else "state" 
    filelist = []

    for name in os.listdir(INTERIM_DIR):
        if not (name.startswith(target_prefix) and name.endswith(".npz")):
            continue

        match = PARAM_PATTERN.search(name)
        if not match:
            continue

        matched_eps = float(match.group(1))
        matched_sigma = float(match.group(2))

        if matched_sigma == sigma and matched_eps == epsilon:
            full_path = os.path.join(INTERIM_DIR, name)
            relative_path = os.path.relpath(full_path, start=os.getcwd())
            filelist.append(relative_path)

    return filelist



def get_phase_file_list(sigma: float = sigma_val, epsilon: float = eps_val) -> list:
    """Return phase-data NPZ paths matching ``sigma`` and ``epsilon``."""

    target_prefix = "debug_phase" if USE_DEBUG_FILES else "phase" 
    filelist = []

    for name in os.listdir(PHASE_DIR):
        if not (name.startswith(target_prefix) and name.endswith(".npz")):
            continue

        match = PARAM_PATTERN.search(name)
        if not match:
            continue

        matched_eps = float(match.group(1))
        matched_sigma = float(match.group(2))

        if matched_sigma == sigma and matched_eps == epsilon:
            full_path = os.path.join(PHASE_DIR, name)
            relative_path = os.path.relpath(full_path, start=os.getcwd())
            filelist.append(relative_path)

    return filelist
