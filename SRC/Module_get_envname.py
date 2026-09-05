#!/usr/bin/env python3
# coding: utf-8

import os
from pathlib import Path
import numpy as np

"""Expose workflow environment variables and derived output paths."""


def fmt_sci(val: float) -> str:
    """Format a value in normalized scientific notation."""
    return "0e+00" if val == 0 else np.format_float_scientific(val, precision=2, trim='-')


def fmt_dec(val: float, precision: int = 6) -> str:
    """Return a compact decimal representation for legacy filenames."""
    txt = f"{val:.{precision}f}".rstrip("0").rstrip(".")
    return txt or "0"


def dedup(seq):
    """
    Return a list with duplicate values removed, keeping the first occurrence and original order.

    Two items count as duplicates only if they are equal (e.g., same string/number);
    different strings like '0.001' and '1e-3' both remain because they differ.
    """
    seen = set()
    out = []
    for item in seq:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def build_candidates(base: str, eps_variants, sig_variants):
    return [f"{base}_eps{e}_sig{s}.npz" for e in eps_variants for s in sig_variants]


def pick_existing(dirpath, base: str, eps_variants, sig_variants) -> str:
    """Resolve filename-format variants, or return the normalized path."""
    dirpath = Path(dirpath)
    candidates = build_candidates(base, eps_variants, sig_variants)
    for name in candidates:
        path = dirpath / name
        if path.exists():
            return str(path)
    return str(dirpath / candidates[0])


# Output directories can be overridden through environment variables.
SPDE_DIR = os.environ.get("SPDE_DIR", "CoupledModelSimulationDATA")
CONCATENATE_DIR = os.environ.get("CONCATENATE_DIR", "ConcatenateDATA")
INTERIM_DIR = os.environ.get("INTERIM_DIR", "Results/Interim")
PHASE_DIR = os.environ.get("PHASE_DIR", "Results/Phase")
BAYES_DIR = os.environ.get("BAYES_DIR", "Results/BayesianInference")
STDOUT_DIR = os.environ.get("STDOUT_DIR", "Results/Logs")

USE_DEBUG_FILES = os.environ.get("USE_DEBUG_FILES", "0").lower() in ("1", "true", "yes") 

# Initial-state indices and physical parameters.
init_a_idx = int(os.environ.get("INIT_A_IDX", "358"))
init_b_idx = int(os.environ.get("INIT_B_IDX", "512"))
eps_raw = os.environ.get("EPS_VAL", "1e-6").strip()
sigma_raw = os.environ.get("SIGMA_VAL", "1e-7").strip()
eps_val = float(eps_raw)
sigma_val = float(sigma_raw)
SECTION = float(os.environ.get("SECTION", "7.0"))

# Accept scientific, decimal, and raw environment-variable filename variants.
eps_variants = dedup([fmt_sci(eps_val), fmt_dec(eps_val), eps_raw])
sigma_variants = dedup([fmt_sci(sigma_val), fmt_dec(sigma_val), sigma_raw])


if USE_DEBUG_FILES:
    base1 = f"debug_concatenated_state{init_a_idx}_{init_b_idx}"
    base2 = f"debug_state{init_a_idx}_{init_b_idx}"
    base3 = "debug_common_state"
    baseB = "debug_Bfunc"
    base4 = f"debug_phase_{init_a_idx}_{init_b_idx}"
    base5 = "debug_bayes"
else:
    base1 = f"concatenated_state{init_a_idx}_{init_b_idx}"
    base2 = f"state{init_a_idx}_{init_b_idx}"
    base3 = "common_state"
    baseB = "Bfunc"
    base4 = f"phase_{init_a_idx}_{init_b_idx}"
    base5 = "bayes"

filename1 = pick_existing(CONCATENATE_DIR, base1, eps_variants, sigma_variants)
filename2 = pick_existing(INTERIM_DIR, base2, eps_variants, sigma_variants)
filename3 = pick_existing(INTERIM_DIR, base3, eps_variants, sigma_variants)
filename_Bfunc = pick_existing(INTERIM_DIR, baseB, eps_variants, sigma_variants)
filename4 = pick_existing(PHASE_DIR, base4, eps_variants, sigma_variants)
filename5 = pick_existing(BAYES_DIR, base5, eps_variants, sigma_variants)
stdout_dir = STDOUT_DIR
