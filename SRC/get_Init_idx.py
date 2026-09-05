#!/usr/bin/env python3
"""
List INIT_A_IDX / INIT_B_IDX pairs by scanning SPDE_DIR outputs.

Extract the state-index pair from filenames such as
``debug_state358_512_eps0_sig0_process_0.npz`` and print each pair once.
"""


import os
import re
import Module_get_envname

USE_DEBUG_FILES = Module_get_envname.USE_DEBUG_FILES
SPDE_DIR = Module_get_envname.SPDE_DIR
EPS_VAL = Module_get_envname.eps_val
SIGMA_VAL = Module_get_envname.sigma_val

PREFIX = "debug_state" if USE_DEBUG_FILES else "state"
PATTERN = re.compile(
    rf"{PREFIX}(\d+)_(\d+)_eps([0-9.eE+-]+)_sig([0-9.eE+-]+)_process_\d+\.npz$"
)

def main() -> None:
    if not os.path.isdir(SPDE_DIR):
        raise SystemExit(f"Directory not found: {SPDE_DIR}")

    pairs = set()
    for name in os.listdir(SPDE_DIR):
        match = PATTERN.match(name)
        if not match:
            continue
        a_idx, b_idx, eps, sigma = match.groups()
        if float(eps) == EPS_VAL and float(sigma) == SIGMA_VAL:
            pairs.add((int(a_idx), int(b_idx)))

    for a_idx, b_idx in sorted(pairs):
        print(f"{a_idx} {b_idx}")

if __name__ == "__main__":
    main()
