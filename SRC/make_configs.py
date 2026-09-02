#!/usr/bin/env python3
# coding: utf-8

"""Generate SPDE JSON configurations from parameter lists defined below."""

import json
import os, sys, shutil
from typing import List, Optional, Sequence, Tuple
import numpy as np


# Simulation parameters.
eps_list: List[float] = [1.0e-6]
# Alternative noise-strength sets retained for reproducibility:
# sigma_list: List[float] = [5.0e-7, 1.0e-6, 2.0e-6]
# sigma_list: List[float] = [1.0e-5, 5.0e-5, 1.0e-4]
# sigma_list: List[float] = [4.0e-6, 2.0e-5, 4.0e-5]
sigma_list: List[float] = [1.0e-6]

gridnum = int(2**10+1)
dt = float(0.01)
log_dt = float(1.0)
b_grid = int(2**5+1)

DEFAULT_T_END = 8.0
# Use the first duration whose threshold is no greater than sigma.


SIGMA_T_END_RULES: List[Tuple[float, float]] = [
    (1.0e-6, 10.0),
    (4.0e-7, 9.0),
    (2.0e-7, 8.0),
]


# Requested initial values are snapped to these grids.
grid_delta_phi = np.linspace(0.0, 250.0, 2**10 + 1)
grid_delta_theta = np.linspace(0.0, 2 * np.pi, 2**10 + 1)


L = float(125)
phi_value_list: List[float] = [
    0.8 * L,
    1.2 * L,
]

theta_indices = np.linspace(0, 2**10 + 1, num=b_grid, dtype=np.int32)[:-1]
theta_value_list: List[float] = [
    float(grid_delta_theta[i]) for i in theta_indices
]



GridSelection = Tuple[int, float, Optional[float]]  # (index, snapped value, original target)


def resolve_t_end(sigma: float) -> float:
    """Return the appropriate T_END for the given noise strength."""
    for threshold, t_end in sorted(SIGMA_T_END_RULES, key=lambda x: x[0], reverse=True):
        if sigma >= threshold:
            return float(t_end)
    return float(DEFAULT_T_END)


def snap_to_grid(grid: np.ndarray, target: float, name: str) -> Tuple[int, float]:
    """Return nearest grid index/value to ``target`` and log the snapping."""
    flat = np.asarray(grid).ravel()
    idx = int(np.abs(flat - target).argmin())
    snapped = float(flat[idx])
    delta = snapped - float(target)
    if not np.isclose(delta, 0.0):
        print(f"[snap] {name}: target {target:.6g} -> grid[{idx}]={snapped:.6g} (diff {delta:+.3g})")
    else:
        print(f"[snap] {name}: target {target:.6g} matches grid[{idx}]")
    return idx, snapped


def prepare_grid_values(grid: Optional[np.ndarray],
                        values: Sequence[float],
                        name: str) -> List[GridSelection]:
    """Snap requested ``values`` to the nearest grid points."""
    if grid is None:
        raise RuntimeError(f"{name} value(s) specified but grid data is unavailable")

    value_list = list(values)
    if not value_list:
        raise ValueError(f"At least one {name} value must be provided")

    selections: List[GridSelection] = []
    for value in value_list:
        idx, snapped = snap_to_grid(grid, float(value), name)
        selections.append((idx, snapped, float(value)))
    return selections


def write_config(eps: float, phi: float, theta: float, sigma: float, idx: int,
                 base_config: dict, out_dir: str, label_prefix: str = "run",
                 custom_label: Optional[str] = None,
                 t_end: Optional[float] = None) -> str:
    """Create one JSON config file and return its path.

    - Links label and stdout automatically.
    - Does not create directories; assume caller prepared them.
    """
    label = custom_label if custom_label is not None \
        else f"{label_prefix}_eps{eps:.6g}_phi{phi:.6g}_theta{theta:.6g}_{idx}"
    cfg = dict(base_config)
    cfg["EPSILON"] = float(eps)
    cfg["INIT_CONDITION"] = [float(phi), float(theta)]
    cfg["label"] = label
    cfg["stdout"] = f"stdout/{label}.txt"
    cfg["SIGMA"] = float(sigma)
    if t_end is not None:
        cfg["T_END"] = float(t_end)
    if "LOG_DT" not in cfg:
        raise KeyError("base_config must define LOG_DT")

    out_path = os.path.join(out_dir, f"{label}.json")
    with open(out_path, "w") as w:
        json.dump(cfg, w, indent=2, ensure_ascii=False)
    return out_path




def make_debug_config(delta_phi_value: Optional[float] = None,
                      delta_theta_value: Optional[float] = None,
                      eps: float = 1.0e-6, sigma: float = 1.0e-7,
                      out_dir: str = "json", label_prefix: str = "debug") -> str:
    """
    Debug helper: mimic main() but generate exactly one JSON using
    the specified indices (a_index for delta_phi, b_index for delta_theta)
    and parameters (eps, sigma).

    Returns the written file path.
    """
    base_config = {
        "DT": dt,
        "GRIDNUM": gridnum,
        "C_VELOCITY": 0.0,
        "T_END": 1.0,
        "TRANSIENT": 20000,
        "REPEAT": 50,
        "SIGMA": 1.0e-7,
        "LOG_DT": log_dt,
    }

    phi_grid = grid_delta_phi
    theta_grid = grid_delta_theta

    target_phi = float(delta_phi_value if delta_phi_value is not None else phi_value_list[0])
    target_theta = float(delta_theta_value if delta_theta_value is not None else theta_value_list[0])

    a, delta_phi = snap_to_grid(phi_grid, target_phi, "debug delta_phi")
    b, delta_theta = snap_to_grid(theta_grid, target_theta, "debug delta_theta")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("stdout", exist_ok=True)

    label = f"debug_state{a}_{b}_eps{eps:.6g}_sig{sigma:.3g}"
    path = write_config(
        eps=eps, phi=delta_phi, theta=delta_theta, sigma=sigma, idx=1,
        base_config=base_config, out_dir=out_dir,
        label_prefix=label_prefix, custom_label=label
    )
    print(f"[debug] wrote {path}")
    return path



def main():
    # Settings shared by every generated case.
    base_config = {
        "DT": dt,
        "GRIDNUM": gridnum,
        "C_VELOCITY": 0.0,
        "T_END": DEFAULT_T_END,
        "TRANSIENT": 20000,
        "REPEAT": 50,
        "SIGMA": 1.0e-7,
        "LOG_DT": log_dt,
    }

    phi_grid = grid_delta_phi
    theta_grid = grid_delta_theta


    out_dir = "json"
    label_prefix = "run"

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("stdout", exist_ok=True)

    try:
        phi_selections = prepare_grid_values(phi_grid, phi_value_list, "delta_phi")
        theta_selections = prepare_grid_values(theta_grid, theta_value_list, "delta_theta")
    except (RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    idx = 0
    for eps in eps_list:
        for sigma in sigma_list:
            t_end = resolve_t_end(sigma)
            for phi_idx, delta_phi, _ in phi_selections:
                for theta_idx, delta_theta, _ in theta_selections:
                    idx += 1
                    state_label = f"state{phi_idx}_{theta_idx}_eps{eps:.6g}_sig{sigma:.3g}"
                    write_config(
                        eps=eps, phi=delta_phi, theta=delta_theta, sigma=sigma, idx=idx,
                        base_config=base_config, out_dir=out_dir,
                        label_prefix=label_prefix, custom_label=state_label,
                        t_end=t_end,
                    )

    print(f"Generated {idx} config(s) in '{out_dir}'.")


def remove_json_dir(strict=True):
    path = "json"
    if os.path.isdir(path):
        try:
            shutil.rmtree(path)
            print("Removed json/")
        except Exception as e:
            print(f"[WARN] failed to remove {path}/: {e}")
    else:
        if strict:
            try:
                raise FileNotFoundError(f"{path}/ does not exist")
            except FileNotFoundError as e:
                # Warn but continue execution
                print(f"[WARN] {e}")
        else:
            print(f"{path}/ not found; skip")



if __name__ == "__main__":
    remove_json_dir()
    main()
