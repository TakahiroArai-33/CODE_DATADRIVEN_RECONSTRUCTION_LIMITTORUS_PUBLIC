#!/usr/bin/env python3
# coding: utf-8

"""Generate short-trajectory SPDE JSON configurations."""

import json
import os, sys, shutil
from typing import List, Optional, Sequence, Tuple
import numpy as np

import make_configs


# Inherit the physical and grid parameters from the long-trajectory setup.
eps_list: List[float] = make_configs.eps_list
sigma_list: List[float] = make_configs.sigma_list
gridnum = make_configs.gridnum
dt = make_configs.dt
log_dt = make_configs.log_dt
b_grid = make_configs.b_grid

DEFAULT_T_END = 0.1


SIGMA_T_END_RULES: List[Tuple[float, float]] = [
    (2.0e-7, 0.1),
]


grid_delta_phi = make_configs.grid_delta_phi
grid_delta_theta = make_configs.grid_delta_theta


L = float(125)
# The 0.02L spacing resolves a 30th-order spatial Fourier series.
phi_value_list: List[float] = np.linspace(0.8*L, 1.2*L, num=21)[1:-1].tolist()
# The 2*pi/32 spacing resolves a 10th-order phase Fourier series.
theta_value_list: List[float] = np.linspace(0.0, 2.0*np.pi, num=b_grid).tolist()




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




def main():
    # Settings shared by every generated case.
    base_config = {
        "DT": dt,
        "GRIDNUM": gridnum,
        "C_VELOCITY": 0.0,
        "T_END": DEFAULT_T_END,
        "TRANSIENT": 20000,
        "REPEAT": 1,
        "SIGMA": 1.0e-7,
        "LOG_DT": log_dt,
    }

    phi_grid = grid_delta_phi
    theta_grid = grid_delta_theta


    out_dir = "json_short"
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
    path = "json_short"
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
