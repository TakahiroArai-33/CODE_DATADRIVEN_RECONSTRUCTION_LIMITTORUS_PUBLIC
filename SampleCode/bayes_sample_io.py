"""Read the phase CSVs saved by Test.ipynb for Bayesian inference."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


PHASE_COLUMNS = ("t", "Phi1", "Theta1", "Phi2", "Theta2")
DATA_KEYS = (
    "Phi1_ast", "Phi2_ast", "Theta1_ast", "Theta2_ast",
    "dot_Phi1", "dot_Phi2", "dot_Theta1", "dot_Theta2",
)


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _saved_number(parameters, key, path, *, positive=False):
    value = parameters.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}: {key!r} must be a saved numerical parameter")
    value = float(value)
    if not np.isfinite(value) or (value <= 0 if positive else value < 0):
        condition = "positive" if positive else "nonnegative"
        raise ValueError(f"{path}: {key!r} must be finite and {condition}")
    return value


def _load_one(output_dir):
    phase_path = output_dir / "phase_comparison.csv"
    parameter_path = output_dir / "parameters.json"
    missing = [path.name for path in (phase_path, parameter_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {', '.join(missing)} in {output_dir}. "
            "Run Test.ipynb with loaddata = False through the final save cell, "
            "or set INPUT_DIRS to a directory containing its saved results."
        )
    try:
        parameters = json.loads(parameter_path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as error:
        raise ValueError(f"Cannot read saved parameters from {parameter_path}: {error}") from error
    if not isinstance(parameters, dict):
        raise ValueError(f"{parameter_path} must contain a JSON object")
    settings = {
        "epsilon": _saved_number(parameters, "epsilon", parameter_path),
        "sigma": _saved_number(parameters, "sigma", parameter_path),
        "domain_length": _saved_number(parameters, "domain_length", parameter_path, positive=True),
        "save_interval": _saved_number(parameters, "save_interval", parameter_path, positive=True),
    }
    try:
        with phase_path.open(encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream), [])
        if len(header) != len(set(header)):
            raise ValueError("duplicate CSV column names")
        missing_columns = set(PHASE_COLUMNS) - set(header)
        if missing_columns:
            raise ValueError(f"missing columns {sorted(missing_columns)}")
        values = np.loadtxt(
            phase_path, delimiter=",", skiprows=1,
            usecols=[header.index(name) for name in PHASE_COLUMNS], ndmin=2,
        )
    except (ValueError, OSError) as error:
        raise ValueError(f"Cannot read saved phases from {phase_path}: {error}") from error
    if values.shape[0] < 2 or values.shape[1] != 5 or not np.isfinite(values).all():
        raise ValueError(f"{phase_path} must contain at least two finite rows of phases")
    time = values[:, 0]
    intervals = np.diff(time)
    expected_dt = settings["save_interval"]
    # CSV time stamps can accumulate floating-point errors over long runs.
    if np.any(intervals <= 0) or not np.allclose(intervals, expected_dt, rtol=1e-8, atol=1e-9):
        raise ValueError(
            f"{phase_path}: times must be strictly increasing and uniformly spaced "
            f"by the saved save_interval = {expected_dt:g}"
        )
    periods = np.array([settings["domain_length"], 2 * np.pi] * 2)
    # Test.ipynb saves wrapped phases. Unwrap each trajectory separately before
    # differencing; wrapping a midpoint first creates false large velocities.
    unwrapped = np.column_stack([
        np.unwrap(values[:, component + 1], period=period)
        for component, period in enumerate(periods)
    ])
    midpoint = 0.5 * (unwrapped[:-1] + unwrapped[1:])
    velocity = np.diff(unwrapped, axis=0) / intervals[:, None]
    midpoint = np.mod(midpoint, periods)
    data = dict(zip(DATA_KEYS, (
        midpoint[:, 0], midpoint[:, 2], midpoint[:, 1], midpoint[:, 3],
        velocity[:, 0], velocity[:, 2], velocity[:, 1], velocity[:, 3],
    )))
    return data, settings, parameters


def load_phase_datasets(output_dirs, bins=16, max_per_cell=100, seed=415643):
    """Pool saved Test.ipynb phases and sample each periodic cell once.

    Accept one output directory or a list of directories with identical
    epsilon, sigma, domain_length and save_interval. Differences never cross
    trajectory boundaries. Return ``data`` (the estimator's eight arrays),
    ``Delta_t``, ``L``, ``epsilon``, ``sigma``, ``diagnostics`` and ``parameters``.
    No simulation or alternative input source is used.
    """
    if isinstance(output_dirs, (str, Path)):
        output_dirs = [output_dirs]
    directories = [Path(path).expanduser().resolve() for path in output_dirs]
    if not directories:
        raise ValueError("INPUT_DIRS must contain at least one saved output directory")
    if len(set(directories)) != len(directories):
        raise ValueError("INPUT_DIRS contains the same output directory more than once")
    n_bins = _positive_integer(bins, "bins")
    max_per_cell = _positive_integer(max_per_cell, "max_per_cell")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    all_data, all_parameters = [], []
    settings = None
    for directory in directories:
        data, current_settings, parameters = _load_one(directory)
        if settings is None:
            settings = current_settings
        else:
            for key, expected in settings.items():
                if not np.isclose(current_settings[key], expected, rtol=1e-10, atol=0.0):
                    raise ValueError(
                        f"{directory}: saved {key} = {current_settings[key]:g} differs "
                        f"from the first dataset ({expected:g}); fit these datasets separately"
                    )
        all_data.append(data)
        all_parameters.append(parameters)
    pooled = {key: np.concatenate([data[key] for data in all_data]) for key in DATA_KEYS}
    delta_phi = np.mod(pooled["Phi1_ast"] - pooled["Phi2_ast"], settings["domain_length"])
    delta_theta = np.mod(pooled["Theta1_ast"] - pooled["Theta2_ast"], 2 * np.pi)
    phi_bin = np.minimum((delta_phi / settings["domain_length"] * n_bins).astype(int), n_bins - 1)
    theta_bin = np.minimum((delta_theta / (2 * np.pi) * n_bins).astype(int), n_bins - 1)
    cell_ids = phi_bin + n_bins * theta_bin
    order = np.argsort(cell_ids, kind="stable")
    _, starts, counts = np.unique(cell_ids[order], return_index=True, return_counts=True)
    rng = np.random.default_rng(seed)
    selected = []
    for start, count in zip(starts, counts):
        members = order[start:start + count]
        selected.append(members if count <= max_per_cell else rng.choice(members, max_per_cell, replace=False))
    selected = np.sort(np.concatenate(selected))
    return {
        "data": {key: np.ascontiguousarray(values[selected]) for key, values in pooled.items()},
        "Delta_t": settings["save_interval"],
        "L": settings["domain_length"] / 2,
        "epsilon": settings["epsilon"],
        "sigma": settings["sigma"],
        "diagnostics": {
            "n_increments": len(cell_ids), "n_samples": len(selected),
            "occupied_cells": len(counts), "total_cells": n_bins * n_bins,
            "sources": [str(directory) for directory in directories],
        },
        "parameters": all_parameters,
    }
