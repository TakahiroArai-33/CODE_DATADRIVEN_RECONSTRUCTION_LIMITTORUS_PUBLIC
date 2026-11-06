"""Load a complete saved notebook result without rerunning its calculations."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


TRAJECTORY_COLUMNS = (
    "t", "U1_bar", "V1_bar", "U2_bar", "V2_bar",
    "A1_tilde_real", "A1_tilde_imag", "A2_tilde_real", "A2_tilde_imag",
)
PHASE_COLUMNS = (
    "t", "eps_t", "delta_phi", "delta_theta", "theory_delta_phi",
    "theory_delta_theta", "Phi1", "Theta1", "Phi2", "Theta2",
)


def _header(path, required):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        header = next(csv.reader(stream), [])
    if not header or len(header) != len(set(header)):
        raise ValueError(f"Invalid or duplicate CSV column names in {path.name}")
    missing = set(required) - set(header)
    if missing:
        raise ValueError(f"Incomplete saved dataset: {path.name} is missing columns {sorted(missing)}")
    return header


def _numeric(path, required, header):
    try:
        values = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2,
                            usecols=[header.index(name) for name in required])
    except (ValueError, OSError) as exc:
        raise ValueError(f"Cannot load saved numerical data from {path.name}: {exc}") from exc
    if values.size == 0 or values.shape[1] != len(required):
        raise ValueError(f"No complete numerical rows in {path.name}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite saved values in {path.name}")
    return dict(zip(required, values.T))


def _increasing(values, label, minimum=2):
    if values.size < minimum or np.any(np.diff(values) <= 0):
        raise ValueError(f"{label} must contain at least {minimum} strictly increasing values")


def _parameters(path, header):
    parameters = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        next(reader)
        for row in reader:
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError(f"Invalid parameter row in {path.name}: {row!r}")
            name, value = row[header.index("name")], row[header.index("value")]
            if not name or name in parameters:
                raise ValueError(f"Empty or duplicate saved parameter {name!r}")
            parameters[name] = value
    return parameters


def _number(parameters, name, *, positive=False, nonnegative=False, integer=False):
    if name not in parameters:
        raise ValueError(f"Incomplete saved dataset: parameters.csv is missing {name!r}")
    try:
        value = float(parameters[name])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Saved parameter {name!r} must be numeric") from exc
    if (not np.isfinite(value) or (positive and value <= 0)
            or (nonnegative and value < 0) or (integer and value != int(value))):
        raise ValueError(f"Invalid saved parameter {name!r}: {parameters[name]!r}")
    return int(value) if integer else value


def load_precomputed_results(output_dir, fallback_b_theory=None):
    """Read saved data, phases, fitted B, and theory using the saved settings.

    All required files, column names, and parameter fields are checked before
    reading the large trajectory/phase tables. No simulation, fitting, phase
    extraction, or theory integration is performed. Older output folders may
    supply the reference B table via ``fallback_b_theory``.
    """
    output = Path(output_dir)
    specifications = {
        "trajectory.csv": TRAJECTORY_COLUMNS,
        "parameters.csv": ("name", "value"),
        "final_state.csv": ("rx", "U1", "V1", "U2", "V2"),
        "phase_comparison.csv": PHASE_COLUMNS,
        "B_estimated.csv": ("theta", "B1", "B2"),
    }
    missing = [name for name in specifications if not (output / name).is_file()]
    theory_path = output / "B_theory.csv"
    if not theory_path.is_file():
        theory_path = Path(fallback_b_theory) if fallback_b_theory is not None else theory_path
    if not theory_path.is_file():
        missing.append("B_theory.csv (or an existing fallback_b_theory)")
    if missing:
        raise FileNotFoundError(f"Incomplete saved dataset in {output}: missing {', '.join(missing)}")
    headers = {name: _header(output / name, columns)
               for name, columns in specifications.items()}
    theory_columns = ("theta", "B_theory")
    theory_header = _header(theory_path, theory_columns)
    coefficient_path = output / "B_coefficients.csv"
    coefficient_columns = ("n", "cos1", "sin1", "cos2", "sin2")
    coefficient_header = (_header(coefficient_path, coefficient_columns)
                          if coefficient_path.is_file() else None)
    parameters = _parameters(output / "parameters.csv", headers["parameters.csv"])
    restored = {
        "dt": _number(parameters, "dt", positive=True),
        "epsilon": _number(parameters, "epsilon", nonnegative=True),
        "sigma": _number(parameters, "sigma", nonnegative=True),
        "seed": _number(parameters, "seed", nonnegative=True, integer=True),
        "save_interval": _number(parameters, "save_interval", positive=True),
        "domain_length": _number(parameters, "domain_length", positive=True),
        "section": _number(parameters, "section"),
        "b_harmonics": _number(parameters, "b_harmonics", positive=True, integer=True),
        "c1": _number(parameters, "c1"),
        "c2": _number(parameters, "c2"),
    }
    if restored["seed"] >= 2**32:
        raise ValueError("Saved seed must be in [0, 2**32)")
    _number(parameters, "simulation_seconds", nonnegative=True)
    for name in ("Phi1_initial", "Theta1_initial", "Phi2_initial", "Theta2_initial"):
        _number(parameters, name)
    saved_t_end = _number(parameters, "t_end", nonnegative=True) if "t_end" in parameters else None
    # Restore and validate the smaller files before allocating large arrays.
    final = _numeric(output / "final_state.csv", specifications["final_state.csv"], headers["final_state.csv"])
    _increasing(final["rx"], "Saved spatial grid")
    ncore = final["rx"].size
    spacing = restored["domain_length"] / ncore
    if not np.allclose(np.diff(final["rx"]), spacing, rtol=1e-9, atol=1e-12):
        raise ValueError("Saved spatial grid does not match domain_length")
    rx = np.append(final["rx"], final["rx"][0] + restored["domain_length"])
    estimated = _numeric(output / "B_estimated.csv", specifications["B_estimated.csv"], headers["B_estimated.csv"])
    _increasing(estimated["theta"], "Saved B grid")
    b_theory = _numeric(theory_path, theory_columns, theory_header)
    _increasing(b_theory["theta"], "Saved theoretical B grid")
    fitted = [{"Bx": estimated["theta"], "By": estimated[f"B{i}"]} for i in (1, 2)]
    for i, values in enumerate(fitted, start=1):
        offset_key = f"b_offset{i}"
        if offset_key in parameters:
            values["offset"] = _number(parameters, offset_key)
    if coefficient_header is not None:
        coefficients = _numeric(coefficient_path, coefficient_columns, coefficient_header)
        if not np.array_equal(coefficients["n"], np.arange(1, coefficients["n"].size + 1)):
            raise ValueError("Saved B coefficient indices must be 1, 2, ..., N")
        for i, values in enumerate(fitted, start=1):
            values.update(alpha=coefficients[f"cos{i}"], beta=coefficients[f"sin{i}"])
    trajectory = _numeric(output / "trajectory.csv", TRAJECTORY_COLUMNS, headers["trajectory.csv"])
    _increasing(trajectory["t"], "Saved trajectory times")
    phase = _numeric(output / "phase_comparison.csv", PHASE_COLUMNS, headers["phase_comparison.csv"])
    _increasing(phase["t"], "Saved phase times")
    if not np.allclose(phase["eps_t"], restored["epsilon"] * phase["t"], rtol=1e-10, atol=1e-12):
        raise ValueError("Saved eps_t is inconsistent with epsilon and phase times")
    tolerance = 1e-10 * max(1.0, np.max(np.abs(trajectory["t"])))
    if (phase["t"][0] < trajectory["t"][0] - tolerance
            or phase["t"][-1] > trajectory["t"][-1] + tolerance):
        raise ValueError("Saved phase times extend outside the trajectory interval")
    moments = {name: trajectory[name] for name in ("t", "U1_bar", "V1_bar", "U2_bar", "V2_bar")}
    for i in (1, 2):
        moments[f"A{i}_tilde"] = trajectory[f"A{i}_tilde_real"] + 1j * trajectory[f"A{i}_tilde_imag"]
    moments["final_state"] = np.array([[final["U1"], final["V1"]], [final["U2"], final["V2"]]])
    return {
        **restored, "parameters": parameters, "moments": moments, "rx": rx,
        "L": restored["domain_length"] / 2,
        "t_end": saved_t_end if saved_t_end is not None else restored["epsilon"] * trajectory["t"][-1],
        "phase": phase, "B1": fitted[0], "B2": fitted[1], "b_theory": b_theory,
    }
