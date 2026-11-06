"""Reference-state loading and initialization for the sample notebook."""

from pathlib import Path

import numpy as np


def integer_steps(duration, dt, label):
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(duration) or duration <= 0:
        raise ValueError(f"dt and {label} must be positive and finite")
    count = int(round(duration / dt))
    if count < 1 or not np.isclose(count * dt, duration, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{label} must be an integer multiple of dt={dt}")
    return count


def load_x0(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        if not path.name.endswith("_axes.csv"):
            raise ValueError("X0 CSV entry point must end in _axes.csv")
        stem = path.name[:-len("_axes.csv")]
        axes = np.genfromtxt(path, delimiter=",", names=True)
        if axes.dtype.names != ("t", "rx"):
            raise ValueError("X0 axes CSV must have header t,rx")
        return {"t": axes["t"], "rx": axes["rx"],
                "U": np.loadtxt(path.with_name(stem + "_U.csv"), delimiter=","),
                "V": np.loadtxt(path.with_name(stem + "_V.csv"), delimiter=",")}
    raise ValueError("X0 reference must be supplied as an _axes.csv file")


def initial_states_from_phases(path, phi1, theta1, phi2, theta2):
    """Build X0(phi1, theta1), X0(phi2, theta2) on the nearest periodic grid."""
    data = load_x0(path)
    rx, ts = np.asarray(data["rx"]), np.asarray(data["t"])
    u, v = np.asarray(data["U"]), np.asarray(data["V"])
    if rx.ndim != 1 or ts.ndim != 1 or rx.size < 4 or ts.size < 3:
        raise ValueError("X0 requires one-dimensional rx/t with periodic endpoints")
    if u.shape != (ts.size, rx.size) or v.shape != u.shape:
        raise ValueError("X0 U/V shape must be (len(t), len(rx))")
    if not all(np.all(np.isfinite(a)) for a in (rx, ts, u, v, phi1, theta1, phi2, theta2)):
        raise ValueError("X0 and initial phases must be finite")
    if np.any(np.diff(ts) <= 0) or np.any(np.diff(rx) <= 0):
        raise ValueError("X0 rx/t must increase strictly")
    if not np.allclose(np.diff(rx), rx[1] - rx[0]):
        raise ValueError("X0 requires a uniform spatial grid")
    if not np.allclose(u[:, 0], u[:, -1]) or not np.allclose(v[:, 0], v[:, -1]):
        raise ValueError("X0 spatial endpoint must duplicate the first point")
    length = float(np.ptp(rx))
    phi_grid = rx[:-1] - rx[0]
    theta_grid = 2 * np.pi * (ts[:-1] - ts[0]) / np.ptp(ts)
    def nearest(grid, target, period):
        return int(np.argmin(np.abs((grid - target + period / 2) % period - period / 2)))
    indices = [(nearest(phi_grid, phi, length), nearest(theta_grid, theta, 2 * np.pi))
               for phi, theta in ((phi1, theta1), (phi2, theta2))]
    initial = np.array([[np.roll(u[it, :-1], ix), np.roll(v[it, :-1], ix)]
                        for ix, it in indices], dtype=np.float64)
    (ix1, it1), (ix2, it2) = indices
    return initial, rx, {
        "phi1": float(phi_grid[ix1]), "theta1": float(theta_grid[it1]),
        "phi2": float(phi_grid[ix2]), "theta2": float(theta_grid[it2]),
        "delta_phi": float(phi_grid[ix1] - phi_grid[ix2]),
        "delta_theta": float(theta_grid[it1] - theta_grid[it2]),
        "lateral_index": ix1 - ix2, "vertical_index": it1 - it2,
        "lateral_index1": ix1, "vertical_index1": it1,
        "lateral_index2": ix2, "vertical_index2": it2,
        "reference_period": float(np.ptp(ts)),
    }
