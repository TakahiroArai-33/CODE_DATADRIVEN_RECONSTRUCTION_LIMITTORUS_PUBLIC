"""Phases and B estimates from one realization of two coupled Gray--Scott systems.

Each system's c and zero-mean B are estimated from its time series within
this same joint trajectory. No ensemble of trajectories is needed.
The spatial period is domain_length=2L; temporal phases have period 2*pi.
Phase differences use system 1 minus system 2.

Adapted from Module_calphase_toolbox.py's calculate_Theta, calculate_c,
calculate_b, fit_B_function and calculate_Phi, and SRC_論文用/Plot_Phase.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RectBivariateSpline, interp1d


def wrap_period(value, period: float):
    """Wrap to [-period/2, period/2), including the negative boundary."""
    return (np.asarray(value) + period / 2.0) % period - period / 2.0


def _vector(value, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite one-dimensional array")
    return array


def _real_vector(value, name: str) -> np.ndarray:
    array = _vector(value, name)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real-valued")
    return array.astype(float)


def find_crossing_time(x, t, section=7.0):
    """Return upward section crossings using the original GPU pipeline rule."""
    mask = (x[:-1] <= section) & (section <= x[1:])
    t1, t2 = t[:-1][mask], t[1:][mask]
    x1, x2 = x[:-1][mask], x[1:][mask]
    fraction = np.abs(section - x1) / np.abs(x2 - x1)
    return (1.0 - fraction) * t1 + fraction * t2


def calculate_Theta(x, t, section=7.0):
    """Return wrapped temporal phase and the corresponding time mask."""
    crossings = find_crossing_time(x, t, section)
    theta_interp = interp1d(crossings, 2.0 * np.pi * np.arange(len(crossings)), kind="linear")
    mask = (t >= crossings[0]) & (t < crossings[-1])
    return np.mod(theta_interp(t[mask]).astype(float), 2.0 * np.pi), mask


def calculate_c(A_tilde, x, t, params, section=7.0):
    """Estimate the translation speed from one trajectory."""
    L = params["L"]
    crossings = find_crossing_time(x, t, section)
    argument = np.unwrap(np.angle(A_tilde))
    argument_interp = interp1d(t, argument, kind="linear")
    delta_argument = np.diff(argument_interp(crossings))
    delta_t = np.diff(crossings)
    increments = (L / np.pi) * delta_argument
    return float(np.sum(delta_t * increments) / np.sum(delta_t**2))


def calculate_b(A_tilde, Theta, t, params):
    """Return per-cycle b samples, following the original phase-grid integral."""
    L, c = params["L"], params["c"]
    argument = np.unwrap(np.angle(A_tilde))
    y = (L / np.pi) * argument - c * t
    theta_unwrapped = np.unwrap(Theta)
    lap_idx = np.floor((theta_unwrapped - theta_unwrapped[0]) / (2.0 * np.pi)).astype(int)
    laps = np.arange(lap_idx.min(), lap_idx.max() + 1)
    core_laps = laps[1:-1]
    y_interp = interp1d(theta_unwrapped, y, kind="linear")
    grid = np.linspace(0.0, 2.0 * np.pi, 21)
    lap_edges = np.concatenate(([0], np.flatnonzero(np.diff(lap_idx)) + 1, [lap_idx.size]))
    lap_slices = {lap: slice(lap_edges[j], lap_edges[j + 1]) for j, lap in enumerate(laps)}

    out = []
    for lap in core_laps:
        span = lap_slices[lap]
        n = theta_unwrapped[span.start] // (2.0 * np.pi)
        xp = grid + 2.0 * np.pi * n
        y_mean = np.trapz(y_interp(xp), xp) / (2.0 * np.pi)
        out.append({
            "lap": int(lap), "t": t[span],
            "theta": np.mod(theta_unwrapped[span], 2.0 * np.pi),
            "b": y[span] - y_mean, "y_mean": float(y_mean),
        })
    return out


def concatenate_theta_b(out):
    """Concatenate the per-cycle theta and b samples."""
    theta = np.concatenate([entry["theta"] for entry in out]).astype(float)
    b = np.concatenate([entry["b"] for entry in out]).astype(float)
    return theta, b


def fit_B_function(Theta, B, M=10, grid_points=4097):
    """Fit the original Fourier model and subtract its uniform phase mean."""
    theta = np.mod(np.asarray(Theta, float), 2.0 * np.pi)
    samples = np.asarray(B, float)
    cos_block = np.cos(np.outer(theta, np.arange(1, M + 1)))
    sin_block = np.sin(np.outer(theta, np.arange(1, M + 1)))
    design = np.column_stack([cos_block, sin_block]).astype(float)
    coefficients, *_ = np.linalg.lstsq(design, samples, rcond=None)
    alpha, beta = coefficients[:M], coefficients[M:]

    def fourier(theta):
        theta = np.asarray(theta, float)
        values = np.zeros_like(theta, dtype=float)
        for m in range(1, M + 1):
            values += alpha[m - 1] * np.cos(m * theta) + beta[m - 1] * np.sin(m * theta)
        return values

    grid = np.linspace(0.0, 2.0 * np.pi, 4097)
    offset = np.trapz(fourier(grid), grid) / (2.0 * np.pi)
    bx = np.linspace(0.0, 2.0 * np.pi, grid_points)
    by = fourier(bx) - offset
    return {"Bx": bx, "By": by, "alpha": alpha, "beta": beta, "M": M, "offset": float(offset)}


def calculate_Phi(A_tilde, Theta, Bfunc, params):
    """Return the spatial phase wrapped to [-L, L)."""
    L = params["L"]
    argument = np.mod(np.angle(A_tilde), 2.0 * np.pi)
    phi = np.mod((L / np.pi) * argument - Bfunc(Theta), 2.0 * L)
    phi[phi >= L] -= 2.0 * L
    return phi


def get_common_time(t1, t2):
    """Return masks selecting the shared interval of two matching time grids."""
    start = np.max([t1.min(), t2.min()])
    end = np.min([t1.max(), t2.max()])
    return (start <= t1) & (t1 <= end), (start <= t2) & (t2 <= end)


def _load_coupling(path: Path, domain_length: float):
    """Load mutual coupling from NPZ or axes/matrix CSV files.

    CSV layout: PREFIX_axes.csv has delta_theta,delta_phi columns;
    PREFIX_Gamma_s.csv and PREFIX_Gamma_t.csv are dense numeric matrices
    without headers, with rows=delta_theta and columns=delta_phi.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "MutualCouplingFunction_8193_axes.csv"
    if path.suffix.lower() == ".csv":
        if not path.name.endswith("_axes.csv"):
            raise ValueError("Specify the coupling *_axes.csv file")
        axes = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding="utf-8-sig")
        if not axes.dtype.names or not {"delta_theta", "delta_phi"}.issubset(axes.dtype.names):
            raise ValueError("Coupling axes CSV needs delta_theta,delta_phi columns")
        theta = _real_vector(np.atleast_1d(axes["delta_theta"]), "coupling delta_theta")
        phi = _real_vector(np.atleast_1d(axes["delta_phi"]), "coupling delta_phi")
        prefix = path.name.removesuffix("_axes.csv")
        fields = [np.loadtxt(path.with_name(f"{prefix}_{key}.csv"), delimiter=",", ndmin=2)
                  for key in ("Gamma_s", "Gamma_t")]
    else:
        with np.load(path, allow_pickle=False) as data:
            phi = _real_vector(data["delta_phi"], "coupling delta_phi")
            theta = _real_vector(data["delta_theta"], "coupling delta_theta")
            fields = [np.asarray(data[key], dtype=float) for key in ("Gamma_s", "Gamma_t")]
    if phi.size < 4 or theta.size < 4:
        raise ValueError("Cubic coupling interpolation requires at least four points per axis")
    for field in fields:
        if field.shape != (theta.size, phi.size) or not np.all(np.isfinite(field)):
            raise ValueError("Coupling fields must be finite with axes (delta_theta, delta_phi)")
    if np.all(np.diff(phi) < 0):
        phi = phi[::-1]
        fields = [field[:, ::-1] for field in fields]
    if np.all(np.diff(theta) < 0):
        theta = theta[::-1]
        fields = [field[::-1, :] for field in fields]
    if np.any(np.diff(phi) <= 0) or np.any(np.diff(theta) <= 0):
        raise ValueError("Coupling axes must be strictly monotonic")
    if not np.isclose(np.ptp(phi), domain_length, rtol=1e-9, atol=1e-10):
        raise ValueError("Coupling spatial period does not match domain_length")
    if not np.isclose(np.ptp(theta), 2 * np.pi, rtol=1e-9, atol=1e-10):
        raise ValueError("Coupling temporal grid must cover one period of 2*pi")
    for field in fields:
        if not (np.allclose(field[0], field[-1], rtol=1e-9, atol=1e-10)
                and np.allclose(field[:, 0], field[:, -1], rtol=1e-9, atol=1e-10)):
            raise ValueError("Coupling field endpoints must be periodic")
    interpolators = [RectBivariateSpline(theta, phi, field, kx=3, ky=3, s=0) for field in fields]
    return phi[0], theta[0], interpolators


def solve_theory(
    t: np.ndarray,
    initial_delta,
    coupling_path: Path,
    epsilon: float,
    domain_length: float = 250.0,
) -> np.ndarray:
    """Return theoretical (delta_phi,delta_theta), shape (2,N), unwrapped.

    The mutual-coupling table already represents the difference of the two
    single-system coupling terms. Integrate dDelta/dtau=Gamma(Delta), with
    tau=epsilon*(t-t[0]), starting at the first observed phase difference.
    This deterministic averaged ODE is a reference drift, not a prediction
    of the individual noise realization when sigma is nonzero.
    """
    t = _real_vector(t, "theory t")
    initial = np.asarray(initial_delta)
    if t.size == 0 or np.any(np.diff(t) <= 0):
        raise ValueError("theory t must be nonempty and strictly increasing")
    if initial.shape != (2,) or np.iscomplexobj(initial) or not np.all(np.isfinite(initial)):
        raise ValueError("initial_delta must contain two finite real values")
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    if not np.isfinite(domain_length) or domain_length <= 0:
        raise ValueError("domain_length must be finite and positive")
    initial = initial.astype(float)
    if epsilon == 0 or t.size == 1:
        return np.repeat(initial[:, None], t.size, axis=1)
    phi_start, theta_start, interpolators = _load_coupling(coupling_path, domain_length)

    def rhs(_tau, state):
        phi = (state[0] - phi_start) % domain_length + phi_start
        theta = (state[1] - theta_start) % (2 * np.pi) + theta_start
        return np.array([float(interpolator.ev(theta, phi)) for interpolator in interpolators])

    tau = epsilon * (t - t[0])
    solution = solve_ivp(rhs, (0., tau[-1]), initial, t_eval=tau, rtol=1e-8, atol=1e-9)
    if not solution.success or solution.y.shape != (2, t.size):
        raise RuntimeError(f"Theoretical phase ODE failed: {solution.message}")
    return solution.y
