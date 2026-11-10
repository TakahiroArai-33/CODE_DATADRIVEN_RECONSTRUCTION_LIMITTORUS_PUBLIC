"""Portable CPU phase-SDE simulation and periodic stratified sampling.

State order is [Phi1, Theta1, Phi2, Theta2]. Phases remain unwrapped during
integration; only coupling arguments and the estimation midpoint phases wrap.
The bundled one-oscillator coupling functions are sampled from the paper data.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from numba import njit


SAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = SAMPLE_DIR / "InputData" / "phase_equation.npz"
ESTIMATION_KEYS = (
    "Phi1_ast", "Phi2_ast", "Theta1_ast", "Theta2_ast",
    "dot_Phi1", "dot_Phi2", "dot_Theta1", "dot_Theta2",
)


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_phase_equation_bundle(path=DEFAULT_MODEL_PATH, repo_root=None, stride=8, overwrite=False):
    """Create a small, self-contained model from the repository's real arrays.

    The original 2049 x 2049 fields include both periodic endpoints. Exact
    [::8, ::8] sampling therefore produces 257 x 257 fields, with no filtering
    or fitting. The original three files are read only. Hashes and stride are
    embedded so the bundle's provenance can be checked without those sources.
    """
    stride = _positive_integer(stride, "stride")
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    root = Path(repo_root).resolve() if repo_root else SAMPLE_DIR.parent
    source_paths = {
        "coupling": root / "PhaseEquation" / "CouplingFunction_8193.npz",
        "frequency": root / "PhaseEquation" / "freq_8193.npz",
        "noise_C": root / "SRC_論文改訂チェック用_ベイズのチェック" / "InterimData" / "theoretical_sde_constants.npz",
    }
    with np.load(source_paths["coupling"], allow_pickle=False) as source:
        phi = source["delta_phi"]
        theta = source["delta_theta"]
        gamma_s = source["Gamma_s"]
        gamma_t = source["Gamma_t"]
    if (phi.size - 1) % stride or (theta.size - 1) % stride:
        raise ValueError("stride must preserve both endpoints of each source grid")
    with np.load(source_paths["frequency"], allow_pickle=False) as source:
        c, omega = float(source["c"]), float(source["omega"])
    with np.load(source_paths["noise_C"], allow_pickle=False) as source:
        covariance_C = source["C"]
    metadata = {
        "schema_version": 1,
        "source_files": {
            key: {"path": str(source.relative_to(root)), "sha256": _file_sha256(source)}
            for key, source in source_paths.items()
        },
        "source_field_shape": list(gamma_s.shape),
        "exact_grid_stride": stride,
        "bundled_field_shape": list(gamma_s[::stride, ::stride].shape),
        "field_sampling": "Exact [::stride, ::stride]; periodic endpoints retained; no filtering or fitting.",
        "frequency_convention": "Raw no-noise c and omega from freq_8193.npz; no noise-induced frequency shift.",
        "noise_convention": "E = sigma**2 * C for each oscillator; noises of the two oscillators are independent.",
        "state_order": ["Phi1", "Theta1", "Phi2", "Theta2"],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination, gamma_s=gamma_s[::stride, ::stride], gamma_t=gamma_t[::stride, ::stride],
        phi_grid=phi[::stride], theta_grid=theta[::stride], C=covariance_C,
        c=c, omega=omega, L=0.5 * float(np.ptp(phi)),
        metadata_json=json.dumps(metadata, ensure_ascii=False),
    )
    return destination


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate_seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    return int(seed)


def _validate_model(model):
    arrays = {key: np.asarray(model[key], dtype=np.float64) for key in
              ("gamma_s", "gamma_t", "phi_grid", "theta_grid", "C")}
    phi, theta = arrays["phi_grid"], arrays["theta_grid"]
    for name, grid in (("phi_grid", phi), ("theta_grid", theta)):
        if grid.ndim != 1 or grid.size < 2 or not np.all(np.isfinite(grid)):
            raise ValueError(f"{name} must be a finite one-dimensional grid")
        differences = np.diff(grid)
        if grid[0] != 0.0 or not np.all(differences > 0) or not np.allclose(differences, differences[0]):
            raise ValueError(f"{name} must be uniformly increasing from zero")
    if not np.isclose(theta[-1], 2 * np.pi):
        raise ValueError("theta_grid must include the endpoint 2*pi")
    for name in ("gamma_s", "gamma_t"):
        field = arrays[name]
        if field.shape != (theta.size, phi.size) or not np.all(np.isfinite(field)):
            raise ValueError(f"{name} must have shape (n_theta, n_phi) and finite values")
        if not (np.allclose(field[0], field[-1], atol=1e-9) and np.allclose(field[:, 0], field[:, -1], atol=1e-9)):
            raise ValueError(f"{name} must agree at the periodic endpoints")
    covariance_C = arrays["C"]
    if covariance_C.shape != (2, 2) or not np.all(np.isfinite(covariance_C)) or not np.allclose(covariance_C, covariance_C.T):
        raise ValueError("C must be a finite symmetric 2 x 2 covariance matrix")
    if np.linalg.eigvalsh(covariance_C).min() < -1e-10:
        raise ValueError("C must be positive semidefinite")
    for name in ("c", "omega", "L"):
        if not np.isfinite(float(model[name])):
            raise ValueError(f"{name} must be finite")
    if float(model["L"]) <= 0 or not np.isclose(phi[-1], 2 * float(model["L"])):
        raise ValueError("phi_grid must include the endpoint 2*L, with L > 0")
    return arrays


def load_phase_equation(path=DEFAULT_MODEL_PATH):
    """Load the portable NPZ without pickle and validate its periodic grids."""
    with np.load(Path(path), allow_pickle=False) as source:
        model = {key: source[key].copy() for key in ("gamma_s", "gamma_t", "phi_grid", "theta_grid", "C")}
        model.update({key: float(source[key]) for key in ("c", "omega", "L")})
        model["metadata"] = json.loads(str(source["metadata_json"])) if "metadata_json" in source.files else {}
    _validate_model(model)
    return model


@njit(cache=True)
def _interpolate(field, delta_phi, delta_theta, phi_period, theta_period):
    """Bilinear interpolation of endpoint-inclusive periodic fields."""
    x = (delta_phi % phi_period) * (field.shape[1] - 1) / phi_period
    y = (delta_theta % theta_period) * (field.shape[0] - 1) / theta_period
    ix = min(int(np.floor(x)), field.shape[1] - 2)
    iy = min(int(np.floor(y)), field.shape[0] - 2)
    ax, ay = x - ix, y - iy
    return ((1 - ay) * (1 - ax) * field[iy, ix] + (1 - ay) * ax * field[iy, ix + 1]
            + ay * (1 - ax) * field[iy + 1, ix] + ay * ax * field[iy + 1, ix + 1])


@njit(cache=True)
def _simulate_batch(initial, dt, n_steps, stride, gamma_s, gamma_t, phi_period, c, omega, epsilon, noise_factor, seed):
    np.random.seed(seed)
    saved = np.empty((initial.shape[0], n_steps // stride + 1, 4), dtype=np.float64)
    sqrt_dt = np.sqrt(dt)
    for trajectory_index in range(initial.shape[0]):
        state = initial[trajectory_index].copy()
        saved[trajectory_index, 0] = state
        saved_index = 1
        for step in range(1, n_steps + 1):
            delta_phi, delta_theta = state[0] - state[2], state[1] - state[3]
            drift_phi1 = c + epsilon * _interpolate(gamma_s, delta_phi, delta_theta, phi_period, 2 * np.pi)
            drift_theta1 = omega + epsilon * _interpolate(gamma_t, delta_phi, delta_theta, phi_period, 2 * np.pi)
            drift_phi2 = c + epsilon * _interpolate(gamma_s, -delta_phi, -delta_theta, phi_period, 2 * np.pi)
            drift_theta2 = omega + epsilon * _interpolate(gamma_t, -delta_phi, -delta_theta, phi_period, 2 * np.pi)
            z1, z2, z3, z4 = np.random.randn(4)
            state[0] += drift_phi1 * dt + sqrt_dt * (noise_factor[0, 0] * z1 + noise_factor[0, 1] * z2)
            state[1] += drift_theta1 * dt + sqrt_dt * (noise_factor[1, 0] * z1 + noise_factor[1, 1] * z2)
            state[2] += drift_phi2 * dt + sqrt_dt * (noise_factor[0, 0] * z3 + noise_factor[0, 1] * z4)
            state[3] += drift_theta2 * dt + sqrt_dt * (noise_factor[1, 0] * z3 + noise_factor[1, 1] * z4)
            if step % stride == 0:
                saved[trajectory_index, saved_index] = state
                saved_index += 1
    return saved


def simulate_dataset(model, epsilon=1e-6, sigma=1e-5, dt=0.05, t_end=1000.0,
                     save_every=20, n_phi=8, n_theta=8, seed=0):
    """Simulate a grid of phase differences with CPU Euler--Maruyama.

    Each oscillator has covariance E=sigma**2*C per unit time. The two
    oscillators have independent noise. Stored phases are never wrapped.
    t_end/dt must be an integer multiple of save_every, so the exact final
    requested time is present in the output rather than silently discarded.
    """
    arrays = _validate_model(model)
    n_phi, n_theta = _positive_integer(n_phi, "n_phi"), _positive_integer(n_theta, "n_theta")
    save_every = _positive_integer(save_every, "save_every")
    seed = _validate_seed(seed)
    epsilon, sigma, dt, t_end = map(float, (epsilon, sigma, dt, t_end))
    if not np.all(np.isfinite([epsilon, sigma, dt, t_end])) or sigma < 0 or dt <= 0 or t_end <= 0:
        raise ValueError("epsilon, sigma, dt and t_end must be finite; sigma >= 0, dt > 0, t_end > 0")
    n_steps = int(round(t_end / dt))
    if n_steps < 1 or not np.isclose(n_steps * dt, t_end, rtol=1e-12, atol=1e-12):
        raise ValueError("t_end must be an integer multiple of dt")
    if n_steps % save_every:
        raise ValueError("the number of integration steps must be divisible by save_every")
    phi_period = float(arrays["phi_grid"][-1])
    initial = np.zeros((n_phi * n_theta, 4), dtype=np.float64)
    phi_initial, theta_initial = np.meshgrid(
        np.linspace(0, phi_period, n_phi, endpoint=False),
        np.linspace(0, 2 * np.pi, n_theta, endpoint=False), indexing="xy",
    )
    initial[:, 0], initial[:, 1] = phi_initial.ravel(), theta_initial.ravel()
    eigvals, eigvecs = np.linalg.eigh(arrays["C"])
    noise_factor = sigma * eigvecs * np.sqrt(np.maximum(eigvals, 0.0))
    trajectory = _simulate_batch(
        initial, dt, n_steps, save_every, np.ascontiguousarray(arrays["gamma_s"]),
        np.ascontiguousarray(arrays["gamma_t"]), phi_period, float(model["c"]),
        float(model["omega"]), epsilon, noise_factor, seed,
    )
    t = np.arange(trajectory.shape[1], dtype=np.float64) * (dt * save_every)
    t[-1] = t_end
    covariance_E = sigma**2 * arrays["C"]
    covariance_full = np.zeros((4, 4))
    covariance_full[:2, :2] = covariance_E
    covariance_full[2:, 2:] = covariance_E
    return {
        "trajectory": trajectory, "t": t, "initial_conditions": initial,
        "Delta_t": dt * save_every, "dt": dt, "save_every": save_every,
        "t_end": t_end, "epsilon": epsilon, "sigma": sigma, "L": float(model["L"]),
        "C": arrays["C"].copy(), "E": covariance_E, "E_full": covariance_full,
        "c": float(model["c"]), "omega": float(model["omega"]), "seed": seed,
        "metadata": {"method": "CPU Euler-Maruyama", "n_phi": n_phi, "n_theta": n_theta,
                     "state_order": ["Phi1", "Theta1", "Phi2", "Theta2"],
                     "model": model.get("metadata", {})},
    }


def _periodic_bin_indices(values, period, bins):
    # Keep the final edge at period: wrapping this edge to zero excludes the
    # complete last row/column in the original thinning implementation.
    edges = np.linspace(0.0, period, bins + 1)
    indices = np.searchsorted(edges, np.mod(values, period), side="right") - 1
    return np.clip(indices, 0, bins - 1)


def prepare_estimation_data(data, bins=16, max_per_cell=100, seed=0):
    """Return the eight midpoint/velocity arrays expected by Bayesian fitting.

    Midpoints and differences are computed from unwrapped trajectories first.
    Per-phase wrapping and phase-difference binning happen afterwards. Bins
    cover the full periodic domain, including the final spatial/temporal bins.
    Within each cell, choose at most max_per_cell samples without replacement.
    """
    if isinstance(bins, (int, np.integer)):
        n_phi_bins = n_theta_bins = _positive_integer(bins, "bins")
    else:
        if len(bins) != 2:
            raise ValueError("bins must be a positive integer or (n_phi_bins, n_theta_bins)")
        n_phi_bins = _positive_integer(bins[0], "n_phi_bins")
        n_theta_bins = _positive_integer(bins[1], "n_theta_bins")
    max_per_cell = _positive_integer(max_per_cell, "max_per_cell")
    seed = _validate_seed(seed)
    trajectory = np.asarray(data["trajectory"], dtype=np.float64)
    t = np.asarray(data["t"], dtype=np.float64)
    if trajectory.ndim != 3 or trajectory.shape[-1] != 4 or trajectory.shape[0] < 1 or trajectory.shape[1] < 2:
        raise ValueError("trajectory must have shape (n_trajectories, n_times>=2, 4)")
    if t.ndim != 1 or t.size != trajectory.shape[1] or not np.all(np.isfinite(t)) or not np.all(np.diff(t) > 0):
        raise ValueError("t must be finite, strictly increasing, and match the trajectory time axis")
    if not np.all(np.isfinite(trajectory)):
        raise ValueError("trajectory contains non-finite phase values")
    L = float(data["L"])
    if not np.isfinite(L) or L <= 0:
        raise ValueError("L must be finite and positive")
    midpoint = 0.5 * (trajectory[:, :-1] + trajectory[:, 1:])
    velocity = np.diff(trajectory, axis=1) / np.diff(t)[None, :, None]
    phi1 = np.mod(midpoint[:, :, 0].ravel(), 2 * L)
    theta1 = np.mod(midpoint[:, :, 1].ravel(), 2 * np.pi)
    phi2 = np.mod(midpoint[:, :, 2].ravel(), 2 * L)
    theta2 = np.mod(midpoint[:, :, 3].ravel(), 2 * np.pi)
    all_data = dict(zip(ESTIMATION_KEYS, (
        phi1, phi2, theta1, theta2,
        velocity[:, :, 0].ravel(), velocity[:, :, 2].ravel(),
        velocity[:, :, 1].ravel(), velocity[:, :, 3].ravel(),
    )))
    phi_bin = _periodic_bin_indices(phi1 - phi2, 2 * L, n_phi_bins)
    theta_bin = _periodic_bin_indices(theta1 - theta2, 2 * np.pi, n_theta_bins)
    cell_ids = phi_bin + n_phi_bins * theta_bin
    order = np.argsort(cell_ids, kind="stable")
    _, starts, counts = np.unique(cell_ids[order], return_index=True, return_counts=True)
    rng = np.random.default_rng(seed)
    selected = []
    for start, count in zip(starts, counts):
        members = order[start:start + count]
        selected.append(members if count <= max_per_cell else rng.choice(members, max_per_cell, replace=False))
    selected = np.sort(np.concatenate(selected))
    return {key: np.ascontiguousarray(values[selected]) for key, values in all_data.items()}
