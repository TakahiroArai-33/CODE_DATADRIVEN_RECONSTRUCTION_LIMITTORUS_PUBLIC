"""CPU/Numba solver for one realization of two coupled Gray--Scott systems.

Adapted from ``Chaek_UbarAtilde_andSPDEAnime/
ModuleGrayScottModel_JIT_simulation.py`` (reaction, diffusion and coupling)
with the Euler--Maruyama update used by ``Simulate_CoupledModel.py``
through ``Module_Coupled_GSM.py``.

Exactly one trajectory of the coupled pair is integrated; there is no
ensemble dimension. The full time loop is compiled, and only spatial
moments are retained. ``rx`` includes both periodic endpoints, while the
state contains only distinct cells. The Fourier sign is
A_tilde = integral u(x) exp(+2 pi i x / domain_length) dx.
Noise is independent between both systems and both species, with amplitude
sigma*sqrt(dt) per spatial cell (or optional block), without 1/sqrt(dx).
"""

from __future__ import annotations

import operator

import numpy as np
from numba import njit
from tqdm.notebook import tqdm


@njit(cache=True)
def _drift(state, result, h, epsilon):
    """Simultaneous periodic reaction--diffusion drift of one coupled pair."""
    n = state.shape[2]
    inv_h2 = 1.0 / (h * h)
    for field in range(2):
        other = 1 - field
        for i in range(n):
            before = (i - 1) % n
            after = (i + 1) % n
            u, v = state[field, 0, i], state[field, 1, i]
            reaction = u * u * v
            result[field, 0, i] = (
                reaction - (0.018 + 0.052) * u
                + (state[field, 0, after] - 2.0 * u + state[field, 0, before]) * inv_h2
                + epsilon * (state[other, 0, i] - u)
            )
            result[field, 1, i] = (
                -reaction + 0.018 * (1.0 - v)
                + 1.9 * (state[field, 1, after] - 2.0 * v + state[field, 1, before]) * inv_h2
                + epsilon * (state[other, 1, i] - v)
            )


@njit(cache=True)
def _record(state, h, phase, bars, modes, index):
    for field in range(2):
        u_sum, v_sum = 0.0, 0.0
        mode = 0.0j
        for i in range(state.shape[2]):
            u, v = state[field, 0, i], state[field, 1, i]
            if not np.isfinite(u) or not np.isfinite(v):
                raise FloatingPointError("Non-finite SPDE state; reduce dt or noise/coupling strength")
            u_sum += u
            v_sum += v
            mode += u * phase[i]
        bars[index, field, 0] = h * u_sum
        bars[index, field, 1] = h * v_sum
        modes[index, field] = h * mode


@njit(cache=True)
def _integrate(initial, h, phase, dt, n_steps, save_stride, epsilon,
               sigma, seed, noise_block, progress_stride):
    np.random.seed(seed)
    state = initial.copy()
    drift = np.empty_like(state)
    noise = np.zeros_like(state)
    count = 1 + n_steps // save_stride + int(n_steps % save_stride != 0)
    times = np.empty(count)
    bars = np.empty((count, 2, 2))
    modes = np.empty((count, 2), dtype=np.complex128)
    times[0] = 0.0
    _record(state, h, phase, bars, modes, 0)
    save_index = 1
    noise_std = sigma * np.sqrt(dt)

    for step in range(1, n_steps + 1):
        _drift(state, drift, h, epsilon)
        if sigma != 0.0:
            for field in range(2):
                for species in range(2):
                    for start in range(0, state.shape[2], noise_block):
                        increment = noise_std * np.random.normal()
                        for i in range(start, start + noise_block):
                            noise[field, species, i] = increment
        state[:] = state + dt * drift + noise

        if step % save_stride == 0 or step == n_steps:
            times[save_index] = step * dt
            _record(state, h, phase, bars, modes, save_index)
            save_index += 1
        if step % progress_stride == 0 or step == n_steps:
            yield step, times, bars, modes, state


def _integer(value, name, minimum):
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if isinstance(value, (bool, np.bool_)) or result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def simulate(initial, rx, dt, n_steps, save_stride, epsilon=0.0, sigma=0.0,
             seed=12345, noise_block=1, progress=False):
    """Integrate one coupled realization using Euler--Maruyama.

    Parameters
    ----------
    initial : array, shape (2, 2, ncore)
        [system, species (u/v), distinct spatial cell]. No repeated endpoint
        and no ensemble axis.
    rx : array, shape (ncore + 1,)
        Uniform, increasing coordinates including both periodic endpoints.
    dt, n_steps, save_stride
        Fixed timestep, number of steps, and moment sampling stride. Initial
        and final states are sampled, even if the final step is off stride.
    epsilon, sigma : nonnegative float
        Symmetric coupling and additive block-noise standard deviation.
    seed : integer in [0, 2**32)
        Reproducible Numba random stream, not bit-identical to CuPy's stream.
    noise_block : positive integer
        Cells sharing each noise sample; must divide ncore. Default 1 means
        independent noise at every distinct spatial cell.
    progress : bool
        Show a tqdm progress bar, updated every 10,000 integration steps.

    Returns
    -------
    dict
        t, U1_bar, V1_bar, U2_bar, V2_bar, A1_tilde, A2_tilde, final_state.
        Bars are spatial integrals, not means. final_state has shape
        (2, 2, ncore), and does not alias initial.
    """
    initial = np.ascontiguousarray(initial, dtype=np.float64)
    rx = np.asarray(rx, dtype=np.float64)
    if initial.ndim != 3 or initial.shape[:2] != (2, 2) or initial.shape[2] < 3:
        raise ValueError("initial must have shape (2, 2, ncore), ncore >= 3")
    ncore = initial.shape[2]
    if rx.ndim != 1 or rx.size != ncore + 1:
        raise ValueError("rx must include both endpoints and have length ncore + 1")
    if not np.all(np.isfinite(initial)) or not np.all(np.isfinite(rx)):
        raise ValueError("initial and rx must contain finite values")
    h = float((rx[-1] - rx[0]) / ncore)
    if h <= 0 or not np.allclose(np.diff(rx), h, rtol=1e-10, atol=1e-12 * h):
        raise ValueError("rx must be uniform and strictly increasing")
    dt, epsilon, sigma = float(dt), float(epsilon), float(sigma)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be nonnegative and finite")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be nonnegative and finite")
    n_steps = _integer(n_steps, "n_steps", 1)
    save_stride = _integer(save_stride, "save_stride", 1)
    seed = _integer(seed, "seed", 0)
    noise_block = _integer(noise_block, "noise_block", 1)
    if seed >= 2**32:
        raise ValueError("seed must be smaller than 2**32")
    if ncore % noise_block:
        raise ValueError("noise_block must divide ncore")
    # Necessary explicit diffusion/coupling stability bound; extreme
    # nonlinear reaction states may impose a stricter limit.
    dt_limit = 1.0 / (2.0 * 1.9 / h**2 + epsilon)
    if dt > dt_limit:
        raise ValueError(f"dt={dt:g} exceeds diffusion/coupling stability limit {dt_limit:g}")
    if not np.isfinite(n_steps * dt):
        raise ValueError("n_steps * dt must be finite")
    phase = np.exp(2j * np.pi * rx[:-1] / (rx[-1] - rx[0]))
    progress_stride = 10000 if progress else n_steps
    completed = 0
    with tqdm(total=n_steps, desc="SPDE", unit="step", unit_scale=True,
              disable=not progress) as bar:
        for step, t, bars, modes, state in _integrate(
            initial, h, phase, dt, n_steps, save_stride, epsilon,
            sigma, seed, noise_block, progress_stride,
        ):
            bar.update(step - completed)
            completed = step
    return {
        "t": t,
        "U1_bar": bars[:, 0, 0], "V1_bar": bars[:, 0, 1],
        "U2_bar": bars[:, 1, 0], "V2_bar": bars[:, 1, 1],
        "A1_tilde": modes[:, 0], "A2_tilde": modes[:, 1],
        "final_state": state,
    }
