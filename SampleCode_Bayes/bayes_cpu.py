"""CPU MAP estimation for two-component phase SDEs.

The Fourier convention, midpoint observations and divergence correction match
the paper scripts. Coefficients are ordered [spatial modes, temporal modes].
The prior is held fixed during repeated optimization of the SAME data batch:
    Xi = Xi_prior + dt * sum(G.H @ inv(E) @ G)
    r  = Xi_prior @ a_prior + dt * sum(G.H @ inv(E) @ y)
         - dt/2 * sum(partial_G.H @ ones(2)).
An iteration is an optimization step, not another independent observation.

Only F (N,R) is stored. The much larger, mostly zero G and partial_G tensors
are never constructed. The Gram matrix and other sufficient statistics are
computed once per fit and reused for all coefficient updates.
"""
from __future__ import annotations

import numpy as np


def fourier_modes(Ms=2, Mt=1):
    """Modes in the same lexicographic order as the original estimator."""
    if int(Ms) != Ms or int(Mt) != Mt or Ms < 0 or Mt < 0:
        raise ValueError("Ms and Mt must be nonnegative integers.")
    return np.array([(s, t) for s in range(-int(Ms), int(Ms)+1)
                     for t in range(-int(Mt), int(Mt)+1)], dtype=int)


def fourier_features(delta_phi, delta_theta, modes, L=125.0):
    """Return F[n,m] = exp(i*(m_s*pi*DeltaPhi/L + m_t*DeltaTheta))."""
    if not np.isfinite(L) or L <= 0:
        raise ValueError("L must be finite and positive.")
    phi, theta = np.broadcast_arrays(delta_phi, delta_theta)
    phase = (np.remainder(phi.ravel(), 2*L)[:, None] * (np.pi/L)*modes[:, 0]
             + np.remainder(theta.ravel(), 2*np.pi)[:, None]*modes[:, 1])
    return np.exp(1j*phase)


def _reverse_modes(modes):
    lookup = {tuple(mode): index for index, mode in enumerate(modes)}
    try:
        return np.array([lookup[tuple(-mode)] for mode in modes])
    except KeyError as error:
        raise ValueError("Every Fourier mode must have its negative partner.") from error


def _prior(modes, prior_mean, prior_precision):
    R = len(modes)
    mean = (np.zeros(2*R, dtype=complex) if prior_mean is None
            else np.asarray(prior_mean, dtype=complex).reshape(-1).copy())
    if mean.shape != (2*R,) or not np.isfinite(mean).all():
        raise ValueError("prior_mean must contain 2*R finite coefficients.")
    precision = np.asarray(prior_precision, dtype=complex)
    if precision.ndim == 0:
        precision = precision * np.eye(2*R, dtype=complex)
    else:
        precision = precision.copy()
    if precision.shape != (2*R, 2*R) or not np.isfinite(precision).all():
        raise ValueError("prior_precision must be a scalar or a finite (2R,2R) matrix.")
    if not np.allclose(precision, precision.conj().T, rtol=1e-10, atol=1e-12):
        raise ValueError("prior_precision must be Hermitian.")
    if np.linalg.eigvalsh(precision).min() < -1e-12:
        raise ValueError("prior_precision must be positive semidefinite.")
    reverse = _reverse_modes(modes)
    reverse_full = np.concatenate((reverse, reverse+R))
    if not np.allclose(mean, mean[reverse_full].conj(), atol=1e-12):
        raise ValueError("prior_mean must satisfy conjugacy: a[-m] = conj(a[m]).")
    if not np.allclose(precision, precision[np.ix_(reverse_full, reverse_full)].conj(),
                       rtol=1e-10, atol=1e-12):
        raise ValueError("prior_precision must preserve Fourier conjugacy.")
    return mean, precision, reverse_full


def _statistics(features, derivatives, modes, L):
    """Sufficient statistics for the block-diagonal two-component G."""
    F = np.asarray(features, dtype=complex)
    y = np.asarray(derivatives, dtype=float)
    if F.ndim != 2 or F.shape[1] != len(modes) or y.shape != (F.shape[0], 2):
        raise ValueError("Expected features (N,R) and derivatives (N,2).")
    if F.shape[0] < 1 or not np.isfinite(F).all() or not np.isfinite(y).all():
        raise ValueError("The data must be nonempty and finite.")
    gram = F.conj().T @ F
    gram = (gram + gram.conj().T)/2
    cross = F.conj().T @ y
    sum_F = F.sum(axis=0)
    partial_s = (1j*modes[:, 0]*(np.pi/L))*sum_F
    partial_t = (1j*modes[:, 1])*sum_F
    divergence = np.concatenate((partial_s.conj(), partial_t.conj()))
    return gram, cross, divergence


def _solve_posterior(stats, Delta_t, E, mean, precision, reverse_full):
    gram, cross, divergence = stats
    noise = np.asarray(E, dtype=float)
    if noise.shape != (2, 2) or not np.isfinite(noise).all():
        raise ValueError("E must be a finite (2,2) matrix.")
    if not np.allclose(noise, noise.T, atol=1e-12) or np.linalg.eigvalsh(noise).min() <= 0:
        raise ValueError("E must be symmetric positive definite.")
    inv_noise = np.linalg.inv(noise)
    # Each block (p,q) is inv(E)[p,q] * F.H @ F.
    Xi = precision + Delta_t*np.kron(inv_noise, gram)
    Xi = (Xi + Xi.conj().T)/2
    likelihood_rhs = (cross @ inv_noise.T).T.reshape(-1)
    # Use the FIXED prior mean, never the previous optimized coefficient vector.
    rhs = precision @ mean + Delta_t*likelihood_rhs - (Delta_t/2)*divergence
    try:
        coefficient = np.linalg.solve(Xi, rhs)
    except np.linalg.LinAlgError as error:
        raise ValueError("Singular posterior precision; use more data or a positive prior.") from error
    # Remove roundoff-only deviations from the conjugacy required by real drift.
    coefficient = (coefficient + coefficient[reverse_full].conj())/2
    return coefficient.reshape(2, -1), Xi


def posterior_update(features, derivatives, Delta_t, modes, L=125.0, E=None,
                     prior_mean=None, prior_precision=1e-6):
    """One conditional MAP coefficient update for a supplied noise covariance.

    Returns ``(coefficients[2,R], precision[2R,2R])``. This function is also a
    small, directly testable representation of the paper's update equations.
    """
    if not np.isfinite(Delta_t) or Delta_t <= 0:
        raise ValueError("Delta_t must be finite and positive.")
    if not np.isfinite(L) or L <= 0:
        raise ValueError("L must be finite and positive.")
    modes = np.asarray(modes, dtype=int)
    mean, precision, reverse = _prior(modes, prior_mean, prior_precision)
    stats = _statistics(features, derivatives, modes, L)
    return _solve_posterior(stats, Delta_t, np.eye(2) if E is None else E,
                            mean, precision, reverse)


def _noise_from_residual(F, y, coefficient, Delta_t, noise_floor):
    # Recompute real residuals directly rather than subtracting nearly equal
    # large sufficient-statistic terms when the noise is small.
    residual = y - (F @ coefficient.T).real
    E = (Delta_t/len(y))*(residual.T @ residual)
    E = (E + E.T)/2
    values, vectors = np.linalg.eigh(E)
    regularized = bool(np.any(values < noise_floor))
    values = np.maximum(values, noise_floor)
    E = (vectors*values) @ vectors.T
    return (E + E.T)/2, regularized


def fit(data, Delta_t, *, L=125.0, Ms=2, Mt=1, oscillator=1,
        max_iterations=50, rtol=1e-5, prior_mean=None, prior_precision=1e-6,
        initial_noise=None, noise_floor=1e-12, estimate_noise=True):
    """Fit one oscillator's spatial/temporal drift and full noise covariance.

    ``data`` contains Phi1_ast, Phi2_ast, Theta1_ast, Theta2_ast, dot_Phi1,
    dot_Phi2, dot_Theta1, dot_Theta2, each a one-dimensional array of length N.
    Select oscillator=1 or 2; its partner's phase defines the phase difference.

    With ``estimate_noise=True``, each cycle solves coefficients at the current
    E and then updates E from the LATEST coefficients. At finite iteration
    count ``noise_for_last_coefficient_update`` records the E used for the last
    conditional precision. At convergence it agrees with E to tolerance.
    ``estimate_noise=False`` fixes E, useful when the noise is known.
    """
    if oscillator not in (1, 2):
        raise ValueError("oscillator must be 1 or 2.")
    if not np.isfinite(Delta_t) or Delta_t <= 0:
        raise ValueError("Delta_t must be finite and positive.")
    if int(max_iterations) != max_iterations or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer.")
    if not np.isfinite(rtol) or rtol <= 0:
        raise ValueError("rtol must be finite and positive.")
    if not np.isfinite(noise_floor) or noise_floor <= 0:
        raise ValueError("noise_floor must be finite and positive.")
    names = [f"{prefix}{index}{suffix}" for prefix, suffix in
             (("Phi", "_ast"), ("Theta", "_ast"), ("dot_Phi", ""), ("dot_Theta", ""))
             for index in (1, 2)]
    arrays = {name: np.asarray(data[name], dtype=float) for name in names}
    first = arrays[names[0]]
    if first.ndim != 1 or first.size < 2:
        raise ValueError("Each input must be a one-dimensional array with at least two samples.")
    if any(a.shape != first.shape or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("All eight input arrays must have matching shapes and finite values.")
    modes = fourier_modes(Ms, Mt)
    i, j = oscillator, 3-oscillator
    F = fourier_features(arrays[f"Phi{i}_ast"]-arrays[f"Phi{j}_ast"],
                         arrays[f"Theta{i}_ast"]-arrays[f"Theta{j}_ast"], modes, L)
    y = np.column_stack((arrays[f"dot_Phi{i}"], arrays[f"dot_Theta{i}"]))
    mean, precision, reverse = _prior(modes, prior_mean, prior_precision)
    stats = _statistics(F, y, modes, L)
    coefficient = mean.reshape(2, -1).copy()
    regularizations = 0
    if initial_noise is None:
        E, regularized = _noise_from_residual(F, y, coefficient, Delta_t, noise_floor)
        regularizations += int(regularized)
    else:
        E = np.asarray(initial_noise, dtype=float).copy()
    errors, noise_errors = [], []
    converged = False
    for _ in range(int(max_iterations)):
        previous_coefficient, previous_noise = coefficient, E.copy()
        coefficient_noise = E.copy()
        coefficient, Xi = _solve_posterior(stats, Delta_t, E, mean, precision, reverse)
        if estimate_noise:
            E, regularized = _noise_from_residual(F, y, coefficient, Delta_t, noise_floor)
            regularizations += int(regularized)
        error = np.linalg.norm(coefficient-previous_coefficient)/max(np.linalg.norm(previous_coefficient), 1e-12)
        noise_error = np.linalg.norm(E-previous_noise)/max(np.linalg.norm(previous_noise), 1e-30)
        errors.append(float(error))
        noise_errors.append(float(noise_error))
        if max(error, noise_error) < rtol:
            converged = True
            break
    return {
        "coefficients": coefficient, "precision": Xi, "E": E,
        "errors": np.asarray(errors), "noise_errors": np.asarray(noise_errors),
        "modes": modes, "converged": converged, "iterations": len(errors),
        "L": float(L), "Delta_t": float(Delta_t), "n_samples": len(y),
        "oscillator": oscillator, "prior_mean": mean.reshape(2, -1),
        "prior_precision": precision,
        "noise_for_last_coefficient_update": coefficient_noise,
        "noise_floor": float(noise_floor), "noise_floor_applications": regularizations,
    }


def evaluate_drift(result, delta_phi, delta_theta, *, include_constant=True):
    """Evaluate the real drift on broadcast-compatible phase coordinates.

    Returns an array with shape ``broadcast_shape + (2,)``: spatial then
    temporal component. For a mesh, supply phi[None,:], theta[:,None].
    """
    phi, theta = np.broadcast_arrays(delta_phi, delta_theta)
    modes = np.asarray(result["modes"])
    coefficient = np.asarray(result["coefficients"]).copy()
    if not include_constant:
        coefficient[:, np.all(modes == 0, axis=1)] = 0
    F = fourier_features(phi, theta, modes, result["L"])
    return (F @ coefficient.T).real.reshape(phi.shape+(2,))
