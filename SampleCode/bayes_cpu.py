"""NumPy version of the updates in SRC/Module_Bayes_GPU.py.

Coefficients are ordered [spatial Fourier modes, temporal Fourier modes].
F stores the nonzero blocks of G; its Gram matrix gives exactly the same
contractions as the original block tensors. The prior stays fixed throughout
the iterations on one data set.
"""

import numpy as np


def fit(delta_phase, velocity, Delta_t, L, Ms=20, Mt=5,
        max_iterations=50, rtol=1e-5):
    """Estimate one system from midpoint phase differences and velocities.

    Both inputs have shape (N, 2), ordered (Phi, Theta). E is unknown: each
    iteration updates it from the OLD coefficients before updating Xi and a.
    The conditional Gaussian posterior mean a also equals its mode.
    """
    modes = np.array([(ms, mt) for ms in range(-Ms, Ms + 1)
                      for mt in range(-Mt, Mt + 1)])
    N, R = len(velocity), len(modes)
    phase = np.mod(delta_phase, [2 * L, 2 * np.pi])
    F = np.exp(1j * (phase[:, :1] * (np.pi / L) * modes[:, 0]
                     + phase[:, 1:] * modes[:, 1]))
    gram = F.conj().T @ F
    cross = F.conj().T @ velocity
    sum_F = F.sum(axis=0)
    partial_sum = np.concatenate((
        (1j * modes[:, 0] * (np.pi / L) * sum_F).conj(),
        (1j * modes[:, 1] * sum_F).conj(),
    ))

    a_prior = np.zeros(2 * R, dtype=complex)
    Xi_prior = 1e-6 * np.eye(2 * R, dtype=complex)
    a = a_prior.copy()
    E = np.eye(2)
    errors = []
    for _ in range(max_iterations):
        a_old = a.copy()

        # Original update order: residual(a_old) -> E -> Xi -> r -> a.
        residual = velocity - (F @ a_old.reshape(2, R).T).real
        E = (Delta_t / N) * (residual.T @ residual)
        E = (E + E.T) / 2
        Einv = np.linalg.inv(E)
        Xi = Xi_prior + Delta_t * np.kron(Einv, gram)
        Xi = (Xi + Xi.conj().T) / 2
        r = (Xi_prior @ a_prior
             + Delta_t * (cross @ Einv.T).T.ravel()
             - (Delta_t / 2) * partial_sum)
        a = np.linalg.solve(Xi, r)

        error = np.linalg.norm(a - a_old) / max(np.linalg.norm(a_old), 1e-8)
        errors.append(error)
        if error < rtol:
            break

    return {"a": a.reshape(2, R), "E": E, "modes": modes,
            "errors": np.array(errors)}


def evaluate_nonconstant(result, phi_grid, theta_grid, L):
    """Evaluate nonconstant posterior mean drift; shape (n_theta,n_phi,2)."""
    modes = result["modes"]
    coefficients = result["a"].copy()
    coefficients[:, np.all(modes == 0, axis=1)] = 0
    ms = np.unique(modes[:, 0])
    mt = np.unique(modes[:, 1])
    basis_s = np.exp(1j * np.outer(phi_grid, ms * (np.pi / L)))
    basis_t = np.exp(1j * np.outer(theta_grid, mt))
    return np.stack([
        (basis_t @ a.reshape(len(ms), len(mt)).T @ basis_s.T).real
        for a in coefficients
    ], axis=-1)
