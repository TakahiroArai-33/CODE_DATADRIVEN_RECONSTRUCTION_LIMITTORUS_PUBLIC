"""Independent formula checks for the CPU phase-SDE estimator."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bayes_cpu import fit, fourier_modes, fourier_features, posterior_update, evaluate_drift


def data_from(phi, theta, derivatives):
    n = len(phi)
    return {
        "Phi1_ast": phi, "Phi2_ast": np.zeros(n),
        "Theta1_ast": theta, "Theta2_ast": np.zeros(n),
        "dot_Phi1": derivatives[:, 0], "dot_Theta1": derivatives[:, 1],
        "dot_Phi2": np.zeros(n), "dot_Theta2": np.zeros(n),
    }


class BayesianUpdatesTest(unittest.TestCase):
    def test_nonzero_prior_mean_analytic_update(self):
        # Constant drift has a closed-form conjugate Gaussian posterior. A
        # moderately strong nonzero prior makes replacing a_prior with a_old
        # (or silently assuming zero prior mean) readily detectable.
        modes = fourier_modes(0, 0)
        F = np.ones((4, 1), dtype=complex)
        y = np.array([[1., 3.], [2., 2.], [3., 1.], [2., 2.]])
        prior_mean = np.array([5., -2.])
        prior_precision = np.diag([7., 11.])
        E = np.array([[2., .3], [.3, 1.]])
        dt = .2
        expected_precision = prior_precision + dt*len(y)*np.linalg.inv(E)
        expected_rhs = prior_precision @ prior_mean + dt*np.linalg.inv(E) @ y.sum(axis=0)
        a, precision = posterior_update(F, y, dt, modes, E=E,
                                       prior_mean=prior_mean, prior_precision=prior_precision)
        np.testing.assert_allclose(precision, expected_precision, rtol=1e-13)
        np.testing.assert_allclose(a.ravel(), np.linalg.solve(expected_precision, expected_rhs), rtol=1e-13)

    def test_block_statistics_equal_direct_G_formula(self):
        rng = np.random.default_rng(82)
        modes = fourier_modes(1, 1)
        R, n, L, dt = len(modes), 37, 125., .17
        phi = rng.uniform(-L, L, n)
        theta = rng.uniform(-np.pi, np.pi, n)
        F = fourier_features(phi, theta, modes, L)
        y = rng.normal(size=(n, 2))
        E = np.array([[.8, .12], [.12, .5]])
        prior = np.eye(2*R)*3.5
        mean = np.zeros((2, R), dtype=complex)
        mean[:, np.all(modes == 0, axis=1)] = [[.4], [-.2]]
        G = np.zeros((n, 2, 2*R), dtype=complex)
        D = np.zeros_like(G)
        G[:, 0, :R], G[:, 1, R:] = F, F
        D[:, 0, :R] = F*(1j*modes[:, 0]*np.pi/L)
        D[:, 1, R:] = F*(1j*modes[:, 1])
        expected_Xi = prior.copy().astype(complex)
        expected_rhs = prior @ mean.ravel()
        for g, d, yn in zip(G, D, y):
            expected_Xi += dt*g.conj().T @ np.linalg.inv(E) @ g
            expected_rhs += dt*g.conj().T @ np.linalg.inv(E) @ yn - dt/2*d.conj().T @ np.ones(2)
        a, Xi = posterior_update(F, y, dt, modes, L, E, mean, prior)
        np.testing.assert_allclose(Xi, expected_Xi, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(a.ravel(), np.linalg.solve(expected_Xi, expected_rhs), rtol=1e-11, atol=1e-12)

    def test_repeated_same_batch_does_not_accumulate_precision(self):
        rng = np.random.default_rng(9)
        n, dt = 100, .5
        y = rng.normal(loc=[1., -.4], scale=.4, size=(n, 2))
        data = data_from(np.zeros(n), np.zeros(n), y)
        prior_mean = np.array([3., -2.])
        prior_precision = np.diag([20., 15.])
        kwargs = dict(Ms=0, Mt=0, prior_mean=prior_mean,
                      prior_precision=prior_precision, initial_noise=np.eye(2), estimate_noise=False)
        once = fit(data, dt, max_iterations=1, **kwargs)
        repeated = fit(data, dt, max_iterations=8, **kwargs)
        np.testing.assert_array_equal(once["precision"], repeated["precision"])
        np.testing.assert_array_equal(once["coefficients"], repeated["coefficients"])
        self.assertTrue(repeated["converged"])

    def test_noise_uses_latest_coefficients_and_is_symmetric(self):
        rng = np.random.default_rng(29)
        n, dt = 400, .2
        phi = rng.uniform(-125, 125, n)
        theta = rng.uniform(-np.pi, np.pi, n)
        y = rng.multivariate_normal([1., -.2], [[.3, .06], [.06, .2]], size=n)
        result = fit(data_from(phi, theta, y), dt, Ms=1, Mt=1, max_iterations=1,
                     initial_noise=np.eye(2), prior_precision=2.)
        pred = evaluate_drift(result, phi, theta)
        residual = y-pred
        expected_E = dt/n*(residual.T @ residual)
        np.testing.assert_allclose(result["E"], expected_E, atol=1e-13)
        np.testing.assert_allclose(result["E"], result["E"].T, atol=1e-14)
        self.assertTrue(np.linalg.eigvalsh(result["E"]).min() > 0)
        self.assertEqual(result["precision"].shape, (18, 18))
        np.testing.assert_allclose(result["precision"], result["precision"].conj().T)
        reverse = np.arange(8, -1, -1)
        np.testing.assert_allclose(result["coefficients"], result["coefficients"][:, reverse].conj(), atol=1e-13)

    def test_known_fourier_drift_and_grid_evaluation(self):
        # Weak measurement noise and an approximately uniform mesh resolve the
        # Fourier basis; compare the resulting function, not implementation steps.
        L = 125.
        phi, theta = np.meshgrid(np.linspace(-L, L, 31, endpoint=False),
                                 np.linspace(-np.pi, np.pi, 25, endpoint=False))
        phi, theta = phi.ravel(), theta.ravel()
        truth = np.column_stack((.4+.03*np.sin(np.pi*phi/L), .2+.07*np.cos(theta)))
        result = fit(data_from(phi, theta, truth), .01, Ms=1, Mt=1,
                     initial_noise=np.eye(2)*1e-5, estimate_noise=False,
                     prior_precision=1e-6)
        np.testing.assert_allclose(evaluate_drift(result, phi, theta), truth, atol=1e-6)
        grid = evaluate_drift(result, np.array([0., 1.])[None, :], np.array([0., .3, .6])[:, None])
        self.assertEqual(grid.shape, (3, 2, 2))


if __name__ == "__main__":
    unittest.main()
