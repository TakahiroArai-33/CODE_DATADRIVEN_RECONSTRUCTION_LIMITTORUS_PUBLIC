"""Small numerical checks; run with python -m unittest discover -s tests -v."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase_sde import (ESTIMATION_KEYS, load_phase_equation, prepare_estimation_data,
                       simulate_dataset, _periodic_bin_indices)


class PhaseSDETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_phase_equation()

    def test_bundle_uses_real_data_with_documented_stride(self):
        self.assertEqual(self.model["gamma_s"].shape, (257, 257))
        self.assertEqual(self.model["gamma_t"].shape, (257, 257))
        self.assertEqual(self.model["metadata"]["exact_grid_stride"], 8)
        self.assertEqual(self.model["metadata"]["source_field_shape"], [2049, 2049])
        self.assertAlmostEqual(self.model["c"], 0.043130953918849364)
        self.assertAlmostEqual(self.model["omega"], 0.02789917730152864)
        self.assertAlmostEqual(self.model["phi_grid"][-1], 250.0)

    def test_reproducible_stochastic_trajectories(self):
        arguments = dict(dt=0.05, t_end=1.0, save_every=2, n_phi=2, n_theta=2, seed=13)
        first = simulate_dataset(self.model, **arguments)
        second = simulate_dataset(self.model, **arguments)
        np.testing.assert_array_equal(first["trajectory"], second["trajectory"])
        different = simulate_dataset(self.model, **dict(arguments, seed=14))
        self.assertFalse(np.array_equal(first["trajectory"], different["trajectory"]))
        np.testing.assert_allclose(first["E"], first["sigma"]**2 * self.model["C"])
        self.assertEqual(first["trajectory"].shape, (4, 11, 4))
        self.assertEqual(first["t"][-1], 1.0)

    def test_zero_noise_and_zero_coupling_give_known_constant_drift(self):
        data = simulate_dataset(self.model, epsilon=0.0, sigma=0.0, dt=0.1, t_end=1.0,
                                save_every=2, n_phi=2, n_theta=3, seed=1)
        drift = np.array([self.model["c"], self.model["omega"], self.model["c"], self.model["omega"]])
        expected = data["initial_conditions"][:, None, :] + data["t"][None, :, None] * drift[None, None, :]
        np.testing.assert_allclose(data["trajectory"], expected, atol=1e-12, rtol=0)
        prepared = prepare_estimation_data(data, bins=4, max_per_cell=100)
        self.assertEqual(set(prepared), set(ESTIMATION_KEYS))
        np.testing.assert_allclose(prepared["dot_Phi1"], self.model["c"], atol=1e-12)
        np.testing.assert_allclose(prepared["dot_Theta2"], self.model["omega"], atol=1e-12)

    def test_sampling_retains_last_row_and_column(self):
        n_phi, n_theta, L = 4, 3, 125.0
        phi, theta = np.meshgrid((np.arange(n_phi) + 0.5) * 2 * L / n_phi,
                                (np.arange(n_theta) + 0.5) * 2 * np.pi / n_theta)
        states = np.zeros((phi.size, 4))
        states[:, 0], states[:, 1] = phi.ravel(), theta.ravel()
        trajectory = np.repeat(states[:, None, :], 3, axis=1)
        prepared = prepare_estimation_data({"trajectory": trajectory, "t": np.array([0., 1., 2.]), "L": L},
                                           bins=(n_phi, n_theta), max_per_cell=1, seed=3)
        self.assertEqual(prepared["Phi1_ast"].size, n_phi * n_theta)
        ix = _periodic_bin_indices(prepared["Phi1_ast"] - prepared["Phi2_ast"], 2 * L, n_phi)
        iy = _periodic_bin_indices(prepared["Theta1_ast"] - prepared["Theta2_ast"], 2 * np.pi, n_theta)
        self.assertEqual(set(zip(ix, iy)), {(x, y) for x in range(n_phi) for y in range(n_theta)})
        endpoint_bins = _periodic_bin_indices(np.array([0., 2 * L, np.nextafter(2 * L, 0), -0.001]), 2 * L, n_phi)
        np.testing.assert_array_equal(endpoint_bins, [0, 0, n_phi - 1, n_phi - 1])

    def test_midpoint_is_formed_before_wrapping(self):
        trajectory = np.array([[[249., 2 * np.pi - 0.1, 0., 0.], [251., 2 * np.pi + 0.1, 0., 0.]]])
        prepared = prepare_estimation_data({"trajectory": trajectory, "t": np.array([0., 1.]), "L": 125.}, bins=4)
        np.testing.assert_allclose(prepared["Phi1_ast"], [0.])
        np.testing.assert_allclose(prepared["Theta1_ast"], [0.], atol=1e-14)
        np.testing.assert_allclose(prepared["dot_Phi1"], [2.])

    def test_invalid_integration_times_are_rejected(self):
        for arguments in (dict(dt=0), dict(dt=-1), dict(t_end=-1), dict(dt=0.3, t_end=1),
                          dict(dt=0.1, t_end=1, save_every=3), dict(save_every=0)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                simulate_dataset(self.model, **arguments)


if __name__ == "__main__":
    unittest.main()
