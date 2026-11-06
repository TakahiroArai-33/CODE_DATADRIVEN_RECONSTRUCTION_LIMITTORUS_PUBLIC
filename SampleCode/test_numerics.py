"""Numerical checks for one noisy realization of two coupled systems.

Run: python -m unittest discover -s 小規模テスト -p 'test_*.py' -v
"""

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from phase_analysis import calculate_phases, solve_theory, wrap_period
from run_single_trajectory import initial_conditions, initial_states_from_phases
from spde_cpu import simulate


def numpy_drift(state, h, epsilon):
    """Independent vectorized centered difference and simultaneous coupling."""
    u, v = state[:, 0], state[:, 1]
    reaction = u * u * v
    laplacian = (np.roll(state, 1, axis=2) - 2 * state
                 + np.roll(state, -1, axis=2)) / h**2
    drift = laplacian * np.array([1.0, 1.9])[None, :, None]
    drift[:, 0] += reaction - (0.018 + 0.052) * u
    drift[:, 1] += -reaction + 0.018 * (1 - v)
    return drift + epsilon * (state[::-1] - state)


class InitialPhaseTests(unittest.TestCase):
    def setUp(self):
        # Distinct temporal rows and spatial values reveal both phase shifts.
        u = 10 * np.arange(5)[:, None] + np.arange(8)[None, :]
        u = np.column_stack((u, u[:, 0]))
        self.data = {"rx": np.linspace(-4, 4, 9), "t": np.arange(5.0),
                     "U": u, "V": u + 100}

    def test_both_systems_start_at_their_own_nearest_phase_grid(self):
        with patch("run_single_trajectory.load_x0", return_value=self.data):
            state, rx, info = initial_states_from_phases(
                "unused", 2.6, 0.45 * np.pi, -1.2, 1.4 * np.pi)
        expected_u = np.array([[15, 16, 17, 10, 11, 12, 13, 14],
                               [31, 32, 33, 34, 35, 36, 37, 30]])
        np.testing.assert_array_equal(state[:, 0], expected_u)
        np.testing.assert_array_equal(state[:, 1], expected_u + 100)
        np.testing.assert_array_equal(rx, self.data["rx"])
        self.assertEqual((info["phi1"], info["phi2"]), (3.0, 7.0))
        self.assertEqual((info["theta1"], info["theta2"]), (np.pi / 2, 3 * np.pi / 2))
        self.assertEqual(info["delta_phi"], -4.0)
        self.assertEqual(info["delta_theta"], -np.pi)

    def test_initial_phases_wrap_independently_in_both_directions(self):
        with patch("run_single_trajectory.load_x0", return_value=self.data):
            original = initial_states_from_phases("unused", 2.6, 0.45 * np.pi, -1.2, 1.4 * np.pi)
            wrapped = initial_states_from_phases("unused", 10.6, 2.45 * np.pi, -17.2, -2.6 * np.pi)
            near_endpoint = initial_states_from_phases("unused", 7.9, -0.1, -8.1, 2 * np.pi + 0.1)
        np.testing.assert_array_equal(original[0], wrapped[0])
        self.assertEqual(original[2], wrapped[2])
        for system in (0, 1):
            np.testing.assert_array_equal(near_endpoint[0][system, 0], np.arange(8))

    def test_legacy_default_keeps_second_system_at_zero_phase(self):
        with patch("run_single_trajectory.load_x0", return_value=self.data):
            state, _, info = initial_conditions("unused")
        np.testing.assert_array_equal(state[0, 0], [26, 27, 20, 21, 22, 23, 24, 25])
        np.testing.assert_array_equal(state[1, 0], np.arange(8))
        self.assertEqual(info["delta_phi"], 2.0)
        self.assertEqual(info["delta_theta"], np.pi)
        self.assertEqual((info["lateral_index"], info["vertical_index"]), (2, 2))


class SpdeNumericsTests(unittest.TestCase):
    def setUp(self):
        self.rx = np.linspace(-8, 8, 33)
        self.initial = np.random.default_rng(17).uniform(0.05, 0.4, (2, 2, 32))

    def test_euler_matches_independent_numpy_step(self):
        dt, epsilon = 0.01, 0.003
        h = self.rx[1] - self.rx[0]
        expected = self.initial + dt * numpy_drift(self.initial, h, epsilon)
        result = simulate(self.initial, self.rx, dt, 1, 1, epsilon=epsilon)
        np.testing.assert_allclose(result["final_state"], expected, atol=1e-15, rtol=1e-13)

    def test_periodic_shift_equivariance_and_fourier_sign(self):
        shift, dt = 7, 0.01
        base = simulate(self.initial, self.rx, dt, 7, 3, epsilon=0.003)
        shifted = simulate(np.roll(self.initial, shift, axis=2), self.rx,
                           dt, 7, 3, epsilon=0.003)
        np.testing.assert_allclose(shifted["final_state"],
                                   np.roll(base["final_state"], shift, axis=2), atol=1e-14)
        np.testing.assert_allclose(shifted["U1_bar"], base["U1_bar"], atol=1e-14)
        np.testing.assert_allclose(shifted["A1_tilde"],
                                   base["A1_tilde"] * np.exp(2j * np.pi * shift / 32), atol=1e-14)
        np.testing.assert_allclose(base["t"], [0, 0.03, 0.06, 0.07])

    def test_euler_coupling_cancels_in_sum_of_two_fields(self):
        # Equal-and-opposite coupling cancels cellwise in a simultaneous Euler step.
        coupled = simulate(self.initial, self.rx, 0.01, 1, 1,
                           epsilon=0.03)["final_state"]
        free = simulate(self.initial, self.rx, 0.01, 1, 1,
                        epsilon=0)["final_state"]
        np.testing.assert_allclose(coupled.sum(axis=0), free.sum(axis=0), atol=1e-15)

    def test_noise_seed_reproducibility_and_input_ownership(self):
        initial_copy = self.initial.copy()
        first = simulate(self.initial, self.rx, 0.01, 11, 5, sigma=0.01, seed=23)
        again = simulate(self.initial, self.rx, 0.01, 11, 5, sigma=0.01, seed=23)
        other = simulate(self.initial, self.rx, 0.01, 11, 5, sigma=0.01, seed=24)
        np.testing.assert_array_equal(first["final_state"], again["final_state"])
        self.assertFalse(np.array_equal(first["final_state"], other["final_state"]))
        np.testing.assert_array_equal(self.initial, initial_copy)
        self.assertFalse(np.shares_memory(first["final_state"], self.initial))

    def test_progress_preserves_noise_and_sampling_across_updates(self):
        dt, n_steps, save_stride = 0.01, 20003, 17
        initial_copy = self.initial.copy()
        kwargs = dict(epsilon=0.003, sigma=0.01, seed=23, noise_block=4)
        without_progress = simulate(self.initial, self.rx, dt, n_steps, save_stride,
                                    progress=False, **kwargs)
        with patch("spde_cpu.tqdm") as progress_bar:
            with_progress = simulate(self.initial, self.rx, dt, n_steps, save_stride,
                                     progress=True, **kwargs)
            bar = progress_bar.return_value.__enter__.return_value
            increments = [call.args[0] for call in bar.update.call_args_list]
            self.assertEqual(increments, [10000, 10000, 3])
            self.assertEqual(sum(increments), n_steps)
            self.assertEqual(progress_bar.call_args.kwargs["total"], n_steps)
            self.assertFalse(progress_bar.call_args.kwargs["disable"])

        self.assertEqual(with_progress.keys(), without_progress.keys())
        for name in without_progress:
            np.testing.assert_array_equal(with_progress[name], without_progress[name],
                                          err_msg=name)
        expected_steps = np.append(np.arange(0, n_steps + 1, save_stride), n_steps)
        np.testing.assert_array_equal(with_progress["t"], expected_steps * dt)
        np.testing.assert_array_equal(self.initial, initial_copy)


class PhaseNumericsTests(unittest.TestCase):
    def test_synthetic_poincare_b_recovery_across_spatial_wraps(self):
        length = 10.0
        t = np.arange(2401) * 0.005
        theta = [np.pi * t, np.pi * (t - 0.5)]
        phi = [0.4 + 3.0 * t, -0.8 + 1.2 * t]
        bx = np.linspace(0, 2 * np.pi, 1001)
        by = -0.3 * np.cos(bx) + 0.2 * np.sin(bx)
        moments = {"t": t}
        for system in (1, 2):
            temporal = theta[system - 1]
            b = np.interp(temporal % (2 * np.pi), bx, by)
            moments[f"U{system}_bar"] = 7.0 + np.sin(temporal)
            moments[f"A{system}_tilde"] = np.exp(2j * np.pi * (phi[system - 1] + b) / length)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "b.pickle"
            with path.open("wb") as stream:
                pickle.dump({"Bx": bx, "By": by}, stream)
            result = calculate_phases(moments, path, section=7.0, domain_length=length)
        mask = result["mask"]
        self.assertGreater(np.ptp(phi[0][mask]), 2 * length)
        for system in (1, 2):
            np.testing.assert_allclose(result[f"Phi{system}"], phi[system - 1][mask], atol=2e-12)
            error = wrap_period(result[f"Theta{system}"] - theta[system - 1][mask], 2 * np.pi)
            np.testing.assert_allclose(error, 0, atol=2e-12)
        np.testing.assert_allclose(result["delta_phi"],
                                   wrap_period((phi[0] - phi[1])[mask], length), atol=2e-12)
        np.testing.assert_allclose(result["delta_theta"], np.pi / 2, atol=2e-12)

    def test_constant_coupling_uses_elapsed_physical_time(self):
        t = np.array([1000.0, 1001.0, 1004.0, 1020.0])
        epsilon = 0.2
        initial = np.array([4.5, -2.0])
        gamma = np.array([2.3, -0.7])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "coupling.npz"
            np.savez(path, delta_phi=np.linspace(-5, 5, 5),
                     delta_theta=np.linspace(-np.pi, np.pi, 5),
                     Gamma_s=np.full((5, 5), gamma[0]), Gamma_t=np.full((5, 5), gamma[1]))
            actual = solve_theory(t, initial, path, epsilon=epsilon, domain_length=10)
            frozen = solve_theory(t, initial, path, epsilon=0, domain_length=10)
        expected = initial[:, None] + gamma[:, None] * epsilon * (t - t[0])
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
        np.testing.assert_array_equal(frozen, np.repeat(initial[:, None], t.size, axis=1))

    def test_coupling_csv_and_npz_produce_the_same_trajectory(self):
        phi = np.linspace(0, 10, 9)
        theta = np.linspace(0, 2 * np.pi, 9)
        gs = np.sin(theta[:, None]) * np.cos(2 * np.pi * phi[None, :] / 10)
        gt = 0.2 * np.cos(theta[:, None]) * np.sin(2 * np.pi * phi[None, :] / 10)
        t = np.linspace(100, 120, 21)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "coupling"
            npz_path = base.with_suffix(".npz")
            csv_path = base.with_name("coupling_axes.csv")
            np.savez(npz_path, delta_phi=phi, delta_theta=theta, Gamma_s=gs, Gamma_t=gt)
            np.savetxt(csv_path, np.column_stack((theta, phi)), delimiter=",",
                       header="delta_theta,delta_phi", comments="", fmt="%.17g")
            np.savetxt(base.with_name("coupling_Gamma_s.csv"), gs, delimiter=",", fmt="%.17g")
            np.savetxt(base.with_name("coupling_Gamma_t.csv"), gt, delimiter=",", fmt="%.17g")
            expected = solve_theory(t, [2.0, 1.0], npz_path, epsilon=0.03, domain_length=10)
            actual = solve_theory(t, [2.0, 1.0], csv_path, epsilon=0.03, domain_length=10)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


class EstimatedBTests(unittest.TestCase):
    @staticmethod
    def known_b(theta, system):
        if system == 1:
            return 0.3 * np.cos(theta) - 0.2 * np.sin(theta) + 0.08 * np.cos(2 * theta)
        return -0.15 * np.cos(theta) + 0.11 * np.sin(2 * theta) - 0.04 * np.cos(3 * theta)

    def synthetic_moments(self, dt=0.004, origin=0.0, duration=20.0):
        elapsed = np.arange(int(round(duration / dt)) + 1) * dt
        moments = {"t": origin + elapsed}
        known_phi = {}
        for system, offset, speed, phase_offset in (
            (1, 0.4, 3.0, 0.31), (2, -0.8, 1.2, -0.73),
        ):
            theta = np.pi * elapsed + phase_offset
            phi = offset + speed * elapsed
            moments[f"U{system}_bar"] = 7.0 + np.sin(theta)
            moments[f"A{system}_tilde"] = np.exp(
                2j * np.pi * (phi + self.known_b(theta, system)) / 10.0
            )
            known_phi[system] = phi
        return moments, known_phi

    def test_estimates_separate_b_and_c_from_one_trajectory_without_file(self):
        moments, known_phi = self.synthetic_moments()
        with patch("phase_analysis._load_b_function", side_effect=AssertionError("No reference B allowed")):
            result = calculate_phases(moments, domain_length=10, b_harmonics=3)
        self.assertEqual(result["b_method"], "estimated")
        self.assertGreater(np.ptp(known_phi[1][result["mask"]]), 4 * 10.0)
        for system, speed in ((1, 3.0), (2, 1.2)):
            self.assertAlmostEqual(result[f"c{system}"], speed, delta=2e-6)
            grid, values = result[f"Bx{system}"], result[f"By{system}"]
            np.testing.assert_allclose(values, self.known_b(grid, system), atol=2e-5)
            self.assertAlmostEqual(float(np.trapz(values, grid) / (2 * np.pi)), 0.0, delta=1e-12)
            self.assertAlmostEqual(float(values[0]), float(values[-1]), delta=1e-12)
            np.testing.assert_allclose(result[f"Phi{system}"],
                                       known_phi[system][result["mask"]], atol=2e-5)
            self.assertGreater(result[f"b_fit_cycles{system}"], 1)
        np.testing.assert_allclose(result["delta_phi"],
                                   wrap_period((known_phi[1] - known_phi[2])[result["mask"]], 10), atol=4e-5)

    def test_estimated_b_is_independent_of_time_origin_and_sampling_refinement(self):
        coarse, _ = self.synthetic_moments(dt=0.004)
        translated, _ = self.synthetic_moments(dt=0.004, origin=1000.0)
        fine, _ = self.synthetic_moments(dt=0.002)
        results = [calculate_phases(data, domain_length=10, b_harmonics=3)
                   for data in (coarse, translated, fine)]
        for system in (1, 2):
            for key in (f"By{system}", f"c{system}"):
                np.testing.assert_allclose(results[0][key], results[1][key], atol=1e-9, rtol=0)
                np.testing.assert_allclose(results[0][key], results[2][key], atol=2e-5, rtol=0)
            np.testing.assert_allclose(results[0][f"Phi{system}"],
                                       results[1][f"Phi{system}"], atol=1e-9, rtol=0)

    def test_estimation_rejects_insufficient_complete_cycles(self):
        moments, _ = self.synthetic_moments(duration=4.5)
        with self.assertRaisesRegex(ValueError, "[Cc]ycl|crossing|t_end"):
            calculate_phases(moments, domain_length=10, b_harmonics=3)

    def test_estimation_rejects_aliased_fourier_fit(self):
        t = np.arange(41) * 0.5
        moments = {"t": t}
        for system in (1, 2):
            moments[f"U{system}_bar"] = np.resize(np.array([7.0, 8.0, 7.0, 6.0]), t.size)
            moments[f"A{system}_tilde"] = np.exp(2j * np.pi * (0.4 + 0.2 * t) / 10)
        with self.assertRaisesRegex(ValueError, "rank|independent|harmonic|sample"):
            calculate_phases(moments, domain_length=10, b_harmonics=3)


if __name__ == "__main__":
    unittest.main()
