"""Checks that loading saved notebook results never recomputes or overwrites them."""

from __future__ import annotations

from contextlib import ExitStack
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


class PrecomputedLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="precomputed-loading-")
        cls.base = Path(cls.temporary.name).resolve()
        cls.environment = patch.dict(os.environ, {
            "MPLCONFIGDIR": str(cls.base / "matplotlib"),
            "XDG_CACHE_HOME": str(cls.base / "cache"),
        })
        cls.environment.start()
        import matplotlib
        matplotlib.use("Agg", force=True)

    @classmethod
    def tearDownClass(cls):
        cls.environment.stop()
        cls.temporary.cleanup()

    def setUp(self):
        self.output = self.base / self._testMethodName
        self.output.mkdir()
        self.t = np.arange(0.0, 30001.0, 1000.0)
        self.phase_t = self.t[1:-1]
        self.parameters = {
            "dt": 0.02, "epsilon": 2e-6, "sigma": 7e-6, "seed": 37,
            "save_interval": 1000.0, "domain_length": 250.0,
            "t_end": 0.06, "section": 6.5, "b_harmonics": 2,
            "c1": 0.04, "c2": 0.05, "simulation_seconds": 12.5,
            "Phi1_initial": 20.0, "Theta1_initial": 0.4,
            "Phi2_initial": 5.0, "Theta2_initial": 0.1,
            "b_method": "estimated", "b_source": "estimated_from_trajectory",
            "b_trim_cycles": 1, "b_offset1": 0.0, "b_offset2": 0.0,
            "b_fit_cycles1": 3, "b_fit_cycles2": 4,
        }
        self.csv("parameters.csv", list(self.parameters.items()), "name,value", fmt="%s")

        amplitude1 = np.exp(1j * self.t / 10000.0)
        amplitude2 = 2.0 * np.exp(1j * self.t / 12000.0)
        self.csv("trajectory.csv", np.column_stack((
            self.t, 7 + np.sin(self.t / 1000), 3 + np.cos(self.t / 1000),
            8 + np.sin(self.t / 1200), 4 + np.cos(self.t / 1200),
            amplitude1.real, amplitude1.imag, amplitude2.real, amplitude2.imag,
        )), "t,U1_bar,V1_bar,U2_bar,V2_bar,A1_tilde_real,A1_tilde_imag,A2_tilde_real,A2_tilde_imag")

        phi1 = np.linspace(-11.0, 19.0, self.phase_t.size)
        phi2 = np.linspace(4.0, -2.0, self.phase_t.size)
        theta1 = np.linspace(0.1, 5.0, self.phase_t.size)
        theta2 = np.linspace(0.2, 4.5, self.phase_t.size)
        self.theory = np.array((np.linspace(13.0, 35.0, self.phase_t.size),
                                np.linspace(-0.8, 0.9, self.phase_t.size)))
        self.phase = np.column_stack((
            self.phase_t, self.parameters["epsilon"] * self.phase_t,
            phi1 - phi2, theta1 - theta2, *self.theory,
            phi1, theta1, phi2, theta2,
        ))
        self.csv("phase_comparison.csv", self.phase,
                 "t,eps_t,delta_phi,delta_theta,theory_delta_phi,theory_delta_theta,Phi1,Theta1,Phi2,Theta2")

        rx = np.linspace(-125.0, 125.0, 9)[:-1]
        self.csv("final_state.csv", np.column_stack((rx, np.ones((8, 4)))),
                 "rx,U1,V1,U2,V2")
        self.b_theta = np.linspace(0.0, 2 * np.pi, 17)
        self.b1, self.b2 = np.sin(self.b_theta), 0.5 * np.cos(self.b_theta)
        self.csv("B_estimated.csv", np.column_stack((self.b_theta, self.b1, self.b2)),
                 "theta,B1,B2")
        self.csv("B_coefficients.csv", [[1, 0, 1, 0.5, 0], [2, 0, 0, 0, 0]],
                 "n,cos1,sin1,cos2,sin2")
        self.b_theory = 0.25 * np.sin(self.b_theta)
        self.csv("B_theory.csv", np.column_stack((self.b_theta, self.b_theory)),
                 "theta,B_theory")
        (self.output / "comparison.png").write_bytes(b"existing figure must be preserved")

    def tearDown(self):
        import matplotlib.pyplot as plt
        plt.close("all")

    def csv(self, name, values, header, fmt="%.17g"):
        np.savetxt(self.output / name, values, delimiter=",", header=header,
                   comments="", fmt=fmt)

    def snapshot(self):
        return {str(path.relative_to(self.output)):
                (path.stat().st_size, path.stat().st_mtime_ns,
                 hashlib.sha256(path.read_bytes()).hexdigest())
                for path in self.output.rglob("*") if path.is_file()}

    def test_loader_restores_saved_arrays_metadata_and_b_reference(self):
        from sample_io import load_precomputed_results
        before = self.snapshot()
        result = load_precomputed_results(self.output,
                                          fallback_b_theory=self.output / "not-present.csv")
        for key in ("dt", "epsilon", "sigma", "seed", "save_interval",
                    "domain_length", "t_end", "section", "b_harmonics", "c1", "c2"):
            self.assertEqual(result[key], self.parameters[key], key)
        self.assertEqual(result["L"], 125.0)
        np.testing.assert_array_equal(result["moments"]["t"], self.t)
        np.testing.assert_array_equal(result["phase"]["t"], self.phase_t)
        np.testing.assert_array_equal(result["phase"]["theory_delta_phi"], self.theory[0])
        np.testing.assert_array_equal(result["phase"]["theory_delta_theta"], self.theory[1])
        np.testing.assert_array_equal(result["B1"]["By"], self.b1)
        np.testing.assert_array_equal(result["B2"]["By"], self.b2)
        np.testing.assert_array_equal(result["b_theory"]["B_theory"], self.b_theory)
        self.assertEqual(result["moments"]["final_state"].shape, (2, 2, 8))
        self.assertEqual(result["rx"].shape, (9,))
        self.assertEqual(self.snapshot(), before)

    def test_legacy_b_reference_fallback_and_optional_coefficients(self):
        from sample_io import load_precomputed_results
        fallback = self.base / "legacy_B_theory.csv"
        (self.output / "B_theory.csv").replace(fallback)
        (self.output / "B_coefficients.csv").unlink()
        result = load_precomputed_results(self.output, fallback_b_theory=fallback)
        np.testing.assert_array_equal(result["b_theory"]["B_theory"], self.b_theory)
        np.testing.assert_array_equal(result["B1"]["By"], self.b1)

    def test_missing_saved_phase_data_fails_without_recomputation(self):
        from sample_io import load_precomputed_results
        (self.output / "phase_comparison.csv").unlink()
        before = self.snapshot()
        with self.assertRaisesRegex((FileNotFoundError, ValueError), "phase_comparison"):
            load_precomputed_results(self.output)
        self.assertEqual(self.snapshot(), before)

    def test_notebook_load_mode_uses_only_saved_results_and_never_writes(self):
        import phase_analysis
        import run_single_trajectory
        import spde_cpu
        from run_test import prepare_cells

        notebook = HERE / "Test.ipynb"
        overrides = {
            "TEST_DIR": HERE, "OUTPUT_DIR": self.output, "loaddata": True,
            "SIGMA": 999.0, "T_END": 999.0, "SEED": 999,
            "DT": 0.5, "EPSILON": 0.5, "SAVE_INTERVAL": 999.0,
            "SECTION": 999.0, "B_HARMONICS": 99,
        }
        cells = prepare_cells(notebook, notebook.read_bytes(), overrides)
        namespace = {"__name__": "__main__", "__file__": str(notebook),
                     "__run_overrides__": overrides}
        before = self.snapshot()
        with ExitStack() as stack:
            functions = [(spde_cpu, "simulate"),
                         (run_single_trajectory, "initial_states_from_phases")]
            functions.extend((phase_analysis, name) for name in (
                "calculate_Theta", "calculate_c", "calculate_b", "concatenate_theta_b",
                "fit_B_function", "calculate_Phi", "get_common_time", "solve_theory",
            ))
            for module, name in functions:
                stack.enter_context(patch.object(module, name,
                    side_effect=AssertionError(f"Load mode called {name}")))
            for _, _, code in cells:
                exec(code, namespace)

        self.assertEqual(namespace["SIGMA"], self.parameters["sigma"])
        self.assertEqual(namespace["DT"], self.parameters["dt"])
        self.assertEqual(namespace["EPSILON"], self.parameters["epsilon"])
        self.assertEqual(namespace["SEED"], self.parameters["seed"])
        self.assertEqual(namespace["SAVE_INTERVAL"], self.parameters["save_interval"])
        self.assertEqual(namespace["T_END"], self.parameters["t_end"])
        self.assertEqual(namespace["SECTION"], self.parameters["section"])
        self.assertEqual(namespace["B_HARMONICS"], self.parameters["b_harmonics"])
        self.assertEqual(namespace["c1"], self.parameters["c1"])
        self.assertEqual(namespace["c2"], self.parameters["c2"])
        for name, key in (("Phi_1", "Phi1_initial"), ("Phi_2", "Phi2_initial"),
                          ("Theta_1", "Theta1_initial"), ("Theta_2", "Theta2_initial")):
            self.assertEqual(namespace[name], self.parameters[key])
        for system in (1, 2):
            self.assertEqual(namespace[f"Phi_{system}_over_L"],
                             self.parameters[f"Phi{system}_initial"] / 125.0)
            self.assertEqual(namespace[f"Theta_{system}_over_pi"],
                             self.parameters[f"Theta{system}_initial"] / np.pi)
        self.assertEqual(namespace["simulation_seconds"], self.parameters["simulation_seconds"])
        np.testing.assert_array_equal(namespace["t_phase"], self.phase_t)
        np.testing.assert_array_equal(namespace["theory"], self.theory)
        np.testing.assert_array_equal(namespace["Phi1"], self.phase[:, 6])
        np.testing.assert_array_equal(namespace["Theta2"], self.phase[:, 9])
        np.testing.assert_array_equal(namespace["B1"]["By"], self.b1)
        np.testing.assert_array_equal(namespace["b_theory"]["B_theory"], self.b_theory)
        np.testing.assert_array_equal(namespace["x"], self.phase[:, 1])
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(namespace["comparison_fig"].axes[0].get_xlim(),
                         (0.0, self.parameters["t_end"]))


if __name__ == "__main__":
    unittest.main()
