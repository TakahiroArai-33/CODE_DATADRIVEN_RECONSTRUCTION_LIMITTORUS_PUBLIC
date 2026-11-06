"""Small end-to-end checks for isolated Python runs of the sample notebook."""

from __future__ import annotations

from datetime import datetime
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
SINGLE = HERE / "run_test.py"
SWEEP = HERE / "run_sigma_sweep.py"


class BatchExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temporary = tempfile.TemporaryDirectory(prefix="sample-batch-tests-")
        cls.base = Path(cls._temporary.name).resolve()
        cls.env = os.environ.copy()
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS"):
            cls.env[name] = "1"
        for name, directory in (("NUMBA_CACHE_DIR", "numba"),
                                ("MPLCONFIGDIR", "matplotlib"),
                                ("XDG_CACHE_HOME", "cache")):
            cls.env[name] = str(cls.base / directory)

    @classmethod
    def tearDownClass(cls):
        cls._temporary.cleanup()

    def setUp(self):
        self.root = self.base / self._testMethodName
        self.root.mkdir()

    def notebook(self, *, fail_high_sigma=False, magic="%matplotlib inline"):
        """Exercise overrides and the real stochastic solver, without a long run."""
        source = f"""from pathlib import Path
import json
import sys
import numpy as np
sys.path.insert(0, {str(HERE)!r})
from spde_cpu import simulate
{magic}
TEST_DIR = Path('.')
DATA_DIR = TEST_DIR / 'data'
loaddata = True
SIGMA = 9.0
T_END = 8.0
SEED = 12345
DT = 0.01
EPSILON = 1e-6
SAVE_INTERVAL = 1.0
OUTPUT_DIR = TEST_DIR / 'fixture-should-not-be-created'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
"""
        calculation = """if loaddata:
    raise AssertionError('The runner did not disable loading saved data')
"""
        if fail_high_sigma:
            calculation += """if SIGMA > 5e-6:
    raise RuntimeError('intentional sigma-specific failure')
"""
        calculation += """initial = np.full((2, 2, 32), 0.2)
rx = np.linspace(0.0, 16.0, 33)
result = simulate(initial, rx, DT, 1, 1, epsilon=EPSILON,
                  sigma=SIGMA, seed=SEED, progress=False)
payload = {
    'sigma': SIGMA, 'seed': SEED, 't_end': T_END,
    'test_dir': str(TEST_DIR.resolve()),
    'data_dir': str(DATA_DIR.resolve()),
    'output_dir': str(OUTPUT_DIR.resolve()),
    'state': result['final_state'].ravel().tolist(),
}
(OUTPUT_DIR / 'fixture_result.json').write_text(json.dumps(payload))
"""
        notebook = {
            "nbformat": 4, "nbformat_minor": 5, "metadata": {},
            "cells": [{"cell_type": "code", "metadata": {}, "outputs": [],
                       "execution_count": None, "source": cell.splitlines(True)}
                      for cell in (source, calculation)],
        }
        path = self.root / "fixture.ipynb"
        path.write_text(json.dumps(notebook))
        return path

    def run_cli(self, script, *args, success=True):
        result = subprocess.run(
            [sys.executable, str(script), *(str(arg) for arg in args)],
            cwd=self.root, env=self.env, capture_output=True, text=True,
            timeout=180,
        )
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    @staticmethod
    def payload(output):
        return json.loads((output / "fixture_result.json").read_text())

    @staticmethod
    def residuals(payload):
        # Independent one-step drift for a spatially uniform u=v=0.2 state.
        u = 0.2 + 0.01 * (0.2**3 - (0.018 + 0.052) * 0.2)
        v = 0.2 + 0.01 * (-0.2**3 + 0.018 * (1.0 - 0.2))
        deterministic = [u] * 32 + [v] * 32 + [u] * 32 + [v] * 32
        return [actual - expected for actual, expected
                in zip(payload["state"], deterministic)]

    def test_single_run_overrides_and_reproducible_nonzero_noise(self):
        notebook = self.notebook()
        outputs = [self.root / "first", self.root / "repeat"]
        for output in outputs:
            self.run_cli(SINGLE, "--notebook", notebook, "--sigma", "1e-6",
                         "--t-end", "0.125", "--seed", "73",
                         "--output-dir", output)
        first, repeat = (self.payload(output) for output in outputs)
        self.assertEqual(first["sigma"], 1e-6)
        self.assertEqual(first["seed"], 73)
        self.assertEqual(first["t_end"], 0.125)
        self.assertEqual(Path(first["test_dir"]), HERE)
        self.assertEqual(Path(first["data_dir"]), HERE / "data")
        self.assertEqual(Path(first["output_dir"]), outputs[0])
        self.assertEqual(first["state"], repeat["state"])
        residuals = self.residuals(first)
        expected_std = 1e-6 * math.sqrt(0.01)
        self.assertTrue(all(value != 0.0 for value in residuals))
        self.assertAlmostEqual(statistics.pstdev(residuals), expected_std,
                               delta=0.3 * expected_std)
        metadata = json.loads((outputs[0] / "run_metadata.json").read_text())
        self.assertEqual(metadata["sigma"], 1e-6)

    def test_sweep_separates_outputs_and_preserves_noise_scaling(self):
        notebook = self.notebook()
        output = self.root / "sweep"
        self.run_cli(SWEEP, "--notebook", notebook,
                     "--sigmas", "1e-6", "1e-5", "--jobs", "2",
                     "--seed", "73", "--t-end", "0.125",
                     "--output-root", output)
        results = []
        for sigma in (1e-6, 1e-5):
            label = f"sigma_{sigma!r}"
            child = output / label
            payload = self.payload(child)
            self.assertEqual(payload["sigma"], sigma)
            self.assertEqual(payload["seed"], 73)
            self.assertEqual(payload["t_end"], 0.125)
            self.assertEqual(Path(payload["output_dir"]), child)
            self.assertTrue((output / "logs" / f"{label}.log").is_file())
            results.append(payload)
        # Identical seed, initial state and timestep: the increment scales with sigma.
        for lower, higher in zip(self.residuals(results[0]), self.residuals(results[1])):
            self.assertAlmostEqual(higher, 10.0 * lower, delta=5e-16)
        manifest = json.loads((output / "sweep.json").read_text())
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual({job["status"] for job in manifest["jobs"]}, {"completed"})
        self.assertEqual(len({job["pid"] for job in manifest["jobs"]}), 2)
        starts = [datetime.fromisoformat(job["started_at"]) for job in manifest["jobs"]]
        finishes = [datetime.fromisoformat(job["finished_at"]) for job in manifest["jobs"]]
        self.assertLess(max(starts), min(finishes), "The two child runs did not overlap")

    def test_one_failed_job_does_not_hide_other_results(self):
        notebook = self.notebook(fail_high_sigma=True)
        output = self.root / "sweep"
        self.run_cli(SWEEP, "--notebook", notebook,
                     "--sigmas", "1e-6", "1e-5", "--jobs", "2",
                     "--output-root", output, success=False)
        self.assertEqual(self.payload(output / "sigma_1e-06")["sigma"], 1e-6)
        self.assertFalse((output / "sigma_1e-05" / "fixture_result.json").exists())
        failed_log = (output / "logs" / "sigma_1e-05.log").read_text()
        self.assertIn("intentional sigma-specific failure", failed_log)
        manifest = json.loads((output / "sweep.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual({job["sigma"]: job["status"] for job in manifest["jobs"]},
                         {1e-6: "completed", 1e-5: "failed"})

    def test_existing_output_is_not_overwritten(self):
        notebook = self.notebook()
        for script, arguments in (
            (SINGLE, ["--sigma", "1e-6", "--output-dir"]),
            (SWEEP, ["--sigmas", "1e-6", "1e-5", "--jobs", "2", "--output-root"]),
        ):
            output = self.root / script.stem
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("existing results")
            self.run_cli(script, "--notebook", notebook, *arguments, output,
                         success=False)
            self.assertEqual(sentinel.read_text(), "existing results")
            self.assertEqual(list(output.iterdir()), [sentinel])

    def test_unsupported_magic_reports_failure(self):
        notebook = self.notebook(magic="%time print('not supported')")
        output = self.root / "unsupported"
        result = self.run_cli(SINGLE, "--notebook", notebook, "--sigma", "1e-6",
                              "--output-dir", output, success=False)
        self.assertRegex(result.stdout + result.stderr, r"(?i)magic|unsupported|syntax")
        self.assertFalse((output / "fixture_result.json").exists())


if __name__ == "__main__":
    unittest.main()
