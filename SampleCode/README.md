# Running the sample from Python

Install `requirements.txt` in your Python environment. The scripts below run
the code cells in `Test.ipynb` directly, using its simulation settings and
the same calculations of the phases. They do not require Jupyter.

Run two noise amplitudes concurrently from the repository root:

```bash
python SampleCode/run_sigma_sweep.py --sigmas 1e-6 1e-5 --jobs 2
```

Each process runs one coupled trajectory on the CPU. Unless overridden,
the notebook settings apply, including `T_END = 8.0`, `EPSILON = 1e-6`,
`DT = 0.01`, `SAVE_INTERVAL = 1.0`, and `SEED = 12345`.
The same seed means that different noise amplitudes use the same underlying
standardized random sequence, scaled by sigma.

Results are written to a new folder under `SampleCode/output_runs/`:

```text
<sweep-folder>/
  Test.executed.ipynb
  sweep.json
  logs/
    sigma_1e-06.log
    sigma_1e-05.log
  sigma_1e-06/
    trajectory.csv
    final_state.csv
    parameters.csv
    phase_comparison.csv
    B_estimated.csv
    B_coefficients.csv
    B_samples.csv
    timeseries.png
    comparison.png
    B_estimation.png
    run_metadata.json
  sigma_1e-05/
    ...
```

`Test.executed.ipynb` is a snapshot of the input notebook; execution outputs
are CSV and PNG files. `sweep.json` records process IDs, output paths, logs,
and completion status. Each `run_metadata.json` records the notebook hash
and the parameters used. Existing output folders are protected from overwriting.

To choose the destination or override the duration and seed:

```bash
python SampleCode/run_sigma_sweep.py --sigmas 1e-6 1e-5 --jobs 2 \
  --output-root SampleCode/output_runs/my_sweep --t-end 8 --seed 12345
```

For a single run:

```bash
python SampleCode/run_test.py --sigma 1e-6 \
  --output-dir SampleCode/output_runs/single_sigma_1e-06
```

The notebook's time-series figure uses the window `t = 20000` to `30000`.
Keep the simulated duration beyond that window when running its full workflow.
All sampled rows and the notebook's CSV precision are preserved.
