This repository contains the code for the following paper:

* Takahiro Arai, Toshio Aoyagi, and Yoji Kawamura, “Data-driven reconstruction of spatiotemporal phase dynamics for traveling and oscillating patterns via Bayesian inference,” [arXiv:2604.23727](https://doi.org/10.48550/arXiv.2604.23727).

This work was supported by JSPS KAKENHI Grant Numbers JP24H00723, JP24K06910, JP25H01474, JP25K01160, JP25K22831, JP26H00477, JP26H02349, and JP26K21344.

Numerical simulations were performed on the Earth Simulator (ES4) at the
Japan Agency for Marine-Earth Science and Technology (JAMSTEC).


## Contents

- [GPU Workflow](#gpu-workflow)
  - [Environment](#environment)
  - [Execution](#execution)
    - [1. Configure the PBS job scripts](#1-configure-the-pbs-job-scripts)
      - [Set options in the scripts](#set-options-in-the-scripts)
      - [Set options with an environment variable](#set-options-with-an-environment-variable)
    - [2. Build the GPU container](#2-build-the-gpu-container)
    - [3. Generate the SPDE data](#3-generate-the-spde-data)
    - [4. Calculate the phases and perform Bayesian inference](#4-calculate-the-phases-and-perform-bayesian-inference)
    - [5. Generate the output](#5-generate-the-output)
    - [6. Notes on GPU usage, computation time, and data storage](#6-notes-on-gpu-usage-computation-time-and-data-storage)
      - [SPDE simulations](#spde-simulations)
      - [Calculation of phases and Bayesian inference](#calculation-of-phases-and-bayesian-inference)
- [Sample code for calculating phases in a local environment](#sample-code-for-calculating-phases-in-a-local-environment)
  - [Using precomputed data](#using-precomputed-data)
- [License](#license)


# GPU Workflow

## Environment

The computational environment used in this work is summarized below.

| Component | Details |
| --- | --- |
| System | [Earth Simulator (ES4)](https://www.jamstec.go.jp/es/en/) at JAMSTEC |
| GPU | NVIDIA A100; one GPU per SPDE job |
| GPU library | CuPy 13.6.0 |
| Execution | Singularity container and PBS batch jobs |

GPU computations use `CupyContainer.sif`, a Singularity image built from
[`cupy/cupy:v13.6.0`](https://hub.docker.com/r/cupy/cupy/?tag=v13.6.0).
The `SRC/wfl_*.sh` scripts are submitted as PBS jobs and run the container with
`singularity exec --nv`.

The plotting script was tested with:

- Python 3.10.18
- NumPy 1.26.4
- SciPy 1.12.0
- Matplotlib 3.10.6

## Execution

The current parameters are:

- Coupling strength: `epsilon = 1.0e-6`
- Noise strength: `sigma = 1.0e-6` only



### 1. Configure the PBS job scripts

The `SRC/wfl_*.sh` files are PBS job scripts used to run GPU computations.
Before submitting them, configure the PBS section (the `#PBS` directives) for
your execution environment, including the queue, wall time, GPU resources, and
account/project settings required by your cluster.

Adjust the `qsub` commands and options in `SRC/All_SPDE_run.sh` and
`SRC/All_SPDE_run_short.sh` to match your computing environment. If you only
need to add `qsub` options, the following examples show two ways to specify
them. Both examples use the PBS `select` and `ngpus` options.

#### Set options in the scripts

Edit `QSUB_GPU_ARGS` in each script:

```bash
# Replace GPU_QUEUE_NAME, adjust resources, and replace or remove <OTHER_OPTIONS>.
QSUB_GPU_ARGS=(-q GPU_QUEUE_NAME -l select=1:ngpus=1 "<OTHER_OPTIONS>")
```

#### Set options with an environment variable

You can also set `QSUB_GPU_OPTIONS` in the shell before running the scripts:

```bash
# Replace GPU_QUEUE_NAME, adjust resources, and replace or remove <OTHER_OPTIONS>.
export QSUB_GPU_OPTIONS='-q GPU_QUEUE_NAME -l select=1:ngpus=1 <OTHER_OPTIONS>'
```

A nonempty `QSUB_GPU_OPTIONS` overrides `QSUB_GPU_ARGS` in both scripts.

### 2. Build the GPU container

Run the following command from the project root:

```bash
singularity build CupyContainer.sif docker://cupy/cupy:v13.6.0
```

### 3. Generate the SPDE data

SPDE simulations use multiple GPUs by running independent simulation jobs in
parallel, with one GPU per job. Edit `NUM_CHUNKS` (currently `10`) at the top of
`SRC/All_SPDE_run.sh` and `SRC/All_SPDE_run_short.sh` to set the number of job
batches submitted by each script.

```bash
bash SRC/All_SPDE_run.sh
bash SRC/All_SPDE_run_short.sh
```

`SRC/All_SPDE_run.sh` submits `SRC/wfl_run_SPDE.sh`, and
`SRC/All_SPDE_run_short.sh` submits `SRC/wfl_run_SPDE_short.sh`.
Wait for all PBS jobs to finish before continuing.

### 4. Calculate the phases and perform Bayesian inference

Submit the following PBS job script with `qsub`, adding any queue and GPU resource
options required by the cluster:

```bash
qsub SRC/wfl_LocalCalculationProtocol.sh
```

`SRC/wfl_LocalCalculationProtocol.sh` calls
`SRC/LocalCalculationProtocol_GPU.sh`.

### 5. Generate the output

After the phase-calculation and Bayesian-inference job finishes, run:

```bash
python SRC/Plot_Estimation_Result.py
```

By default, the script reads the result generated in
`Results/BayesianInference`. Edit `BAYES_FILES` in
`SRC/Plot_Estimation_Result.py` only when plotting results stored elsewhere.



### 6. Notes on GPU usage, computation time, and data storage

Although our code uses GPUs, the method itself does not inherently require a GPU. 
The most computationally demanding step is data generation through SPDE simulations, as a large number of long trajectories are required.
The method we proposed in the above reference has a relatively low computational cost.
Below, we summarize the hardware used, computation times, and storage requirements.

Storing the generated time series and phase data for all values of $\sigma$ used in this study requires approximately 1 TB of storage.

#### SPDE simulations

With the parameter settings used in this study, generating a single long trajectory took approximately 7 hours on one NVIDIA A100 GPU. 
Generating 64 long trajectories for a single value of σ took approximately 2 days, with 10 GPU jobs running in parallel under the PBS job scheduler.

Each SPDE simulation uses a spatial grid with 1,024 distinct points. Less powerful GPUs may also be suitable, although their performance has not been evaluated. In principle, these simulations could also be performed on a CPU.

The saved data require approximately 700 MB per long trajectory, or approximately 45 GB for all 64 trajectories at each value of σ. 
The stored time series contain $\tilde{A}_i$ and $\bar{u}_i$, with a sampling interval of 1.0; the full spatial profiles of $X_i(x,t)$ are not saved at every sampling time.

#### Calculation of phases and Bayesian inference

The phases are calculated on the CPU, except when the method described in Appendix B is used for temporal phases. In that case, our implementation uses GPUs to compute the Hilbert transform and perform the Fourier-based phase conversion.
The phase data require approximately the same amount of storage as the saved $\tilde{A}_i$ and $\bar{u}_i$ data.

GPUs are also used to subsample the data when preparing the dataset for Bayesian inference and to perform large-scale tensor contractions and summations during the iterative updates.

The Bayesian updates took approximately 100 seconds in total across both systems, with three updates per system. This timing excludes data loading, preprocessing, basis construction, and output generation. Data subsampling was not timed separately.

Once the SPDE data have been generated, the complete data-processing and Bayesian-inference workflow, including file input and output, is expected to take approximately 2 hours per value of σ.


----

# Sample code for calculating phases in a local environment

Reproducing the full results reported in the paper cited above requires substantial computation time and a large amount of storage space.

For a smaller example, [`SampleCode/Test.ipynb`](SampleCode/Test.ipynb) calculates the temporal phase Θ and spatial phase Φ from a single trajectory of two coupled systems and compares the results with theory. 
It runs on a CPU without PBS or a GPU.

The reference environment for this notebook is:

- Python 3.10.18
- NumPy 1.26.4
- SciPy 1.12.0
- Numba 0.59.1
- Matplotlib 3.10.6
- tqdm 4.67.1
- JupyterLab 4.4.6
- ipykernel 6.30.1
- ipywidgets 8.1.7

`SampleCode/data/` contains the data required to run the notebook, including the limit-torus solution and data for the phase equations derived using the adjoint method [Arai and Kawamura, Phys. Rev. Research 7, 043079 (2025)].
`SampleCode/output/` contains results from a previous run. 
Re-running the notebook will overwrite output files with the same names.


## Using precomputed data

Precomputed data are also available on Zenodo.

- Precomputed sample data for “Data-driven reconstruction of spatiotemporal phase dynamics for traveling and oscillating patterns via Bayesian inference” (tentative)


To use them, download and extract the files into `SampleCode/output/` and set the following in the notebook:

```python
# Note: running with loaddata = False will overwrite existing output files.
loaddata = True
```

When `loaddata = True`, the SPDE simulation is skipped, and the data stored in `output/` are loaded and plotted.
The saved parameters are restored, and the saved phases and theoretical predictions are used directly.

When `loaddata = False`, the SPDE simulation is run, and the existing files in `output/` are overwritten.


---

# License

This project is licensed under the [MIT License](LICENSE).

Copyright (c) 2026 Takahiro Arai.
