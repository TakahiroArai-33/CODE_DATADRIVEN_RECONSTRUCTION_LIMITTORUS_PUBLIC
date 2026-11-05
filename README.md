This repository contains the code for the following paper:

* Takahiro Arai, Toshio Aoyagi, and Yoji Kawamura, “Data-driven reconstruction of spatiotemporal phase dynamics for traveling and oscillating patterns via Bayesian inference,” [arXiv:2604.23727](https://doi.org/10.48550/arXiv.2604.23727).

This work was supported by JSPS KAKENHI Grant Numbers JP24H00723, JP24K06910, JP25H01474, JP25K01160, JP25K22831, JP26H00477, JP26H02349, and JP26K21344.

Numerical simulations were performed on the Earth Simulator (ES4) at the
Japan Agency for Marine-Earth Science and Technology (JAMSTEC).


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

### 4. Run phase calculation and Bayesian inference

Submit the following wfl script with `qsub`, adding any queue and GPU resource
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



### 6. Notes on GPU usage and computational time

#### SPDE

Generating the SPDE data is the most time-consuming step. With the current parameters, each long trajectory takes approximately 7 hours on one GPU. In our experiments, generating 64 long trajectories for a single value of σ took approximately 2 days, with 10 GPU jobs running in parallel using the PBS job scheduler.

Although we used NVIDIA A100 GPUs in our experiments, each SPDE simulation uses only 1,024 distinct spatial grid points. Less powerful GPUs may also be suitable, although their performance has not been evaluated. The SPDE simulations could, in principle, also be performed on a CPU.

#### Phase calculation and Bayesian inference

Temporal phases are obtained by linear interpolation on the CPU. The Appendix B method uses GPUs for the Hilbert transform and Fourier-based phase conversion.

GPUs are also used for data subsampling when preparing the dataset for Bayesian inference and for large-scale tensor contractions and summations during the iterative updates.
The iterative Bayesian updates took approximately 100 seconds in total for both systems (three updates per system), excluding data loading, preprocessing, basis construction, and output generation. Data subsampling was not timed separately.

Overall, data processing and Bayesian inference, including file input and output, are expected to take approximately two hours per value of σ.


----

# Sample code for calculation of the phase

Reproducing the full results is computationally intensive and generates large amounts of data. 
The workflow described above uses GPUs and the PBS job scheduler.

For a smaller example, [`SampleCode/Test.ipynb`](SampleCode/Test.ipynb) calculates the temporal phase Θ and spatial phase Φ from a single trajectory of two coupled systems and compares the results with theory. It runs on a CPU without PBS or a GPU.

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



---

# License

This project is licensed under the [MIT License](LICENSE).

Copyright (c) 2026 Takahiro Arai.
