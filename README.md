This repository contains the code for the following paper:

* Takahiro Arai, Toshio Aoyagi, and Yoji Kawamura, 
“Data-driven reconstruction of spatiotemporal phase dynamics for traveling and oscillating patterns via Bayesian inference,” 
[arXiv:2604.23727](https://doi.org/10.48550/arXiv.2604.23727).


# GPU Workflow

## Environment

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

### 2. Build the GPU container

Run the following command from the project root:

```bash
singularity build CupyContainer.sif docker://cupy/cupy:v13.6.0
```

### 3. Generate the SPDE data

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
