#!/bin/bash

# Compute phase time series from SPDE output, then run Bayesian inference.
# SRC/wfl_LocalCalculationProtocol.sh invokes this workflow on a GPU cluster.


# Shared workflow settings.
export USE_DEBUG_FILES=0 # 0: production; 1: debug data
export SPDE_DIR=CoupledModelSimulationDATA
export CONCATENATE_DIR=Concatenate_SPDEdatas
export INTERIM_DIR=Results/Interim
export PHASE_DIR=Results/Phase
export BAYES_DIR=Results/BayesianInference
export STDOUT_DIR=Results/Logs
export SECTION="7.0" # Poincare section

# Phase estimator: 0 for linear interpolation, 1 for the Kralemann method.
export USE_KRALEMANN=0

# Singularity command used for every Python stage.
PYTHON_CMD=(singularity exec --home "$PWD" --nv CupyContainer.sif python3)
SRC_DIR="${SRC_DIR:-SRC}"

run_python() {
    local script="$1"
    shift
    "${PYTHON_CMD[@]}" "$SRC_DIR/$script" "$@"
}
# Parameters must match available simulation data.
if [ "$USE_DEBUG_FILES" -eq 1 ]; then
    echo "Debug files mode" >&2
    sigma_list=(1.0e-7 2.5e-7 5.0e-7 1.0e-6 2.0e-6 4.0e-6)
    eps_list=(1e-6)
else
    echo "Normal mode" >&2
    # Alternative noise-strength sets retained for reproducibility:
    # sigma_list=(5.0e-7 1.0e-6 2.0e-6)
    # sigma_list=(1.0e-5 5.0e-5 1.0e-4)
    # sigma_list=(4.0e-6 2.0e-5 4.0e-5)
    sigma_list=(1.0e-6)
    eps_list=(1e-6)
fi

concatenate() {
    run_python Calculate_Concatenate_SPDEdatas.py
}

run_for_init_pairs() {
    # Run a command for every initial-state pair found by get_Init_idx.py.
    if ((${#@} == 0)); then
        echo "No command specified." >&2
        return 1
    fi

    local pairs=() line
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        pairs+=("$line")
    done < <("${PYTHON_CMD[@]}" "$SRC_DIR/get_Init_idx.py")

    if [[ ${#pairs[@]} -eq 0 ]]; then
        echo "No INIT pairs found; aborting." >&2
        return 1
    fi

    for pair in "${pairs[@]}"; do
        local INIT_A_IDX INIT_B_IDX
        IFS=' ' read -r INIT_A_IDX INIT_B_IDX <<<"$pair"
        export INIT_A_IDX INIT_B_IDX
        run_python "$@"
    done
}


run_phase_protocol() {
    local eps sigma

    for eps in "${eps_list[@]}"; do
        for sigma in "${sigma_list[@]}"; do
            export EPS_VAL=$eps 
            export SIGMA_VAL=$sigma 
            echo "-------------------------------------------------"
            echo "Using EPS_VAL=$EPS_VAL, SIGMA_VAL=$SIGMA_VAL" >&2
            echo "run for Calculate_Prep.py"
            run_for_init_pairs Calculate_Prep.py || return 1

            echo "run for Calculate_Theta.py"
            run_for_init_pairs Calculate_Theta.py || return 1

            echo "run for Calculate_c.py"
            run_python Calculate_c.py || return 1

            echo "run for Calculate_bi.py"
            run_for_init_pairs Calculate_bi.py || return 1

            echo "run for Calculate_Bfunc.py"
            run_python Calculate_Bfunc.py || return 1

            echo "run for Calculate_Phi.py"
            run_for_init_pairs Calculate_Phi.py || return 1
            
        done
    done
}


bayes_protocol(){
    local eps sigma

    for eps in "${eps_list[@]}"; do
        for sigma in "${sigma_list[@]}"; do
            export EPS_VAL=$eps
            export SIGMA_VAL=$sigma
            echo "-------------------------------------------------"
            echo "Using EPS_VAL=$EPS_VAL, SIGMA_VAL=$SIGMA_VAL" >&2
            echo "run for Bayes_Estimation_GPU.py"
            run_python Bayes_Estimation_GPU.py || return 1

        done
    done
}


# Clear derived outputs while preserving SPDE and concatenated data.
clear_output_dirs() {
    # rm -f $SPDE_DIR/*
    # rm -f $CONCATENATE_DIR/*
    rm -f $INTERIM_DIR/*
    rm -f $PHASE_DIR/*
    rm -f $BAYES_DIR/*
    rm -f $STDOUT_DIR/*
}


ensure_output_dirs() {
    mkdir -p -- "$SPDE_DIR"
    mkdir -p -- "$CONCATENATE_DIR"
    mkdir -p -- "$INTERIM_DIR"
    mkdir -p -- "$PHASE_DIR"
    mkdir -p -- "$BAYES_DIR"
    mkdir -p -- "$STDOUT_DIR"
    mkdir -p -- "Results/Diagnostics"
}




clear_output_dirs
ensure_output_dirs
concatenate
run_phase_protocol

# Run Bayesian inference with the full noise covariance.
export NOISE_COV_ZERO=0
echo "Running Bayesian inference with the full noise covariance."
bayes_protocol

# To use diagonal noise covariance instead, enable this block.
# export NOISE_COV_ZERO=1
# echo "Running Bayesian inference with diagonal noise covariance."
# bayes_protocol
