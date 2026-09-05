#!/bin/bash
# PBS workflow for a GPU cluster.
# Configure queue, GPU resources, and environment modules at submission time.
#PBS -r n
#PBS -l walltime=48:00:00

# Submitted by All_SPDE_run_short.sh for short trajectories.

set -euo pipefail
PROJECT_ROOT="${PBS_O_WORKDIR:?PBS_O_WORKDIR not set}"
SRC_DIR="SRC"
cd "$PROJECT_ROOT"

mkdir -p stdout



# Process each JSON path listed in TARGET_LIST.
list="${TARGET_LIST:?TARGET_LIST not set}"
while IFS= read -r config || [[ -n "$config" ]]; do
  [[ -n "$config" ]] || continue

  # Ignore debug configurations in production jobs.
  if [[ "$config" == *debug*.json ]]; then
    echo "[$(date)] skip ${config} (debug config)"
    continue
  fi

  echo "[$(date)] start ${config}"

  # Skip configurations whose final output already exists.
  if python3 "$SRC_DIR/json_process.py" "${config}"; then
    echo "[$(date)] skip ${config} (already finished)"
    continue
  fi


  # A short trajectory requires one simulation segment.
  for i in {1..1}; do
    PROFILE_SPDE="${PROFILE_SPDE:-0}" singularity exec --home "$PWD" --nv CupyContainer.sif \
      python3 "$SRC_DIR/Simulate_CoupledModel.py" --config "${config}"
  done

  echo "[$(date)] done ${config}"
done < "$list"
