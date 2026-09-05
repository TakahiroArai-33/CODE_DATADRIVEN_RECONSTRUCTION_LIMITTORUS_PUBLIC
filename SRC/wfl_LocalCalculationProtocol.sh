#!/bin/bash
# PBS workflow for a GPU cluster.
# Configure queue, GPU resources, and environment modules at submission time.
#PBS -r n
#PBS -l walltime=48:00:00


# Submit with: qsub SRC/wfl_LocalCalculationProtocol.sh
# LocalCalculationProtocol_GPU.sh currently uses linear phase interpolation.

set -euo pipefail
PROJECT_ROOT="${PBS_O_WORKDIR:?PBS_O_WORKDIR not set}"
export SRC_DIR="SRC"
cd "$PROJECT_ROOT"

mkdir -p stdout

# DirCopy.py uses this date in the archive directory name.
export RUNDATE=$(date +%Y%m%d)

source "$SRC_DIR/LocalCalculationProtocol_GPU.sh"

# Archive the results under a name selected by USE_KRALEMANN.
python "$SRC_DIR/DirCopy.py"
