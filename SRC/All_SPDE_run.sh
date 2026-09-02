#!/bin/bash
# All_SPDE_run.sh

# Submit multiple instances of wfl_run_SPDE.sh through qsub.
# Split the configuration list into NUM_CHUNKS chunks.
NUM_CHUNKS=10

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Supply cluster-specific GPU queue and resource options through the environment.
QSUB_GPU_ARGS=()
if [[ -n "${QSUB_GPU_OPTIONS:-}" ]]; then
  read -r -a QSUB_GPU_ARGS <<< "$QSUB_GPU_OPTIONS"
fi

rm -rf json 2>/dev/null || true
python "$SCRIPT_DIR/make_configs.py"

mkdir -p config_lists

# Remove stale chunks before generating the new job lists.
rm -f config_lists/chunk_* 2>/dev/null || true
# List and sort production JSON files only.
find json -maxdepth 1 -type f -name 'state*.json' ! -name '*debug*.json' | sort > config_lists/all_configs.txt
# Distribute the list across chunk_00, chunk_01, ... in round-robin order.
total_configs=$(wc -l < config_lists/all_configs.txt)
if (( total_configs == 0 )); then
  echo "[WARN] no production configs found (json/state*.json)" >&2
else
  split -d -n r/${NUM_CHUNKS} config_lists/all_configs.txt config_lists/chunk_
fi

# Submit one PBS job per nonempty chunk.
for chunk in config_lists/chunk_*; do
  [[ -s "$chunk" ]] || continue
  job=$(basename "$chunk")
  qsub "${QSUB_GPU_ARGS[@]}" -N "spe_${job}" \
      -v TARGET_LIST="$chunk",PROFILE_SPDE="${PROFILE_SPDE:-0}" \
      "$SCRIPT_DIR/wfl_run_SPDE.sh"
done
