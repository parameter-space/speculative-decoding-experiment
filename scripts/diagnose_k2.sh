#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/seraph_common.sh"
conda activate "$S1_ENV"
: "${S1_DATA_DIR:?Set S1_DATA_DIR to the existing compute-local dataset directory.}"
: "${CUDA_VISIBLE_DEVICES:?Run inside your existing GPU allocation.}"
DATA_RESOLVED="$(cd -- "$S1_DATA_DIR" && pwd -P)"
case "$DATA_RESOLVED" in
  /local_datasets/leetj3610/*|/data2/local_datasets/leetj3610/*) ;;
  *) printf '%s\n' 'STOP: dataset must resolve to your permitted local SSD directory.' >&2; exit 1 ;;
esac
test -f "$DATA_RESOLVED/manifest.json"
# Select only the first GPU from Slurm's existing visibility; never assume a physical GPU ID.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE="$DATA_RESOLVED/cache"
mkdir -p "$PROJECT_DIR/runs"
RUN_DIR="$(mktemp -d "$PROJECT_DIR/runs/diagnose-${SLURM_JOB_ID}-XXXXXX")"
printf 'Diagnostic directory: %s\n' "$RUN_DIR"
python -u -m signal_study.diagnose --data-dir "$DATA_RESOLVED" --output "$RUN_DIR/probe" 2>&1 | tee "$RUN_DIR/console.log"
printf 'Report: %s/probe/diagnostic.json\n' "$RUN_DIR"
