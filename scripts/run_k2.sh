#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/seraph_common.sh"
conda activate "$S1_ENV"
: "${S1_DATA_DIR:?Set S1_DATA_DIR to your permitted compute-local dataset directory.}"
case "$S1_DATA_DIR" in
  /local_datasets/leetj3610/*|/data2/local_datasets/leetj3610/*) ;;
  *) printf '%s\n' 'STOP: dataset path must be inside your permitted local SSD directory.' >&2; exit 1 ;;
esac
mkdir -p -- "$S1_DATA_DIR"
DATA_RESOLVED="$(cd -- "$S1_DATA_DIR" && pwd -P)"
case "$DATA_RESOLVED" in
  /local_datasets/leetj3610/*|/data2/local_datasets/leetj3610/*) ;;
  *) printf '%s\n' 'STOP: dataset path resolves outside the local SSD directory.' >&2; exit 1 ;;
esac
export HF_DATASETS_CACHE="$DATA_RESOLVED/cache"
if [[ ! -f "$DATA_RESOLVED/manifest.json" ]]; then
  python -m signal_study.data --config configs/smoke.json --data-dir "$DATA_RESOLVED"
fi
RUN_DIR="$PROJECT_DIR/runs/$(date -u +%Y%m%dT%H%M%SZ)-s1-${SLURM_JOB_ID}"
printf 'Result directory: %s\n' "$RUN_DIR"
python -m signal_study.parallel --config configs/smoke.json --upstream vendor/SD-square --data-dir "$DATA_RESOLVED" --output "$RUN_DIR"
printf 'Result directory: %s\n' "$RUN_DIR"
