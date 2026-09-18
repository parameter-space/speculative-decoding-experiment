#!/usr/bin/env bash
#SBATCH --job-name=sd2-s1-fp64
#SBATCH --partition=batch_ugrad
#SBATCH --account=ugrad
#SBATCH --qos=qos_leetj3610_2026_2
#SBATCH --nodelist=ariel-k2
#SBATCH --gres=gpu:high_perf:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/s1-fp64-%j.out
#SBATCH --error=logs/s1-fp64-%j.err
set -euo pipefail
source scripts/seraph_common.sh
conda activate "$S1_ENV"
: "${S1_DATA_DIR:?Set S1_DATA_DIR to the frozen K2 compute-local dataset directory}"
S1_DATA_DIR="$(realpath -e -- "$S1_DATA_DIR")"
case "$S1_DATA_DIR" in
  /local_datasets/leetj3610/*|/data2/local_datasets/leetj3610/*) ;;
  *) printf '%s\n' 'STOP: dataset must be in the permitted compute-local directory.' >&2; exit 1 ;;
esac
test -f "$S1_DATA_DIR/manifest.json"
export S1_SDPA_BACKEND=math
export HF_DATASETS_CACHE="$S1_DATA_DIR/cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
mkdir -p "$PROJECT_DIR/runs"
S1_RUN_PARENT="$(mktemp -d "$PROJECT_DIR/runs/s1-fp64-${SLURM_JOB_ID}-XXXXXX")"
printf 'Result directory: %s\n' "$S1_RUN_PARENT/measurement"
python -u -m signal_study.run --config configs/smoke.json --upstream vendor/SD-square \
  --data-dir "$S1_DATA_DIR" --output "$S1_RUN_PARENT/measurement" --target-fp64-reference
