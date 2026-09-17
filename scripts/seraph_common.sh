#!/usr/bin/env bash
# Source only inside the user's allocated compute shell; no work on master.
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  printf '%s\n' 'STOP: run inside an allocated Slurm compute job, not on master.' >&2
  return 1 2>/dev/null || exit 1
fi
case "$(hostname -s)" in
  ariel-k2) ;;
  *) printf '%s\n' 'STOP: this first-run package is configured for ariel-k2.' >&2; return 1 2>/dev/null || exit 1 ;;
esac
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
case "$PROJECT_DIR" in
  /ceph_data/leetj3610/*) ;;
  *) printf '%s\n' 'STOP: put this project below /ceph_data/leetj3610.' >&2; return 1 2>/dev/null || exit 1 ;;
esac
source /data/leetj3610/anaconda3/etc/profile.d/conda.sh
export HF_HOME=/ceph_data/leetj3610/cache/huggingface
export TORCH_HOME=/ceph_data/leetj3610/cache/torch
export TRITON_CACHE_DIR=/ceph_data/leetj3610/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/ceph_data/leetj3610/cache/torchinductor
export PIP_CACHE_DIR=/ceph_data/leetj3610/cache/pip
export MPLCONFIGDIR=/ceph_data/leetj3610/cache/matplotlib
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
S1_ENV=/data/leetj3610/anaconda3/envs/sd2_s1
cd "$PROJECT_DIR"
