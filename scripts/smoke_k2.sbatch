#!/usr/bin/env bash
#SBATCH --job-name=sd2-s1-smoke
#SBATCH --partition=batch_ugrad
#SBATCH --account=ugrad
#SBATCH --qos=qos_leetj3610_2026_2
#SBATCH --nodelist=ariel-k2
#SBATCH --gres=gpu:high_perf:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/s1-%j.out
#SBATCH --error=logs/s1-%j.err
set -euo pipefail
# Submit from the project root after mkdir -p logs and export S1_DATA_DIR=...
# Install/setup once in an interactive compute allocation first.
export S1_SDPA_BACKEND=math
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
printf '%s\n' 'S1 smoke: math SDPA, CPU FP64 metrics, unchanged model precision/gate caps, cached models/data.'
bash scripts/run_k2.sh
