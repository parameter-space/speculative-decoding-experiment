#!/usr/bin/env bash
#SBATCH --job-name=sd2-probability
#SBATCH --partition=batch_ugrad
#SBATCH --account=ugrad
#SBATCH --qos=qos_leetj3610_2026_2
#SBATCH --nodelist=ariel-k2
#SBATCH --gres=gpu:high_perf:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/probability-%j.out
#SBATCH --error=logs/probability-%j.err
set -euo pipefail
# Submit from the project root after mkdir -p logs and export S1_DATA_DIR=...
# Bounded regression on the exact failed calibration case, not full S1.
bash scripts/diagnose_k2.sh --calibration-prompt-id \
  dialogue:train_sft:4986ad8a9ee866215b22b9cf6065959f0f4b19fda609b87a1eeb480ea8a5bef7
