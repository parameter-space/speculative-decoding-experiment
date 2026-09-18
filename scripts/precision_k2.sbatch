#!/usr/bin/env bash
#SBATCH --job-name=sd2-precision
#SBATCH --partition=batch_ugrad
#SBATCH --account=ugrad
#SBATCH --qos=qos_leetj3610_2026_2
#SBATCH --nodelist=ariel-k2
#SBATCH --gres=gpu:high_perf:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/precision-%j.out
#SBATCH --error=logs/precision-%j.err
set -euo pipefail
bash scripts/diagnose_k2.sh --target-fp32-probe --calibration-prompt-id \
  dialogue:train_sft:4986ad8a9ee866215b22b9cf6065959f0f4b19fda609b87a1eeb480ea8a5bef7
