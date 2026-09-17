#!/usr/bin/env bash
#SBATCH --job-name=sd2-s1-smoke
#SBATCH --partition=debug_ugrad
#SBATCH --account=ugrad
#SBATCH --qos=qos_leetj3610_2026_2
#SBATCH --nodelist=ariel-k2
#SBATCH --gres=gpu:high_perf:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/s1-%j.out
#SBATCH --error=logs/s1-%j.err
set -euo pipefail
# Submit from the project root after mkdir -p logs and export S1_DATA_DIR=...
# Install/setup once in an interactive compute allocation first.
bash scripts/run_k2.sh
