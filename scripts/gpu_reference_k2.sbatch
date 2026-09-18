#!/usr/bin/env bash
#SBATCH --job-name=sd2-gpu-ref
#SBATCH --partition=batch_ugrad
#SBATCH --account=ugrad
#SBATCH --qos=qos_leetj3610_2026_2
#SBATCH --nodelist=ariel-k2
#SBATCH --gres=gpu:high_perf:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/gpu-ref-%j.out
#SBATCH --error=logs/gpu-ref-%j.err
set -euo pipefail
bash scripts/diagnose_k2.sh --target-fp64-reference --gpu-attention-reference
