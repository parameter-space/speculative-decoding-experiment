#!/usr/bin/env bash
#SBATCH --job-name=sd2-cpu-ref
#SBATCH --partition=batch_ugrad
#SBATCH --account=ugrad
#SBATCH --qos=qos_leetj3610_2026_2
#SBATCH --nodelist=ariel-k2
#SBATCH --gres=gpu:high_perf:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/cpu-ref-%j.out
#SBATCH --error=logs/cpu-ref-%j.err
set -euo pipefail
: "${S1_REFERENCE_REPORT:?Set S1_REFERENCE_REPORT to the original failed full FP64 preflight-summary.json}"
test -f "$S1_REFERENCE_REPORT"
bash scripts/diagnose_k2.sh --target-fp64-reference --reference-audit-report "$S1_REFERENCE_REPORT" --cpu-attention-recovery
