#!/usr/bin/env bash
#SBATCH --job-name=sd2-ref-audit
#SBATCH --partition=batch_ugrad
#SBATCH --account=ugrad
#SBATCH --qos=qos_leetj3610_2026_2
#SBATCH --nodelist=ariel-k2
#SBATCH --gres=gpu:high_perf:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/ref-audit-%j.out
#SBATCH --error=logs/ref-audit-%j.err
set -euo pipefail
: "${S1_REFERENCE_REPORT:?Set S1_REFERENCE_REPORT to the previous FP64 preflight-summary.json}"
test -f "$S1_REFERENCE_REPORT"
bash scripts/diagnose_k2.sh --target-fp64-reference --reference-audit-report "$S1_REFERENCE_REPORT"
