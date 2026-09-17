#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/seraph_common.sh"
# No ToS acceptance, no changes to my_env, and no automatic fallback versions.
if [[ ! -d "$S1_ENV" ]]; then
  conda create --yes --prefix "$S1_ENV" python=3.12 pip
fi
conda activate "$S1_ENV"
python -c 'import sys; assert sys.version_info[:2] == (3, 12), "sd2_s1 must use Python 3.12"'
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip check
PIN=00f578aaabc4d48efd9a10ccd0e8d0297bdc6e07
if [[ ! -d vendor/SD-square ]]; then
  git clone --no-checkout https://github.com/ETH-DISCO/SD-square.git vendor/SD-square
  git -C vendor/SD-square checkout --detach "$PIN"
fi
[[ "$(git -C vendor/SD-square rev-parse HEAD)" == "$PIN" ]] || { printf '%s\n' 'Unexpected upstream revision; not changing it.' >&2; exit 1; }
[[ -z "$(git -C vendor/SD-square status --porcelain --untracked-files=no)" ]] || { printf '%s\n' 'Upstream has local edits; stopping.' >&2; exit 1; }
python -c 'import sys; sys.path.insert(0,"vendor/SD-square"); import main; import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() in (1,2); print("Allocated GPU count:", torch.cuda.device_count()); [(print("GPU:", torch.cuda.get_device_name(i)), print("BF16 test:", (torch.ones(2,device="cuda:"+str(i),dtype=torch.bfloat16)*2).sum().item())) for i in range(torch.cuda.device_count())]'
python scripts/test_local.py
printf '%s\n' 'Environment checks finished. No 8B checkpoint experiment has run yet.'
