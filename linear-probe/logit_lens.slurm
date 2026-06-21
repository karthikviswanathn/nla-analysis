#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --job-name=logit-lens
#SBATCH --output=/gpfs/work5/0/gusr0688/fair_stuff/nla-analysis/logs/logitlens_%j.out

set -uo pipefail
BASE=/gpfs/work5/0/gusr0688/fair_stuff/nla-analysis
cd "$BASE"
source "$BASE/.venv-sglang/bin/activate"
export HF_HOME=/projects/gusr0688/.cache/huggingface
export PYTHONUNBUFFERED=1
mkdir -p logs
python linear-probe/logit_lens.py || { echo "logit_lens failed"; exit 1; }
echo "[logit-lens] DONE"
