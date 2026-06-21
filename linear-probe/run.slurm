#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --job-name=entity-probe
#SBATCH --output=/gpfs/work5/0/gusr0688/fair_stuff/nla-analysis/logs/probe_%j.out

# Linear-probe experiment: extract Qwen layer-20 activations for the probe dataset, then run the
# probe + controls. transformers only (no SGLang) -> no CUDA_HOME needed.
set -uo pipefail
BASE=/gpfs/work5/0/gusr0688/fair_stuff/nla-analysis
cd "$BASE"
source "$BASE/.venv-sglang/bin/activate"
export HF_HOME=/projects/gusr0688/.cache/huggingface
export PYTHONUNBUFFERED=1
mkdir -p logs

echo "[probe] PHASE 1: extract Qwen layer-20 activations"
python linear-probe/extract_acts.py || { echo "extract failed"; exit 1; }

echo "[probe] PHASE 2: probe + controls"
python linear-probe/probe.py || { echo "probe failed"; exit 1; }
echo "[probe] DONE"
