#!/bin/bash
#SBATCH --job-name=nla-analysis
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --account=gusr38169
#SBATCH --output=logs/%x-%j.out

# Snellius GPU job template. See SNELLIUS_GPU.md for the full guide.
# Usage:
#   sbatch job.slurm [script.py] [n_prompts]
# Example:
#   sbatch job.slurm my_script.py 1024

set -euo pipefail
cd /gpfs/work5/0/gusr0688/fair_stuff/nla-analysis

# --- Run fully offline (weights/datasets are in the local HF cache) ---
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# Keep BLAS single-threaded so joblib workers do not oversubscribe cores.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PY=/home/karthikv/.conda/envs/llmenv/bin/python
SCRIPT="${1:-my_script.py}"
N_PROMPTS="${2:-512}"

echo "host=$(hostname) start=$(date)"
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

srun $PY "$SCRIPT" --n_prompts "$N_PROMPTS"

echo "end=$(date)"
