#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:25:00
#SBATCH --job-name=nla-pertoken
#SBATCH --output=/gpfs/work5/0/gusr0688/fair_stuff/nla-analysis/logs/pertoken_%j.out

# Per-token decode of ONE prompt (every token of the full chat sequence), max_new_tokens=200.
set -uo pipefail
BASE=/gpfs/work5/0/gusr0688/fair_stuff/nla-analysis
cd "$BASE"
source "$BASE/.venv-sglang/bin/activate"
export HF_HOME=/projects/gusr0688/.cache/huggingface
export CUDA_HOME=/sw/arch/RHEL9/EB_production/2024/software/CUDA/12.6.0
export REPO="$BASE/nla-inference"
export VEC_FILE="$BASE/logs/nla_pertoken_${SLURM_JOB_ID}.npz"
export DEMO_PROMPT="The Eiffel Tower is the most famous landmark in Paris."
export DEMO_ALL_TOKENS=1
export MAX_NEW_TOKENS=200
mkdir -p logs
export CKPT=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('kitft/nla-qwen2.5-7b-L20-av'))")

echo "[pertoken] PHASE 1: extract EVERY layer-20 token activation for: $DEMO_PROMPT"
python real_activation_demo.py extract || { echo "[pertoken] extract failed"; exit 1; }

SGLOG="$BASE/logs/sglang_pertoken_${SLURM_JOB_ID}.log"
python -m sglang.launch_server --model-path "$CKPT" \
    --host 127.0.0.1 --port 30000 --tp 1 \
    --disable-radix-cache --mem-fraction-static 0.85 \
    --trust-remote-code --context-length 4096 > "$SGLOG" 2>&1 &
SGPID=$!
ready=0
for i in $(seq 1 180); do
  if grep -q "ready to roll" "$SGLOG" 2>/dev/null || curl -sf http://127.0.0.1:30000/health_generate >/dev/null 2>&1; then
    ready=1; echo "[pertoken] server ready ~$((i*2))s"; break; fi
  if ! kill -0 "$SGPID" 2>/dev/null; then echo "[pertoken] SERVER DIED:"; tail -40 "$SGLOG"; exit 1; fi
  sleep 2
done
[ "$ready" = 1 ] || { echo "[pertoken] TIMEOUT"; tail -40 "$SGLOG"; kill "$SGPID" 2>/dev/null; exit 1; }

echo "[pertoken] decoding every token (greedy, max_new_tokens=200):"
python real_activation_demo.py decode
kill "$SGPID" 2>/dev/null
echo "[pertoken] DONE"
