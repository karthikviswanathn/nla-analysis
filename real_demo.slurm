#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:25:00
#SBATCH --job-name=nla-realdemo
#SBATCH --output=/gpfs/work5/0/gusr0688/fair_stuff/nla-analysis/logs/realdemo_%j.out

# Real-activation demo: extract layer-20 vectors from Qwen2.5-7B-Instruct (phase 1,
# frees GPU on exit), then serve the NLA actor and verbalize them (phase 2).
set -uo pipefail
BASE=/gpfs/work5/0/gusr0688/fair_stuff/nla-analysis
cd "$BASE"
source "$BASE/.venv-sglang/bin/activate"
export HF_HOME=/projects/gusr0688/.cache/huggingface
export CUDA_HOME=/sw/arch/RHEL9/EB_production/2024/software/CUDA/12.6.0   # H100: deep_gemm import
export REPO="$BASE/nla-inference"
export VEC_FILE="$BASE/logs/nla_vecs_${SLURM_JOB_ID}.npz"
mkdir -p logs
export CKPT=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('kitft/nla-qwen2.5-7b-L20-av'))")

echo "[demo] PHASE 1: extract layer-20 activations from Qwen2.5-7B-Instruct"
python real_activation_demo.py extract || { echo "[demo] extract failed"; exit 1; }

SGLOG="$BASE/logs/sglang_realdemo_${SLURM_JOB_ID}.log"
echo "[demo] PHASE 2: launch SGLang actor ($CKPT)"
python -m sglang.launch_server --model-path "$CKPT" \
    --host 127.0.0.1 --port 30000 --tp 1 \
    --disable-radix-cache --mem-fraction-static 0.85 \
    --trust-remote-code --context-length 4096 > "$SGLOG" 2>&1 &
SGPID=$!
ready=0
for i in $(seq 1 180); do
  if grep -q "ready to roll" "$SGLOG" 2>/dev/null || curl -sf http://127.0.0.1:30000/health_generate >/dev/null 2>&1; then
    ready=1; echo "[demo] server ready ~$((i*2))s"; break; fi
  if ! kill -0 "$SGPID" 2>/dev/null; then echo "[demo] SERVER DIED:"; tail -40 "$SGLOG"; exit 1; fi
  sleep 2
done
[ "$ready" = 1 ] || { echo "[demo] TIMEOUT"; tail -40 "$SGLOG"; kill "$SGPID" 2>/dev/null; exit 1; }

echo "[demo] decoding real activations (greedy):"
python real_activation_demo.py decode
kill "$SGPID" 2>/dev/null
echo "[demo] DONE"
