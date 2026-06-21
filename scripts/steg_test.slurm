#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --job-name=nla-steg
#SBATCH --output=/gpfs/work5/0/gusr0688/fair_stuff/nla-analysis/logs/steg_%j.out

# Steganography test, Qwen-7B (step 1). 4 phases, one heavy model resident at a time:
#   extract (Qwen-7B) -> decode (SGLang AV) -> transform (strong 70B) -> score (AR critic)
# 2 GPUs: phases 1/2/4 use one; phase 3 shards the 70B transform model across both (device_map=auto).
set -uo pipefail
BASE=/gpfs/work5/0/gusr0688/fair_stuff/nla-analysis
cd "$BASE"
source "$BASE/.venv-sglang/bin/activate"
export HF_HOME=/projects/gusr0688/.cache/huggingface
export CUDA_HOME=/sw/arch/RHEL9/EB_production/2024/software/CUDA/12.6.0   # H100: deep_gemm import
export REPO="$BASE/nla-inference"
export TRANSFORM_MODEL=unsloth/Llama-3.3-70B-Instruct   # faithful paraphrase/translate (not weak Qwen-7B)
export RESULTS="$BASE/logs/steg_${SLURM_JOB_ID}.json"
export VEC_FILE="$BASE/logs/steg_${SLURM_JOB_ID}.npz"
mkdir -p logs

export CKPT_AV=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('kitft/nla-qwen2.5-7b-L20-av'))")
export CKPT_AR=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('kitft/nla-qwen2.5-7b-L20-ar'))")
echo "[steg] AV=$CKPT_AV"
echo "[steg] AR=$CKPT_AR"

echo "[steg] PHASE 1: extract layer-20 gold vectors (base Qwen2.5-7B-Instruct)"
python steg-test/steg_test.py extract || { echo "[steg] extract failed"; exit 1; }

echo "[steg] PHASE 2: serve AV + decode explanations"
SGLOG="$BASE/logs/sglang_steg_${SLURM_JOB_ID}.log"
python -m sglang.launch_server --model-path "$CKPT_AV" \
    --host 127.0.0.1 --port 30000 --tp 1 \
    --disable-radix-cache --mem-fraction-static 0.85 \
    --trust-remote-code --context-length 4096 > "$SGLOG" 2>&1 &
SGPID=$!
ready=0
for i in $(seq 1 180); do
  if grep -q "ready to roll" "$SGLOG" 2>/dev/null || curl -sf http://127.0.0.1:30000/health_generate >/dev/null 2>&1; then
    ready=1; echo "[steg] server ready ~$((i*2))s"; break; fi
  if ! kill -0 "$SGPID" 2>/dev/null; then echo "[steg] SERVER DIED:"; tail -50 "$SGLOG"; exit 1; fi
  sleep 2
done
[ "$ready" = 1 ] || { echo "[steg] TIMEOUT"; tail -50 "$SGLOG"; kill "$SGPID" 2>/dev/null; exit 1; }
python steg-test/steg_test.py decode || { echo "[steg] decode failed"; kill "$SGPID" 2>/dev/null; exit 1; }
kill "$SGPID" 2>/dev/null; sleep 3   # free GPU before transform/score

echo "[steg] PHASE 3: transform explanations (transform model = $TRANSFORM_MODEL)"
python steg-test/steg_test.py transform || { echo "[steg] transform failed"; exit 1; }

echo "[steg] PHASE 4: score with AR critic"
python steg-test/steg_test.py score || { echo "[steg] score failed"; exit 1; }
echo "[steg] DONE"
