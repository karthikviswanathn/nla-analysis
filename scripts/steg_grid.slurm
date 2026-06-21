#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/work5/0/gusr0688/fair_stuff/nla-analysis/logs/grid_%x_%j.out

# One model of the 4-model steganography grid. Submit with:
#   sbatch --job-name=grid-qwen    --export=ALL,MODEL=qwen    steg_grid.slurm
#   sbatch --job-name=grid-gemma12 --export=ALL,MODEL=gemma12 steg_grid.slurm
#   sbatch --job-name=grid-gemma27 --export=ALL,MODEL=gemma27 steg_grid.slurm
#   sbatch --job-name=grid-llama70 --export=ALL,MODEL=llama70 steg_grid.slurm
# 2 GPUs: AV serve uses per-model tp; the Llama-70B transform server uses tp 2.
set -uo pipefail
BASE=/gpfs/work5/0/gusr0688/fair_stuff/nla-analysis
cd "$BASE"
source "$BASE/.venv-sglang/bin/activate"
export HF_HOME=/projects/gusr0688/.cache/huggingface
export CUDA_HOME=/sw/arch/RHEL9/EB_production/2024/software/CUDA/12.6.0   # H100: deep_gemm import
export REPO="$BASE/nla-inference"
export PYTHONUNBUFFERED=1
export MODEL="${MODEL:?set MODEL=qwen|gemma12|gemma27|llama70}"
export RESULTS="$BASE/logs/grid_${MODEL}_${SLURM_JOB_ID}.json"
export VEC_FILE="$BASE/logs/grid_${MODEL}_${SLURM_JOB_ID}.npz"
export TRANSFORM_URL="http://127.0.0.1:30001/v1/chat/completions"
export TRANSFORM_NAME="transform"
mkdir -p logs

case "$MODEL" in
  qwen)    AV_REPO=kitft/nla-qwen2.5-7b-L20-av;    AR_REPO=kitft/nla-qwen2.5-7b-L20-ar;    AV_EXTRA="--tp 1";;
  gemma12) AV_REPO=kitft/nla-gemma3-12b-L32-av;    AR_REPO=kitft/nla-gemma3-12b-L32-ar;    AV_EXTRA="--tp 1";;
  gemma27) AV_REPO=kitft/nla-gemma3-27b-L41-av;    AR_REPO=kitft/nla-gemma3-27b-L41-ar;    AV_EXTRA="--tp 1 --dtype bfloat16";;
  llama70) AV_REPO=kitft/Llama-3.3-70B-NLA-L53-av; AR_REPO=kitft/Llama-3.3-70B-NLA-L53-ar; AV_EXTRA="--tp 2";;
  *) echo "unknown MODEL=$MODEL"; exit 2;;
esac
export CKPT_AV=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$AV_REPO'))")
export CKPT_AR=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$AR_REPO'))")
TRANSFORM_CKPT=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('unsloth/Llama-3.3-70B-Instruct'))")
echo "[grid:$MODEL] AV=$AV_REPO  AR=$AR_REPO  av_extra='$AV_EXTRA'"

wait_ready () {  # $1=port $2=log $3=pid $4=label
  for i in $(seq 1 300); do
    if grep -q "ready to roll" "$2" 2>/dev/null || curl -sf "http://127.0.0.1:$1/health_generate" >/dev/null 2>&1; then
      echo "[$4] server ready ~$((i*2))s"; return 0; fi
    if ! kill -0 "$3" 2>/dev/null; then echo "[$4] SERVER DIED:"; tail -40 "$2"; return 1; fi
    sleep 2
  done
  echo "[$4] TIMEOUT"; tail -40 "$2"; return 1
}

echo "[grid:$MODEL] PHASE 1: extract"
python steg-test/steg_grid.py extract || { echo "[grid:$MODEL] extract failed"; exit 1; }

echo "[grid:$MODEL] PHASE 2: serve AV + decode"
SGLOG_AV="$BASE/logs/sglang_av_${MODEL}_${SLURM_JOB_ID}.log"
python -m sglang.launch_server --model-path "$CKPT_AV" --host 127.0.0.1 --port 30000 $AV_EXTRA \
    --disable-radix-cache --mem-fraction-static 0.85 --trust-remote-code --context-length 4096 > "$SGLOG_AV" 2>&1 &
AVPID=$!
wait_ready 30000 "$SGLOG_AV" "$AVPID" "av:$MODEL" || { kill "$AVPID" 2>/dev/null; exit 1; }
python steg-test/steg_grid.py decode || { echo "[grid:$MODEL] decode failed"; kill "$AVPID" 2>/dev/null; exit 1; }
kill "$AVPID" 2>/dev/null; wait "$AVPID" 2>/dev/null; sleep 3   # fully free GPUs before transform server

echo "[grid:$MODEL] PHASE 3: serve transform LM (Llama-3.3-70B-Instruct, tp 2) + transform"
SGLOG_T="$BASE/logs/sglang_tf_${MODEL}_${SLURM_JOB_ID}.log"
python -m sglang.launch_server --model-path "$TRANSFORM_CKPT" --host 127.0.0.1 --port 30001 --tp 2 \
    --served-model-name transform --mem-fraction-static 0.85 --trust-remote-code --context-length 4096 > "$SGLOG_T" 2>&1 &
TPID=$!
wait_ready 30001 "$SGLOG_T" "$TPID" "tf:$MODEL" || { kill "$TPID" 2>/dev/null; exit 1; }
python steg-test/steg_grid.py transform || { echo "[grid:$MODEL] transform failed"; kill "$TPID" 2>/dev/null; exit 1; }
kill "$TPID" 2>/dev/null; wait "$TPID" 2>/dev/null; sleep 3   # fully free GPUs before the critic

echo "[grid:$MODEL] PHASE 4: score (AR critic)"
python steg-test/steg_grid.py score || { echo "[grid:$MODEL] score failed"; exit 1; }
echo "[grid:$MODEL] DONE"
