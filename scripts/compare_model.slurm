#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --job-name=nla-compare
#SBATCH --output=/gpfs/work5/0/gusr0688/fair_stuff/nla-analysis/logs/compare_%j.out

# One model of the four-way comparison. Pass MODEL via:
#   sbatch --export=ALL,MODEL=gemma12 compare_model.slurm
# Extracts its OWN base activation for a shared prompt, then verbalizes with its actor.
set -uo pipefail
BASE=/gpfs/work5/0/gusr0688/fair_stuff/nla-analysis
cd "$BASE"
source "$BASE/.venv-sglang/bin/activate"
export HF_HOME=/projects/gusr0688/.cache/huggingface
export CUDA_HOME=/sw/arch/RHEL9/EB_production/2024/software/CUDA/12.6.0
export REPO="$BASE/nla-inference"
export MODEL="${MODEL:-gemma12}"
export VEC_FILE="$BASE/logs/cmp_${MODEL}_${SLURM_JOB_ID}.npz"
export DEMO_PROMPT="The Eiffel Tower is the most famous landmark in Paris."
export N_TAIL_CONTENT=3
export MAX_NEW_TOKENS=200
mkdir -p logs

case "$MODEL" in
  qwen)    ACTOR=kitft/nla-qwen2.5-7b-L20-av;     EXTRA="--tp 1";;
  gemma12) ACTOR=kitft/nla-gemma3-12b-L32-av;     EXTRA="--tp 1";;
  gemma27) ACTOR=kitft/nla-gemma3-27b-L41-av;     EXTRA="--tp 1 --dtype bfloat16";;  # av stored fp32
  llama70) ACTOR=kitft/Llama-3.3-70B-NLA-L53-av;  EXTRA="--tp 2";;                    # needs 2 GPUs
  *) echo "unknown MODEL=$MODEL"; exit 2;;
esac
export CKPT=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$ACTOR'))")
echo "[cmp:$MODEL] actor=$ACTOR  extra='$EXTRA'"

echo "[cmp:$MODEL] PHASE 1: extract base activation"
python real_activation_demo.py extract || { echo "[cmp:$MODEL] extract failed"; exit 1; }

SGLOG="$BASE/logs/sglang_cmp_${MODEL}_${SLURM_JOB_ID}.log"
python -m sglang.launch_server --model-path "$CKPT" \
    --host 127.0.0.1 --port 30000 $EXTRA \
    --disable-radix-cache --mem-fraction-static 0.85 \
    --trust-remote-code --context-length 4096 > "$SGLOG" 2>&1 &
SGPID=$!
ready=0
for i in $(seq 1 240); do
  if grep -q "ready to roll" "$SGLOG" 2>/dev/null || curl -sf http://127.0.0.1:30000/health_generate >/dev/null 2>&1; then
    ready=1; echo "[cmp:$MODEL] server ready ~$((i*2))s"; break; fi
  if ! kill -0 "$SGPID" 2>/dev/null; then echo "[cmp:$MODEL] SERVER DIED:"; tail -50 "$SGLOG"; exit 1; fi
  sleep 2
done
[ "$ready" = 1 ] || { echo "[cmp:$MODEL] TIMEOUT"; tail -50 "$SGLOG"; kill "$SGPID" 2>/dev/null; exit 1; }

echo "[cmp:$MODEL] decoding (greedy):"
python real_activation_demo.py decode
kill "$SGPID" 2>/dev/null
echo "[cmp:$MODEL] DONE"
