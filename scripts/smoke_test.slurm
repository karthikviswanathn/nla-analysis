#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --job-name=nla-smoke
#SBATCH --output=/gpfs/work5/0/gusr0688/fair_stuff/nla-analysis/logs/smoke_%j.out

# End-to-end smoke test: SGLang serves the cached Qwen NLA actor, we inject
# random vectors and confirm decodes are English prose (injection works) and
# not CJK soup (injection failed). See nla-inference/README.md §Debugging.
set -uo pipefail
BASE=/gpfs/work5/0/gusr0688/fair_stuff/nla-analysis
cd "$BASE"
source "$BASE/.venv-sglang/bin/activate"
export HF_HOME=/projects/gusr0688/.cache/huggingface          # safe project-space cache (inline only)

# Hopper (H100/sm90) only: sglang imports deep_gemm, which asserts CUDA_HOME at
# import time. We never call a deep_gemm op (bf16, not fp8) — just point it at a
# real toolkit so the import succeeds. Set CUDA_HOME ONLY (no `module load`) so
# the system CUDA 12.6 libs don't shadow torch's bundled cu128 on LD_LIBRARY_PATH.
# Harmless on A100 (deep_gemm is skipped for sm<90).
export CUDA_HOME=/sw/arch/RHEL9/EB_production/2024/software/CUDA/12.6.0

mkdir -p logs

# Resolve the already-cached snapshot dir (instant — no re-download)
export CKPT=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('kitft/nla-qwen2.5-7b-L20-av'))")
echo "[smoke] checkpoint: $CKPT"
SGLOG="$BASE/logs/sglang_${SLURM_JOB_ID}.log"

# --- launch SGLang server in background ---
python -m sglang.launch_server --model-path "$CKPT" \
    --host 127.0.0.1 --port 30000 --tp 1 \
    --disable-radix-cache --mem-fraction-static 0.85 \
    --trust-remote-code --context-length 4096 > "$SGLOG" 2>&1 &
SGPID=$!
echo "[smoke] sglang pid=$SGPID; waiting for readiness (log: $SGLOG)"

# --- wait until ready (or die) ---
ready=0
for i in $(seq 1 180); do
  if grep -q "ready to roll" "$SGLOG" 2>/dev/null || curl -sf http://127.0.0.1:30000/health_generate >/dev/null 2>&1; then
    ready=1; echo "[smoke] server ready after ~$((i*2))s"; break
  fi
  if ! kill -0 "$SGPID" 2>/dev/null; then echo "[smoke] !! SERVER DIED early; tail of log:"; tail -60 "$SGLOG"; exit 1; fi
  sleep 2
done
if [ "$ready" != 1 ]; then echo "[smoke] !! TIMEOUT waiting for server; tail of log:"; tail -60 "$SGLOG"; kill "$SGPID" 2>/dev/null; exit 1; fi

# --- decode several random vectors; auto-flag CJK ---
export PYTHONPATH="$BASE/nla-inference:${PYTHONPATH:-}"
python - <<'PY'
import os, re, numpy as np
from nla_inference import NLAClient
client = NLAClient(os.environ["CKPT"], sglang_url="http://127.0.0.1:30000")
cjk = re.compile(r'[　-鿿가-힯豈-﫿]')
np.random.seed(0)
n_eng = 0
for k in range(5):
    v = np.random.randn(client.cfg.d_model).astype(np.float32)
    out = client.generate(v, temperature=0.7, max_new_tokens=150)
    frac = sum(bool(cjk.search(c)) for c in out) / max(len(out), 1)
    eng = frac < 0.15
    n_eng += eng
    print(f"\n===== decode {k}  CJK-fraction={frac:.0%}  -> {'ENGLISH ok' if eng else 'CJK SOUP!'} =====")
    print(out[:700])
verdict = "WORKS ✅" if n_eng >= 3 else "SUSPECT ❌ (see README §Debugging: injection-failure smell)"
print(f"\n[smoke] {n_eng}/5 decodes English  ->  injection {verdict}")
PY

kill "$SGPID" 2>/dev/null
echo "[smoke] DONE"
