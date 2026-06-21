#!/usr/bin/env bash
# Download all 8 NLA repos into the project-space HF cache (off $HOME, which was the problem).
# HF_HOME is set INLINE here only (not persisted to venv/profile/CLAUDE.md).
# Restartable: hf download resumes/skips already-cached files, so re-running is safe.
set -uo pipefail

BASE=/gpfs/work5/0/gusr0688/fair_stuff/nla-analysis
source "$BASE/.venv-sglang/bin/activate"

# THE safe cache (project space). Default ~/.cache is on home (~97% full) — never use it.
export HF_HOME=/projects/gusr0688/.cache/huggingface

REPOS=(
  kitft/nla-qwen2.5-7b-L20-av     # Qwen first = testable early
  kitft/nla-qwen2.5-7b-L20-ar
  kitft/nla-gemma3-12b-L32-ar     # then ascending size
  kitft/nla-gemma3-12b-L32-av
  kitft/nla-gemma3-27b-L41-ar
  kitft/Llama-3.3-70B-NLA-L53-ar
  kitft/nla-gemma3-27b-L41-av
  kitft/Llama-3.3-70B-NLA-L53-av
)

fail=0
for repo in "${REPOS[@]}"; do
  echo "=== [$(date +%H:%M:%S)] START $repo ==="
  if path=$(hf download "$repo"); then
    echo "=== [$(date +%H:%M:%S)] OK    $repo"
    echo "    snapshot: $path"
  else
    echo "=== [$(date +%H:%M:%S)] FAIL  $repo ==="; fail=1
  fi
done
echo "=== [$(date +%H:%M:%S)] ALL FINISHED (fail=$fail) ==="
