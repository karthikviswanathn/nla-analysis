# NLA Inference on Snellius

Running [`kitft/nla-inference`](https://github.com/kitft/nla-inference) (cloned in `./nla-inference/`)
on the Snellius cluster (SURF). NLA = Natural Language Autoencoder: an **actor** verbalizes an
activation vector → text (served via SGLang `input_embeds`), and an optional **critic**
reconstructs text → vector for MSE/cosine scoring (pure torch, no server).

Full recipe and debugging notes live in `nla-inference/README.md` and the `nla_inference.py`
docstring — this file records the **cluster-specific** decisions only.

## Cluster GPUs (verified 2026-06-18 via `srun ... nvidia-smi`)

| Partition | GPU | VRAM/GPU | GPUs/node | Node RAM | Cores | Compute | Billing/GPU‑hr |
|---|---|---|---|---|---|---|---|
| `gpu_a100` | A100-SXM4 | **40 GB** | 4 (160 GB) | 480 GB | 72 | sm_80 | 128 |
| `gpu_h100` | NVIDIA H100 | **94 GB** (95830 MiB) | 4 (376 GB) | 720 GB | 64 | sm_90 | 192 |

- Driver `595.71.05` → supports CUDA 12.x/13.
- Both are 4-GPU nodes → tensor-parallel up to `--tp 4` stays on one node (NVLink). No multi-node needed for any model below.
- Login node has **no GPU** (`nvidia-smi` absent) — always `srun`/`sbatch` onto a GPU partition.
- Account `gusr38169`, default `normal` QOS. H100 costs ~1.5× the A100 per GPU-hour — reserve it for the 27B/70B models.

## Model feasibility (bf16 serving; NLA prompts are short ⇒ KV cache negligible, it's a weights-fit question)

Usable VRAM at the recipe's `--mem-fraction-static 0.85`: ~34 GB/A100, ~80 GB/H100.

| Model (HF repo) | Weights | Allocation | Notes |
|---|---|---|---|
| **Qwen2.5-7B** L20 — `kitft/nla-qwen2.5-7b-L20-{av,ar}` | ~15 GB | **1× A100** (`--tp 1`) | ✅ Start here. Not gated, no patch, no BOS/embed-scale gotchas, reference model. |
| **Gemma-3-12B** L32 — `kitft/nla-gemma3-12b-L32-{av,ar}` | ~24 GB | 1× A100 (tight) / 1× H100 | gated (HF_TOKEN), needs Gemma3 mm-bypass patch, `embed_scale=√3840≈62`, `injection_scale=80000`. |
| **Gemma-3-27B** L41 — `kitft/nla-gemma3-27b-L41-{av,ar}` | ~54 GB | **1× H100**, or 2× A100 (`--tp 2`) | gated + patch (as above). |
| **Llama-3.3-70B** L53 — `kitft/Llama-3.3-70B-NLA-L53-{av,ar}` | ~141 GB | **2–4× H100** (`--tp 2/4`) | gated. 4× A100 40GB is ~35 GB/GPU vs ~34 usable → **will OOM**; use H100. |

Weights: [`kitft/nla-models` HF collection](https://huggingface.co/collections/kitft/nla-models). Each model is an `av` (actor) + `ar` (critic) pair.

## Weights — already downloaded into the project HF cache (418 GiB, all 8 repos)

All 4 models × (av+ar) are cached under
`/projects/gusr0688/.cache/huggingface/hub/models--kitft--*` (standard hub layout, 418 GiB).
`~/.cache` is a **symlink → `/projects/gusr0688/.cache`** (project space), so HF tooling finds
them there and **nothing touches the near-full `$HOME`**. Re-pull / add models with
`./download_weights.sh` (sets `HF_HOME` to the cache inline; resumable).

Resolve a checkpoint's local snapshot dir from the cache (instant if cached):
```python
from huggingface_hub import snapshot_download
ckpt = snapshot_download("kitft/nla-qwen2.5-7b-L20-av")
```

## Recommended path: Qwen2.5-7B (1× A100; or H100 — see smoke_test.slurm)

```bash
srun --partition=gpu_a100 --gpus=1 --cpus-per-task=16 --time=2:00:00 --pty bash
source /gpfs/work5/0/gusr0688/fair_stuff/nla-analysis/.venv-sglang/bin/activate
export HF_HOME=/projects/gusr0688/.cache/huggingface
export CKPT=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('kitft/nla-qwen2.5-7b-L20-av'))")
python -m sglang.launch_server --model-path "$CKPT" \
    --port 30000 --disable-radix-cache --mem-fraction-static 0.85 --trust-remote-code --tp 1
# --disable-radix-cache is REQUIRED (input_embeds requests have no token IDs to key on).
# On gpu_h100 ALSO set:  export CUDA_HOME=/sw/arch/RHEL9/EB_production/2024/software/CUDA/12.6.0
#   (sglang imports deep_gemm on sm90 and asserts a toolkit path — see gotchas; A100 doesn't need it).

# Client (any host that can reach the server):
#   client = NLAClient(CKPT, sglang_url="http://<gpu-node>:30000")
#   text = client.generate(activation_vector)   # [d_model] array

# End-to-end correctness check (load actor → input_embeds → inject → 5 random decodes + auto CJK check):
sbatch smoke_test.slurm     # ✅ verified PASS on H100 2026-06-18 (5/5 English, ready in ~76s)
```

## Environment — one self-contained uv venv (do everything here)

`/gpfs/work5/0/gusr0688/fair_stuff/nla-analysis/.venv-sglang` (**uv-managed CPython 3.11.14**).
One venv runs **server + client + critic** — no conda, no `llmenv` needed (its torch
2.3.1/cu118 was too old for SGLang anyway).

```bash
source /gpfs/work5/0/gusr0688/fair_stuff/nla-analysis/.venv-sglang/bin/activate
```

**Verified end-to-end on H100 (2026-06-18):** torch 2.9.1+cu128, sglang 0.5.9, transformers
4.57.1, flashinfer 0.6.3, sgl_kernel. Full smoke test (load Qwen actor → SGLang `input_embeds`
→ inject → decode) passes 5/5 English (`smoke_test.slurm`). Imports fine on A100 (sm_80) too.

**Must use a uv-MANAGED Python, not system `/usr/bin/python3.11`.** The system interpreter ships
no `Python.h`, so Triton can't JIT-compile its CUDA helper at first GPU use and SGLang dies during
cuda-graph capture. uv's managed CPython bundles the headers → `--python-preference only-managed`.

Rebuild from scratch (keep cache/tmp/python **off `$HOME`** — home quota is ~97% full):
```bash
export UV_CACHE_DIR=/gpfs/work5/0/gusr0688/fair_stuff/nla-analysis/.uv/cache
export UV_PYTHON_INSTALL_DIR=/gpfs/work5/0/gusr0688/fair_stuff/nla-analysis/.uv/python
export TMPDIR=/gpfs/work5/0/gusr0688/fair_stuff/nla-analysis/.uv/tmp
uv python install 3.11                                          # managed CPython (has Python.h)
uv venv .venv-sglang --python 3.11 --python-preference only-managed
uv pip install --python .venv-sglang/bin/python \
    "sglang[all]>=0.5.6" transformers safetensors httpx orjson pyyaml numpy pyarrow
```

## Cluster / env gotchas (learned the hard way 2026-06-18)

- **venv on uv-managed Python** (not system `python3.11`): system has no `Python.h` → Triton JIT
  fails on first GPU op → SGLang dies at cuda-graph capture. Use `--python-preference only-managed`.
- **H100 (sm90) needs `CUDA_HOME`**: sglang imports `deep_gemm` on Hopper, whose import asserts a
  CUDA-toolkit path (`AssertionError`, not caught). We never call deep_gemm (bf16, not fp8) — just
  `export CUDA_HOME=/sw/arch/RHEL9/EB_production/2024/software/CUDA/12.6.0`. **`CUDA_HOME` only —
  do NOT `module load`**, or the system CUDA 12.6 libs shadow torch's bundled cu128 on
  `LD_LIBRARY_PATH`. A100 (sm80) skips deep_gemm entirely, no `CUDA_HOME` needed.
- **Keep everything off `$HOME`** (quota ~97%): uv cache/python/tmp and the HF cache all live under
  `gusr0688` project space; `~/.cache` is symlinked there.

## NLA gotchas that bite (see README for full list)

- **`injection_scale` is mandatory** and per-model (Qwen 150, Gemma 80000) — vectors are rescaled to this L2 norm; raw vectors are OOD → garbage.
- **Gemma `embed_scale`**: multiply embeddings by `√hidden_size` after lookup (Qwen/Llama = 1.0).
- **Gemma-3 server needs a patch**: the multimodal wrapper ignores `input_embeds` → silent `\n\n\n`. Patch ships in the training repo (`patches/nla_gemma3_mm_input_embeds.patch`).
- **Send `input_embeds` only**, never also `input_ids`. Use one-step `tokenize=True` (avoid double-BOS).
- **Correctness check:** a few full AV decodes returning English (not CJK soup) confirms injection works before trusting any MSE.
