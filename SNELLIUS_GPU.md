# Running GPU jobs on Snellius

How to submit GPU jobs on the Snellius SLURM cluster. The patterns and the
`job.slurm` template here are adapted from the `token_geometry` repo's `.slurm`
scripts (`../token_geometry/*.slurm`).

## TL;DR

```bash
# from this directory
sbatch job.slurm            # uses the defaults baked into the script
squeue --me                 # watch your queued/running jobs
tail -f logs/<job>-<id>.out # follow the log
scancel <jobid>             # kill a job
```

## SLURM account & partitions

- **Account:** `gusr38169` (every job needs `#SBATCH --account=gusr38169`).
- **Partitions:**
  - `gpu_h100` — 4×H100 (80 GB) per node, shared. Default choice.
  - `gpu_a100` — A100 **40 GB** — too small for fp32 8B models; use bf16 there.
- `--gpus=1` is enough for a single 8B model. Request more only if you shard.
- Pick `--cpus-per-task` to match your CPU-side parallelism (joblib workers etc.);
  8–32 is typical. Keep `--time` tight — shorter walltime schedules faster.

## Required `#SBATCH` header

```bash
#SBATCH --job-name=my-job
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --account=gusr38169
#SBATCH --output=logs/%x-%j.out   # %x=job-name, %j=job-id; logs/ must exist
```

## Python environment

Use the `llmenv` conda env (torch 2.3.1, transformers 4.52.4, etc.). Two equivalent
ways to invoke it:

```bash
# A) absolute interpreter path (used by the token_geometry .slurm scripts)
PY=/home/karthikv/.conda/envs/llmenv/bin/python
$PY my_script.py

# B) activate first (used by run_extract_id.sh)
source ~/.bashrc
conda activate llmenv
python my_script.py
```

The base/default `python` lacks numpy and will fail — always use `llmenv`.

## Run offline

Weights, datasets, and tokens are pre-fetched into the local HF cache
(`~/.cache/huggingface`). Compute nodes have no/limited internet, so always run
offline to avoid hangs:

```bash
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
```

If you need a new model/dataset, download it on the **login node** first (online),
then submit the job offline.

## Avoid BLAS oversubscription

When CPU-side parallelism (joblib `n_jobs=-1`) runs alongside multithreaded BLAS,
pin BLAS to one thread so the workers don't fight over cores:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

## transformers 4.52.4 + torch<2.5 load bug

Model load crashes in `post_init` with
`argument of type 'NoneType' is not iterable` (`ALL_PARALLEL_STYLES` is `None` but
the `tp_plan` check still runs). Fix: load with a config whose
`base_model_tp_plan=None` (single-GPU, no tensor parallel). See
`../token_geometry/recreate-plots/extract_sharpness.py` for a working example.

## Notes / gotchas

- `logs/` must exist before submit or SLURM silently drops stdout — it's created here.
- `set -euo pipefail` at the top of the script so a failing command aborts the job.
- Launch the actual work under `srun` so it runs inside the SLURM step allocation.
- Pass parameters as positional args with defaults, e.g. `"${1:-512}"`, so one
  script serves many runs: `sbatch job.slurm 1024`.
- The token_geometry `run_extract_id.sh`/`submit_chunks.sh` originally targeted LUMI
  (`--partition=small-g`, `--account=project_465001340`). On Snellius use
  `gpu_h100`/`gpu_a100` and `--account=gusr38169`.
