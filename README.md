# nla-analysis

Working directory under `fair_stuff/`.

## Running GPU jobs on Snellius

See [SNELLIUS_GPU.md](SNELLIUS_GPU.md) for the full guide (account, partitions,
offline HF cache, env, known bugs). The repo ships a ready-to-edit
[job.slurm](job.slurm) template:

```bash
sbatch job.slurm my_script.py 1024   # submit
squeue --me                          # check status
tail -f logs/<job>-<id>.out          # follow output
```

Example scripts these patterns were derived from live in
`../token_geometry/*.slurm`.
