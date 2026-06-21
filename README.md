# nla-analysis

Demonstrating limitations of **Natural Language Autoencoders (NLAs)** on small open-weight models.
We re-run three limitation experiments from the [NLA paper](https://transformer-circuits.pub/2026/nla/index.html)
(steganography, confabulation, writing quality) on four open NLAs (Qwen-2.5-7B, Gemma-3-12B,
Gemma-3-27B, Llama-3.3-70B), then dig into why small models confuse entities.

**The writeup is the deliverable: [WRITEUP.md](WRITEUP.md).**

## Repo layout

| Path | What's in it |
|---|---|
| [WRITEUP.md](WRITEUP.md) | Main report: executive summary, the three experiments, and the entity-confusion deep dive. |
| [CLAUDE.md](CLAUDE.md) | Cluster-specific notes: Snellius GPUs/partitions, model feasibility, env setup, NLA gotchas. |
| [`nla-inference/`](nla-inference/) | Vendored [`kitft/nla-inference`](https://github.com/kitft/nla-inference) — the AV/AR inference code (used as-is). |
| [`knowledge-base/`](knowledge-base/) | Background notes: [nla.md](knowledge-base/nla.md), [limitations.md](knowledge-base/limitations.md), [llama_problem.md](knowledge-base/llama_problem.md). |
| [`steg-test/`](steg-test/) | Steganography experiment: single-model test (`steg_test.py`), 4-model grid (`steg_grid.py`), 50 passages (`prompts50.py`), table/plot scripts, and the [writeup](steg-test/steg_Experiment.md). |
| [`factual-accuracy/`](factual-accuracy/) | Confabulation analysis: 3-way verdict scoring + position/domain breakdowns (`confab_analysis.py`, `confab_position.py`) and the stacked-bar plot (`stacked_plot.py`). |
| [`writing-quality/`](writing-quality/) | Writing-quality judging report (`report.py`). |
| [`linear-probe/`](linear-probe/) | Entity deep dive: extract Qwen layer-20 activations + logistic-regression probe (`extract_acts.py`, `probe.py`); logit-lens / tuned-lens readouts (excluded from the writeup). |
| [`scripts/`](scripts/) | All SLURM batch scripts (`*.slurm`). |
| [`results/`](results/) | Raw experiment outputs (JSON/NPZ) per model + run id. |
| [`figures/`](figures/) | Plots used in the writeup. |
| [`patches/`](patches/) | The Gemma-3 multimodal `input_embeds` bypass patch. |
| `judge_explanations.py` | Standalone OpenAI-based explanation judge (alternative to the Haiku-via-Claude-Code judging used for the final tables). |
| `download_weights.sh` | Pull/resume the 8 NLA weight repos into the project HF cache. |

## Running GPU jobs on Snellius

The cluster recipe (partitions, account, HF cache, env, gotchas) lives in [CLAUDE.md](CLAUDE.md).
All experiments run via SLURM:

```bash
sbatch scripts/<job>.slurm     # submit
squeue --me                    # check status
tail -f logs/<job>_<id>.out    # follow output
```

Everything runs in one self-contained uv venv (`.venv-sglang`); see CLAUDE.md for the
environment build and the SGLang serving caveats (`--disable-radix-cache`, H100 `CUDA_HOME`,
Gemma-3 patch).
