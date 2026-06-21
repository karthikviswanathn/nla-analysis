"""Plot the 4-model steganography grid (the paper figure, nla.md:302-315).

2x3 panels (one per transform). x = baseline FVE of the activation (binned, the "FVE Norm"
axis). y = % FVE maintained = 100 * FVE(transformed) / FVE(original), averaged per bin. One
line per NLA model. Reads the latest logs/grid_<model>_*.json for each model.

  python steg_grid_plot.py [out.png]
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/gpfs/work5/0/gusr0688/fair_stuff/nla-analysis"
MODELS = [("qwen", "Qwen2.5-7B", "#1f77b4"),
          ("gemma12", "Gemma-3-12B", "#ff7f0e"),
          ("gemma27", "Gemma-3-27B", "#2ca02c"),
          ("llama70", "Llama-3.3-70B", "#d62728")]
PANELS = [("shuffle", "Shuffle"),
          ("paraphrase_light", "Paraphrase (light)"),
          ("paraphrase_aggressive", "Paraphrase (aggressive)"),
          ("translate_fr", "French translation"),
          ("coherence", "Coherence rewrite"),
          ("summary", "Paragraph summary")]
EDGES = np.array([0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90])   # FVE-norm bins
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2
MIN_PER_BIN = 2
# Baselines. We have one ground-truth FVE variance: Qwen training-set Var(v_nrm)=0.7335 (the paper's
# convention). Our within-corpus pool estimate for Qwen is 0.5826 (extract probe), i.e. the narrow
# 50-prompt corpus is more anisotropic than the training set and under-estimates the variance by
# ~1.26x. We calibrate every model's pool estimate (config.var_pm) by that factor so all baselines
# match the published convention; Qwen is pinned to its known value. FVE is recomputed from saved
# mse, so this is tunable here with NO GPU re-run.
QWEN_TRUE = 0.7335
QWEN_POOL = 0.5826
CALIB = QWEN_TRUE / QWEN_POOL
BASELINE_OVERRIDE = {"qwen": QWEN_TRUE}


def latest_json(model):
    fs = sorted(glob.glob(os.path.join(BASE, "logs", f"grid_{model}_*.json")), key=os.path.getmtime)
    return fs[-1] if fs else None


def collect(model):
    """-> dict transform -> (x=fve_orig array, y=%maintained array) over all activations."""
    path = latest_json(model)
    if not path:
        return None, None
    r = json.load(open(path))
    if "var_pm" not in r["config"] or not r["samples"] or "scores" not in r["samples"][0]:
        return None, None   # extract/score not finished yet
    V = BASELINE_OVERRIDE.get(model) or r["config"]["var_pm"] * CALIB

    def fve(sc):
        m = sc.get("mse")
        return None if m is None else 1.0 - m / V

    out = {nt: ([], []) for nt, _ in PANELS}
    for s in r["samples"]:
        if "scores" not in s:
            continue
        o = fve(s["scores"]["original"])
        if o is None or o <= 0.05:                       # ratio undefined / unstable near 0
            continue
        for nt, _ in PANELS:
            v = fve(s["scores"].get(nt, {}))
            if v is None:
                continue
            out[nt][0].append(o)
            out[nt][1].append(100.0 * v / o)
    return r["config"], out


def binned(x, y):
    x, y = np.asarray(x), np.asarray(y)
    idx = np.digitize(x, EDGES) - 1
    means, ns = [], []
    for b in range(len(CENTERS)):
        sel = idx == b
        ns.append(int(sel.sum()))
        means.append(float(np.mean(y[sel])) if sel.sum() >= MIN_PER_BIN else np.nan)
    return np.array(means), ns


def main():
    out_png = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "figures", "steg_grid.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    data = {}
    for m, _, _ in MODELS:
        cfg, d = collect(m)
        if d is not None:
            data[m] = d
            print(f"[{m}] {cfg.get('var_pm', '?'):} var_pm; "
                  f"shuffle bin counts {binned(*d['shuffle'])[1]}")
        else:
            print(f"[{m}] no results json yet — skipping")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for ax, (nt, title) in zip(axes.flat, PANELS):
        ax.set_title(title, fontsize=11)
        ax.axhline(100, color="0.8", lw=1, ls="--", zorder=0)
        for m, label, color in MODELS:
            if m not in data:
                continue
            means, ns = binned(*data[m][nt])
            ax.plot(CENTERS, means, marker="o", ms=4, color=color, label=label)
        ax.set_xlim(0.30, 0.90)
        ax.set_ylim(0, 118)
        ax.grid(True, alpha=0.25)
    for ax in axes[:, 0]:
        ax.set_ylabel("% FVE maintained")
    for ax in axes[1, :]:
        ax.set_xlabel("FVE Norm (baseline FVE bin)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Measuring steganography between the AV and AR  —  4 open NLAs, N=50 prompts",
                 y=1.05, fontsize=13)
    fig.text(0.5, -0.02, "Meaning-preserving transforms (shuffle, paraphrase, French) maintain FVE; "
             "coherence rewrites and paragraph summaries do not (they remove information).",
             ha="center", fontsize=9, color="0.3")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out_png}")


if __name__ == "__main__":
    main()
