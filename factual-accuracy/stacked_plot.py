"""Stacked-bar confabulation figure (paper style): factuality of NLA claims by specificity.

Reads logs/judge3_<model>.json (claims graded with the 3-way verdict true_supported /
false_related / false_unrelated). Pools claims across the open NLAs and, for each specificity
level (Theme / Entity / Detail), plots the % that are true-supported, false-but-related, and
false-and-unrelated — reproducing the paper's "higher-level claims are more often supported;
false claims are usually somewhat relevant" figure.

  python factual-accuracy/stacked_plot.py [out.png] [models...]
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
LEVELS = ["theme", "entity", "specific_detail"]
LEVEL_LABELS = {"theme": "Theme\ngenre, topic,\nstructure, era",
                "entity": "Entity\nperson, place,\norg, title",
                "specific_detail": "Detail\nquote, date,\nnumber, value"}
LEVEL_SHORT = {"theme": "Theme", "entity": "Entity", "specific_detail": "Detail"}
MODEL_NAMES = {"qwen": "Qwen2.5-7B", "gemma12": "Gemma-3-12B",
               "gemma27": "Gemma-3-27B", "llama70": "Llama-3.3-70B"}
VERDICTS = ["true_supported", "false_related", "false_unrelated"]   # bottom -> top
V_LABEL = {"true_supported": "True; supported by text",
           "false_related": "False; related",
           "false_unrelated": "False; unrelated"}
V_COLOR = {"true_supported": "#2b6ca3", "false_related": "#e2d2c3", "false_unrelated": "#9c4a2c"}


def tally(models):
    """-> (pct[level][verdict], n_per_level, total_claims) pooled over `models`."""
    counts = {lv: {v: 0 for v in VERDICTS} for lv in LEVELS}
    total = 0
    for m in models:
        p = os.path.join(BASE, "logs", f"judge3_{m}.json")
        if not os.path.exists(p):
            continue
        for it in json.load(open(p))["items"]:
            for c in it["claims"]:
                lv, v = c.get("specificity"), c.get("verdict")
                if lv in counts and v in counts[lv]:
                    counts[lv][v] += 1; total += 1
    pct = {lv: {v: (100.0 * counts[lv][v] / sum(counts[lv].values()) if sum(counts[lv].values()) else 0.0)
                for v in VERDICTS} for lv in LEVELS}
    nper = {lv: sum(counts[lv].values()) for lv in LEVELS}
    return pct, nper, total


def draw_stack(ax, pct, labels, fontsz=11):
    x = np.arange(len(LEVELS))
    bottom = np.zeros(len(LEVELS))
    for v in VERDICTS:
        vals = np.array([pct[lv][v] for lv in LEVELS])
        ax.bar(x, vals, bottom=bottom, width=0.62, color=V_COLOR[v], label=V_LABEL[v],
               edgecolor="white", linewidth=0.5)
        for xi, (val, b) in enumerate(zip(vals, bottom)):
            if val >= 5:
                ax.text(xi, b + val / 2, f"{val:.0f}%", ha="center", va="center",
                        color="white" if v != "false_related" else "#444", fontsize=fontsz)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 100)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def per_model_grid(out, models):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharey=True)
    for ax, m in zip(axes.flat, models):
        pct, nper, total = tally([m])
        draw_stack(ax, pct, [LEVEL_SHORT[lv] for lv in LEVELS], fontsz=10)
        ax.set_title(f"{MODEL_NAMES.get(m, m)}  (N={total} claims)", fontsize=12)
    for ax in axes[:, 0]:
        ax.set_ylabel("% of claim type")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Factuality of NLA claims by specificity — per model", y=1.05, fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "figures", "confab_stacked.png")
    models = sys.argv[2:] or ["qwen", "gemma12", "gemma27", "llama70"]
    os.makedirs(os.path.dirname(out), exist_ok=True)

    # --- pooled figure ---
    pct, ns, total_claims = tally(models)
    fig, ax = plt.subplots(figsize=(8, 6))
    draw_stack(ax, pct, [LEVEL_LABELS[lv] for lv in LEVELS], fontsz=11)
    ax.set_ylabel("% of claim type", fontsize=11)
    ax.set_xlabel("Type of claim made by NLA", fontsize=11, labelpad=8)
    ax.set_title("Higher-level claims are more often supported.\nFalse claims are usually somewhat relevant.",
                 color="#9c4a2c", fontsize=13, pad=12)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=10)
    fig.text(0.5, -0.04,
             f"Factuality of NLA claims by specificity, pooled across {len(models)} open NLAs "
             f"(N={total_claims} claims). Claims about themes are far more likely to be supported "
             f"than specific details; most false claims are at least related to the input.",
             ha="center", fontsize=8.5, color="0.35", wrap=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}  (pooled, N={total_claims})")

    # --- per-model grid ---
    per_model_grid(os.path.join(os.path.dirname(out), "confab_stacked_per_model.png"), models)
    for m in models:
        pct, ns, tot = tally([m])
        print(f"  {m:8s} (N={tot}): " + " | ".join(
            f"{LEVEL_SHORT[lv]} T{pct[lv]['true_supported']:.0f}/R{pct[lv]['false_related']:.0f}/U{pct[lv]['false_unrelated']:.0f}"
            for lv in LEVELS))


if __name__ == "__main__":
    main()
