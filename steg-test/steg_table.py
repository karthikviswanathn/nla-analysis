"""4x6 steganography table for the four open NLAs (final checkpoint only).

The paper sweeps the x-axis across TRAINING CHECKPOINTS; we only have each model's final
released checkpoint, so a line plot has one point per model. Instead we report a table:
rows = models, cols = transforms, cell = % FVE maintained at the final checkpoint.

% FVE maintained = 100 * mean_i FVE(variant_i) / mean_i FVE(original_i)   (ratio of means: robust
to per-sample noise, esp. for Gemma whose tiny predict-the-mean variance makes individual ratios
unstable). FVE = 1 - mse_nrm / var_pm, var_pm = each model's own large content-pool baseline.

Also prints raw mean FVE / cos / mse per cell and per-model baselines for transparency.

  python steg_table.py [out.md]
"""
import glob
import json
import os
import sys

import numpy as np

BASE = "/gpfs/work5/0/gusr0688/fair_stuff/nla-analysis"
MODELS = [("qwen", "Qwen2.5-7B"), ("gemma12", "Gemma-3-12B"),
          ("gemma27", "Gemma-3-27B"), ("llama70", "Llama-3.3-70B")]
PANELS = [("shuffle", "Shuffle"), ("paraphrase_light", "Paraphrase (light)"),
          ("paraphrase_aggressive", "Paraphrase (aggr.)"), ("translate_fr", "French transl."),
          ("coherence", "Coherence rewrite"), ("summary", "Paragraph summary")]
KIND = {"shuffle": "MP", "paraphrase_light": "MP", "paraphrase_aggressive": "MP",
        "translate_fr": "MP", "coherence": "CTRL", "summary": "CTRL"}
# Qwen training-set Var(v_nrm) is published (0.7335); use it for Qwen. Others: own pool estimate.
KNOWN_BASELINE = {"qwen": 0.7335}


def latest(model):
    fs = sorted(glob.glob(os.path.join(BASE, "results", f"grid_{model}_*.json")), key=os.path.getmtime)
    return fs[-1] if fs else None


def load(model):
    p = latest(model)
    if not p:
        return None
    r = json.load(open(p))
    if not r["samples"] or "scores" not in r["samples"][0]:
        return None
    V = KNOWN_BASELINE.get(model, r["config"]["var_pm"])
    S = r["samples"]

    def arr(nt, key):
        return np.array([s["scores"][nt][key] for s in S
                         if s["scores"].get(nt, {}).get(key) is not None], dtype=float)

    fve = {nt: 1 - arr(nt, "mse") / V for nt, _ in [("original", "")] + PANELS}
    cos = {nt: arr(nt, "cos") for nt, _ in [("original", "")] + PANELS}
    return {"V": V, "pool_V": r["config"].get("var_pm"), "n": len(S), "fve": fve, "cos": cos}


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "results", "steg_table.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    data = {m: load(m) for m, _ in MODELS}

    L = []
    L.append("# Steganography (failure mode A) — 4 open NLAs, final checkpoint, N=50 prompts (×3 positions = 150 activations)\n")
    L.append("Cell = **% FVE maintained** = 100 · mean FVE(variant) / mean FVE(original). "
             "Meaning-preserving (MP) transforms should stay high; information-removing controls (CTRL) drop.\n")

    # main table: % FVE maintained
    hdr = "| Model | " + " | ".join(f"{t} ({KIND[k]})" for k, t in PANELS) + " |"
    sep = "|" + "---|" * (len(PANELS) + 1)
    L.append("## % FVE maintained\n")
    L.append(hdr)
    L.append(sep)
    for m, name in MODELS:
        d = data[m]
        if d is None:
            L.append(f"| {name} | " + " | ".join(["—"] * len(PANELS)) + " |   _(pending)_")
            continue
        fo = d["fve"]["original"].mean()
        cells = []
        for k, _ in PANELS:
            fv = d["fve"][k].mean()
            cells.append(f"{100*fv/fo:.0f}%")
        L.append(f"| {name} | " + " | ".join(cells) + " |")

    # context table: baselines and original reconstruction quality
    L.append("\n## Per-model context\n")
    L.append("| Model | baseline var_pm | mean orig FVE | mean orig cos | N |")
    L.append("|---|---|---|---|---|")
    for m, name in MODELS:
        d = data[m]
        if d is None:
            L.append(f"| {name} | _(pending)_ | | | |"); continue
        vnote = f"{d['V']:.4f}" + (" (known)" if m in KNOWN_BASELINE else f" (pool; raw {d['pool_V']:.4f})")
        L.append(f"| {name} | {vnote} | {d['fve']['original'].mean():.3f} | "
                 f"{d['cos']['original'].mean():.3f} | {d['n']} |")

    # robustness cross-check: mean cosine(recon, gold) per cell (baseline-free)
    L.append("\n## Cross-check — mean cosine(reconstruction, gold) [baseline-free]\n")
    L.append("| Model | original | " + " | ".join(t for _, t in PANELS) + " |")
    L.append("|" + "---|" * (len(PANELS) + 2))
    for m, name in MODELS:
        d = data[m]
        if d is None:
            L.append(f"| {name} | " + " | ".join(["—"] * (len(PANELS) + 1)) + " |"); continue
        row = [f"{d['cos']['original'].mean():.3f}"] + [f"{d['cos'][k].mean():.3f}" for k, _ in PANELS]
        L.append(f"| {name} | " + " | ".join(row) + " |")

    L.append("\n_Note: Gemma residual streams are extremely anisotropic (massive outlier dimensions), "
             "so their predict-the-mean variance is tiny and cosine is compressed near 1 — FVE is the "
             "discriminating metric there. We lack training checkpoints, so this is the final-checkpoint "
             "snapshot, not the paper's across-training sweep._\n")

    txt = "\n".join(L)
    open(out, "w").write(txt)
    print(txt)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
