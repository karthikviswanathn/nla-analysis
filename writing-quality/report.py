"""Writing-quality table for the 4 open NLAs (paper limitation #7: legibility).

Reads logs/judge_<model>.json (produced by ../judge_explanations.py) and reports mean writing-quality
scores per model. We only have final checkpoints, so this is a final-checkpoint snapshot, not the
paper's across-training decline.

  python writing-quality/report.py [out.md]
"""
import json
import os
import sys

import numpy as np

BASE = "/gpfs/work5/0/gusr0688/fair_stuff/nla-analysis"
MODELS = [("qwen", "Qwen2.5-7B"), ("gemma12", "Gemma-3-12B"),
          ("gemma27", "Gemma-3-27B"), ("llama70", "Llama-3.3-70B")]
DIMS = ["clarity", "coherence", "conciseness", "overall"]


def load(model):
    p = os.path.join(BASE, "logs", f"judge_{model}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "writing-quality", "writing_quality.md")
    L = ["# Writing quality of NLA explanations (paper limitation #7) — final checkpoint, N=50/model\n",
         "Judge: OpenAI grader, each explanation scored 1-5 per dimension (5 = best). Legibility only "
         "(independent of factual correctness).\n",
         "| Model | clarity | coherence | conciseness | overall | N |",
         "|---|---|---|---|---|---|"]
    for m, name in MODELS:
        d = load(m)
        if not d:
            L.append(f"| {name} | — | — | — | — | _pending_ |"); continue
        cells = []
        for dim in DIMS:
            xs = [it["writing"].get(dim) for it in d["items"] if isinstance(it["writing"].get(dim), (int, float))]
            cells.append(f"{np.mean(xs):.2f}" if xs else "—")
        L.append(f"| {name} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {d['n']} |")
    L.append("\n_Higher = more legible. Compare across the four open models; we cannot show the "
             "paper's over-training decline (no intermediate checkpoints)._\n")
    txt = "\n".join(L)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write(txt)
    print(txt)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
