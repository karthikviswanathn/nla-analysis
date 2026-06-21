"""Factual-accuracy / confabulation table for the 4 open NLAs (paper limitation #1).

Reads logs/judge_<model>.json (from ../judge_explanations.py). Reports claim-support rate broken
down by SPECIFICITY (theme / entity / specific_detail) per model, plus overall support, the
contradiction rate, and claims per explanation. The paper's finding to look for: thematic claims
are supported far more than specific ones (theme > entity > specific).

  python factual-accuracy/report.py [out.md]
"""
import json
import os
import sys

BASE = "/gpfs/work5/0/gusr0688/fair_stuff/nla-analysis"
MODELS = [("qwen", "Qwen2.5-7B"), ("gemma12", "Gemma-3-12B"),
          ("gemma27", "Gemma-3-27B"), ("llama70", "Llama-3.3-70B")]
LEVELS = ["theme", "entity", "specific_detail"]


def load(model):
    p = os.path.join(BASE, "logs", f"judge_{model}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def rate(claims, level=None, verdict="supported"):
    sub = [c for c in claims if level is None or c.get("specificity") == level]
    return (100 * sum(c.get("verdict") == verdict for c in sub) / len(sub)) if sub else float("nan")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "factual-accuracy", "factual_accuracy.md")
    L = ["# Factual accuracy / confabulation of NLA explanations (paper limitation #1) — final checkpoint, N=50/model\n",
         "Claims about the source text, graded by an OpenAI grader. Cells = **% of claims supported** "
         "at each specificity level (higher = more accurate). Expect theme > entity > specific.\n",
         "| Model | theme | entity | specific | all (support%) | contradicted% | claims/expl | N |",
         "|---|---|---|---|---|---|---|---|"]
    for m, name in MODELS:
        d = load(m)
        if not d:
            L.append(f"| {name} | — | — | — | — | — | — | _pending_ |"); continue
        claims = [c for it in d["items"] for c in it["claims"]]
        n_expl = len(d["items"])
        def f(x):
            return f"{x:.0f}%" if x == x else "—"   # NaN check
        L.append(f"| {name} | {f(rate(claims,'theme'))} | {f(rate(claims,'entity'))} | "
                 f"{f(rate(claims,'specific_detail'))} | {f(rate(claims))} | "
                 f"{f(rate(claims,None,'contradicted'))} | {len(claims)/max(n_expl,1):.1f} | {n_expl} |")
    L.append("\n_Support% = fraction of extracted claims judged true of the source. The theme>entity>"
             "specific gradient is the paper's confabulation signature: NLAs get the gist right but "
             "invent specific details._\n")
    txt = "\n".join(L)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write(txt)
    print(txt)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
