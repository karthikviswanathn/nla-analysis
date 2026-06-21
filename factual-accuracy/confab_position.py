"""Confabulation as a function of TOKEN POSITION (how much context the activation has).

The main judge run graded only each prompt's LAST content-token explanation. Here we additionally
grade the FIRST and MID content-token explanations (3 spread positions/prompt were extracted in the
grid run) with the same Haiku judge — first/mid graded by Haiku subagents into results/jpos/out_*.json,
last reused from logs/judge_<model>.json — and compare confabulation across positions.

Finding: confabulation is strongly context-dependent. At the first content token (minimal preceding
context) the AV guesses the passage topic from ~one word and is wrong ~half-to-two-thirds of the time
(even theme claims ~33% supported); by mid/last token it has read enough and theme support is ~90-97%.
"""
import json

import numpy as np

BASE = "/gpfs/work5/0/gusr0688/fair_stuff/nla-analysis"
MODELS = [("qwen", "Qwen-7B"), ("gemma12", "Gemma-12B"), ("gemma27", "Gemma-27B"), ("llama70", "Llama-70B")]


def claims_first_mid(m, rank):
    return {x["pid"]: x["claims"] for x in json.load(open(f"{BASE}/results/jpos/out_{m}_{rank}.json"))}


def claims_last(m):
    return {it["pid"]: it["claims"] for it in json.load(open(f"{BASE}/logs/judge_{m}.json"))["items"]}


def rate(claims_by_pid):
    cl = [c for v in claims_by_pid.values() for c in v]
    n = max(len(cl), 1)
    ns = sum(c["verdict"] != "supported" for c in cl)
    con = sum(c["verdict"] == "contradicted" for c in cl)
    return len(cl), 100 * ns / n, 100 * con / n, len(cl) / len(claims_by_pid)


def main():
    print("CONFABULATION BY TOKEN POSITION  (not-supported %  (contra%)  claims/expl)")
    print(f"{'model':10s} {'FIRST (least ctx)':>22s} {'MID':>22s} {'LAST (most ctx)':>22s}")
    print("-" * 80)
    agg = {p: [] for p in ["first", "mid", "last"]}
    for m, name in MODELS:
        getters = {"first": claims_first_mid(m, "first"), "mid": claims_first_mid(m, "mid"), "last": claims_last(m)}
        cells = []
        for tag in ["first", "mid", "last"]:
            n, ns, con, cpe = rate(getters[tag])
            cells.append(f"{ns:5.1f}% ({con:4.1f}) {cpe:.1f}c")
            agg[tag].append(ns)
        print(f"{name:10s} {cells[0]:>22s} {cells[1]:>22s} {cells[2]:>22s}")
    print("-" * 80)
    print(f"  mean not-supported:  first={np.mean(agg['first']):.1f}%  "
          f"mid={np.mean(agg['mid']):.1f}%  last={np.mean(agg['last']):.1f}%")

    print("\nSPECIFICITY MIX & SUPPORT BY POSITION (pooled over models)")
    for tag, getter in [("first", lambda m: claims_first_mid(m, "first")),
                        ("mid", lambda m: claims_first_mid(m, "mid")), ("last", claims_last)]:
        allc = [c for m, _ in MODELS for v in getter(m).values() for c in v]
        n = max(len(allc), 1)
        out = []
        for s in ["theme", "entity", "specific_detail"]:
            sc = [c for c in allc if c["specificity"] == s]
            sup = 100 * sum(c["verdict"] == "supported" for c in sc) / max(len(sc), 1)
            out.append(f"{s} {100*len(sc)/n:.0f}% (sup {sup:.0f}%)")
        print(f"  {tag:5s}: " + " | ".join(out))


if __name__ == "__main__":
    main()
