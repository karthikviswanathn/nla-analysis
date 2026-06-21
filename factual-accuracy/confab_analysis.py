"""What gets confabulated — is it the prompt, the token, or the model?

Reads logs/judge_<model>.json (Haiku-4.5 graded claims: specificity x verdict, per the
last-content-token explanation of each of 50 prompts, for 4 models). Confabulation metric per
explanation = fraction of its claims that are NOT 'supported' (unsupported + contradicted);
'contradicted' alone = the strict verifiably-false rate.

Outputs: per-model rates, specificity breakdown, per-prompt (prompt-vs-model variance
decomposition + cross-model agreement), per-domain, and last-token-type breakdowns.
"""
import json
import os
from collections import defaultdict

import numpy as np

BASE = "/gpfs/work5/0/gusr0688/fair_stuff/nla-analysis"
MODELS = [("qwen", "Qwen-7B"), ("gemma12", "Gemma-12B"), ("gemma27", "Gemma-27B"), ("llama70", "Llama-70B")]

# pid -> domain (matches prompts50.py grouping)
DOMAIN = {}
for lo, hi, name in [(0, 4, "history"), (5, 9, "phys-sci"), (10, 12, "law"), (13, 16, "literature"),
                     (17, 19, "news"), (20, 23, "computing"), (24, 27, "bio/med"), (28, 30, "econ"),
                     (31, 33, "philosophy"), (34, 36, "geography"), (37, 39, "math"), (40, 41, "arts"),
                     (42, 43, "sport"), (44, 45, "cooking"), (46, 47, "astronomy"), (48, 49, "linguistics")]:
    for p in range(lo, hi + 1):
        DOMAIN[p] = name


def load():
    data = {}
    for m, _ in MODELS:
        d = json.load(open(os.path.join(BASE, "logs", f"judge_{m}.json")))
        data[m] = {it["pid"]: it for it in d["items"]}
    return data


def not_supp_rate(item):
    """fraction of claims not 'supported'; None if no claims."""
    cl = item.get("claims", [])
    if not cl:
        return None
    return sum(c["verdict"] != "supported" for c in cl) / len(cl)


def contra_rate(item):
    cl = item.get("claims", [])
    if not cl:
        return None
    return sum(c["verdict"] == "contradicted" for c in cl) / len(cl)


def main():
    data = load()
    pids = sorted(set(p for m in data for p in data[m]))

    print("=" * 80)
    print("1. BY MODEL — confabulation & writing")
    print("=" * 80)
    print(f"{'model':10s} {'claims':>7s} {'%support':>9s} {'%unsupp':>8s} {'%contra':>8s} "
          f"{'%NOT-supp':>10s} {'claims/expl':>11s} {'write(1-5)':>11s}")
    for m, name in MODELS:
        cl = [c for it in data[m].values() for c in it.get("claims", [])]
        n = len(cl)
        sup = sum(c["verdict"] == "supported" for c in cl)
        uns = sum(c["verdict"] == "unsupported" for c in cl)
        con = sum(c["verdict"] == "contradicted" for c in cl)
        wr = np.mean([it["writing"]["overall"] for it in data[m].values() if "writing" in it])
        cpe = n / len(data[m])
        print(f"{name:10s} {n:>7d} {100*sup/n:>8.1f}% {100*uns/n:>7.1f}% {100*con/n:>7.1f}% "
              f"{100*(uns+con)/n:>9.1f}% {cpe:>11.1f} {wr:>11.2f}")

    print("\n" + "=" * 80)
    print("2. BY SPECIFICITY — support rate (paper predicts theme > entity > specific_detail)")
    print("=" * 80)
    print(f"{'model':10s} " + "  ".join(f"{s:>16s}" for s in ["theme", "entity", "specific_detail"]))
    for m, name in MODELS:
        cl = [c for it in data[m].values() for c in it.get("claims", [])]
        row = []
        for spec in ["theme", "entity", "specific_detail"]:
            s = [c for c in cl if c["specificity"] == spec]
            row.append(f"{100*sum(x['verdict']=='supported' for x in s)/len(s):.0f}% (n={len(s)})" if s else "—")
        print(f"{name:10s} " + "  ".join(f"{r:>16s}" for r in row))

    # per (prompt, model) NOT-supported matrix
    M = np.full((len(pids), len(MODELS)), np.nan)
    for i, p in enumerate(pids):
        for j, (m, _) in enumerate(MODELS):
            if p in data[m]:
                r = not_supp_rate(data[m][p])
                if r is not None:
                    M[i, j] = r

    print("\n" + "=" * 80)
    print("3. PROMPT vs MODEL — where does the confab variance come from?")
    print("=" * 80)
    grand = np.nanmean(M)
    prompt_mean = np.nanmean(M, axis=1)   # per prompt (across models)
    model_mean = np.nanmean(M, axis=0)    # per model
    # two-way variance decomposition (balanced-ish; uses available cells)
    ss_total = np.nansum((M - grand) ** 2)
    ss_model = np.nansum(~np.isnan(M), axis=0) @ ((model_mean - grand) ** 2)
    ss_prompt = np.nansum(~np.isnan(M), axis=1) @ ((prompt_mean - grand) ** 2)
    ss_resid = ss_total - ss_model - ss_prompt
    print(f"grand mean NOT-supported = {100*grand:.1f}%")
    print(f"  variance from MODEL  : {100*ss_model/ss_total:5.1f}%  (models differ by mean: "
          f"{', '.join(f'{n}={100*mm:.0f}%' for (m,n),mm in zip(MODELS,model_mean))})")
    print(f"  variance from PROMPT : {100*ss_prompt/ss_total:5.1f}%")
    print(f"  residual (interaction+noise): {100*ss_resid/ss_total:5.1f}%")
    # cross-model agreement: do models confabulate on the SAME prompts?
    cols = [M[:, j] for j in range(len(MODELS))]
    cors = []
    for a in range(len(MODELS)):
        for b in range(a + 1, len(MODELS)):
            mask = ~np.isnan(cols[a]) & ~np.isnan(cols[b])
            if mask.sum() > 3:
                cors.append(np.corrcoef(cols[a][mask], cols[b][mask])[0, 1])
    print(f"  mean pairwise cross-model corr of per-prompt confab = {np.mean(cors):+.2f} "
          f"(high => prompt-driven; ~0 => model-specific)")

    print("\n--- WORST prompts (mean NOT-supported across models) ---")
    order = np.argsort(-np.nan_to_num(prompt_mean, nan=-1))
    src = {p: data[m][p]["source"] for m, _ in MODELS for p in data[m] if "source" in data[m][p]}
    for i in order[:8]:
        p = pids[i]
        print(f"  p{p:2d} {100*prompt_mean[i]:5.0f}%  [{DOMAIN.get(p,'?'):11s}] {src.get(p,'')[:70]}")
    print("--- BEST prompts ---")
    for i in order[::-1][:5]:
        p = pids[i]
        print(f"  p{p:2d} {100*prompt_mean[i]:5.0f}%  [{DOMAIN.get(p,'?'):11s}] {src.get(p,'')[:70]}")

    print("\n" + "=" * 80)
    print("4. BY DOMAIN — mean NOT-supported (aggregate across models)")
    print("=" * 80)
    dom = defaultdict(list)
    for i, p in enumerate(pids):
        if not np.isnan(prompt_mean[i]):
            dom[DOMAIN.get(p, "?")].append(prompt_mean[i])
    for d, v in sorted(dom.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"  {d:12s} {100*np.mean(v):5.0f}%   (n={len(v)} prompts)")

    print("\n" + "=" * 80)
    print("5. BY LAST-TOKEN TYPE — punctuation vs content word")
    print("=" * 80)
    for label, pred in [("punctuation (. , ;)", lambda t: not t.strip(" .,:;!?ĠĊ").strip().isalnum()
                                                          and not t.strip(" Ġ").isalpha()),
                        ("word", lambda t: t.strip(" ĠĊ").isalpha())]:
        rates = []
        for m, _ in MODELS:
            for p, it in data[m].items():
                r = not_supp_rate(it)
                tok = it.get("token", "")
                if r is not None and pred(tok):
                    rates.append(r)
        if rates:
            print(f"  {label:22s} mean NOT-supported = {100*np.mean(rates):.1f}%  (n={len(rates)} explanations)")


if __name__ == "__main__":
    main()
