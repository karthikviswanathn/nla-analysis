"""Single-pass LLM-judge over NLA explanations -> writing quality + factual accuracy.

Reuses the explanations already produced by the steg grid runs (logs/grid_<model>_*.json): no
GPU / no AV decoding needed. Per model we take the LAST content-token explanation of each of the
50 prompts (cleanest, most context) and send ONE judge call that returns BOTH:
  (1) writing quality  — clarity / coherence / conciseness / overall (1-5), legibility only
  (2) factual accuracy — checkable claims about the SOURCE text, each tagged
        specificity {theme, entity, specific_detail} x verdict {supported, unsupported, contradicted}

Judge = OpenAI (api.openai.com works from the cluster; the Azure endpoint is firewall-blocked).
Writes logs/judge_<model>.json consumed by writing-quality/report.py and factual-accuracy/report.py.

  source .env && python judge_explanations.py [qwen gemma12 gemma27 llama70]
"""
import concurrent.futures as cf
import glob
import json
import os
import sys
import time

import httpx

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o")
KEY = os.environ["OPENAI_API_KEY"]
URL = "https://api.openai.com/v1/chat/completions"
CONC = int(os.environ.get("JCONC", "6"))
BASE = "/gpfs/work5/0/gusr0688/fair_stuff/nla-analysis"
ALL_MODELS = ["qwen", "gemma12", "gemma27", "llama70"]

SYS = """You evaluate explanations produced by a Natural Language Autoencoder (NLA). An NLA reads a \
single internal activation of a language model that is processing some TEXT and verbalizes, in \
English, what that activation encodes about the text. You are given the SOURCE TEXT the model was \
processing and the NLA EXPLANATION of one activation taken while reading it. Do two things.

(1) WRITING QUALITY (legibility only, independent of factual correctness): rate the explanation's
    clarity, coherence, and conciseness, each an integer 1-5 (5 = best), plus an overall 1-5.

(2) FACTUAL ACCURACY: extract the discrete, checkable claims the explanation makes ABOUT THE SOURCE
    TEXT. Skip "next-token" guesses and hedged meta-claims about the model's internal processing.
    Extract claims at ALL THREE specificity levels when the explanation makes them — do NOT label
    everything "specific_detail". Most explanations assert a high-level topic/genre AND name entities
    AND give details; capture each separately.
      specificity:
        "theme"  = the overall topic, domain, genre, or format the explanation ascribes to the text
                   (e.g. "is a historical record", "concerns biology", "reads like a quiz/prompt").
        "entity" = a particular named entity, person, place, term, or concept it says the text is about
                   (e.g. "mentions a drought", "about a treaty", "concerns mitochondria/Hannibal").
        "specific_detail" = an exact quoted phrase, a precise date/number, a cited source, or a precise
                   factual assertion (e.g. quotes "rain rituals", "happened in 218 BC", cites a source).
      verdict: "supported" (clearly true of the source text), "unsupported" (plausible/related but not
               inferable from the source), or "contradicted" (conflicts with the source).
    Example: source "Hannibal crossed the Alps in 218 BC"; explanation says it is a quiz about Columbus
    and quotes "sailed westward". -> theme "is a quiz/question format" (unsupported), entity "about
    Columbus" (contradicted; it is about Hannibal), specific_detail quotes "sailed westward"
    (unsupported).

Return ONLY a JSON object, no prose:
{"writing":{"clarity":int,"coherence":int,"conciseness":int,"overall":int},
 "claims":[{"claim":string,"specificity":"theme|entity|specific_detail","verdict":"supported|unsupported|contradicted"}]}"""


def judge_one(client, source, explanation):
    user = f"SOURCE TEXT:\n{source}\n\nNLA EXPLANATION:\n{explanation}"
    payload = {"model": JUDGE_MODEL, "temperature": 0,
               "response_format": {"type": "json_object"},
               "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": user}]}
    last = None
    for attempt in range(6):
        try:
            r = client.post(URL, headers={"Authorization": f"Bearer {KEY}"}, json=payload)
            if r.status_code == 429 or r.status_code >= 500:
                last = f"HTTP {r.status_code}"; time.sleep(2 * (attempt + 1)); continue
            r.raise_for_status()
            return json.loads(r.json()["choices"][0]["message"]["content"])
        except Exception as e:
            last = str(e); time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"judge failed after retries: {last}")


def latest_grid(model):
    fs = sorted(glob.glob(os.path.join(BASE, "logs", f"grid_{model}_*.json")), key=os.path.getmtime)
    return fs[-1] if fs else None


def pick_last_token(samples):
    """One explanation per prompt: the last content-token position (most context)."""
    by_pid = {}
    for s in samples:
        if not s.get("explanation"):
            continue
        if s["pid"] not in by_pid or s["pos"] > by_pid[s["pid"]]["pos"]:
            by_pid[s["pid"]] = s
    return [by_pid[k] for k in sorted(by_pid)]


def run(model):
    p = latest_grid(model)
    if not p:
        print(f"[judge:{model}] no grid json"); return
    r = json.load(open(p))
    items = pick_last_token(r["samples"])
    if not items:
        print(f"[judge:{model}] no explanations yet (decode not done?)"); return
    client = httpx.Client(timeout=120.0)
    out = []

    def work(s):
        v = judge_one(client, s["sample_text"], s["explanation"])
        return {"pid": s["pid"], "token": s["token"], "pos": s["pos"],
                "source": s["sample_text"], "explanation": s["explanation"],
                "writing": v.get("writing", {}), "claims": v.get("claims", [])}

    fails = 0
    with cf.ThreadPoolExecutor(max_workers=CONC) as ex:
        for i, res in enumerate(ex.map(work, items)):
            out.append(res)
            if i % 10 == 0:
                print(f"[judge:{model}] {i+1}/{len(items)}", flush=True)
    client.close()
    res = {"model": model, "judge_model": JUDGE_MODEL, "n": len(out), "grid_json": os.path.basename(p),
           "items": out}
    op = os.path.join(BASE, "logs", f"judge_{model}.json")
    json.dump(res, open(op, "w"), indent=1, ensure_ascii=False)
    print(f"[judge:{model}] wrote {op}  ({len(out)} explanations judged)")


if __name__ == "__main__":
    models = sys.argv[1:] or ALL_MODELS
    for m in models:
        run(m)
