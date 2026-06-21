"""Real-activation NLA demo: extract layer-20 residual-stream vectors from
Qwen2.5-7B-Instruct for a few prompts, then have the NLA actor verbalize them.

Two phases (separate processes so the base model frees the GPU before SGLang):
  python real_activation_demo.py extract   # base model -> save vectors to VEC_FILE
  python real_activation_demo.py decode     # NLAClient -> explanation per vector

Env: VEC_FILE (npz path), CKPT (actor snapshot dir), REPO (nla-inference dir, for decode).
Recipe per examples/qwen7b_layer20_step4200.txt:
  layer 20 = HF hidden_states[21] (HF puts embeddings at [0]); NLA decode greedy (temp=0).
"""
import os, sys, json
import numpy as np

PROMPTS = [
    "The Eiffel Tower is the most famous landmark in Paris.",
    "Photosynthesis converts sunlight into energy inside plant leaves.",
    "The orchestra played a slow and melancholic symphony.",
    "Investors panicked when the stock market crashed in 2008.",
]
VEC_FILE = os.environ.get("VEC_FILE", "/tmp/nla_vecs.npz")
N_TAIL_CONTENT = int(os.environ.get("N_TAIL_CONTENT", "4"))  # decode last N content tokens / prompt

# Per-model: extraction base + HF hidden_states index (= extraction layer_index + 1,
# since HF puts embeddings at hidden_states[0]). Actor checkpoint comes via $CKPT.
MODELS = {
    "qwen":    dict(base="Qwen/Qwen2.5-7B-Instruct",          hs_index=21),  # layer 20
    "gemma12": dict(base="google/gemma-3-12b-it",             hs_index=33),  # layer 32
    "gemma27": dict(base="google/gemma-3-27b-it",             hs_index=42),  # layer 41
    "llama70": dict(base="unsloth/Llama-3.3-70B-Instruct", hs_index=54),  # layer 53 (ungated mirror; meta repo is gated)
}
MODEL = os.environ.get("MODEL", "qwen")
LAYER_HS_INDEX = MODELS[MODEL]["hs_index"]

# Optional overrides: single prompt, every-token mode, decode length.
if os.environ.get("DEMO_PROMPT"):
    PROMPTS = [os.environ["DEMO_PROMPT"]]
ALL_TOKENS = os.environ.get("DEMO_ALL_TOKENS") == "1"   # decode every token of the full chat seq
MAX_NEW = int(os.environ.get("MAX_NEW_TOKENS", "120"))


def _content_positions(full_ids, content_ids):
    """Find the contiguous span of content_ids inside the full chat sequence."""
    n = len(content_ids)
    for i in range(len(full_ids) - n + 1):
        if full_ids[i:i + n] == content_ids:
            return list(range(i, i + n))
    return []  # fallback: not found


def extract():
    import torch
    from transformers import AutoTokenizer, AutoConfig
    mid = MODELS[MODEL]["base"]
    tok = AutoTokenizer.from_pretrained(mid)
    cfg = AutoConfig.from_pretrained(mid)
    # 70B doesn't fit one GPU → shard with device_map="auto" (needs accelerate).
    big = MODEL == "llama70"
    kw = dict(dtype=torch.bfloat16)
    if big:
        kw["device_map"] = "auto"
    if str(getattr(cfg, "model_type", "")).startswith("gemma3") or hasattr(cfg, "text_config"):
        from transformers import AutoModelForImageTextToText  # Gemma-3 is a multimodal wrapper
        model = AutoModelForImageTextToText.from_pretrained(mid, **kw)
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(mid, **kw)
    model = (model if big else model.to("cuda")).eval()

    vecs, labels = [], []
    for pi, p in enumerate(PROMPTS):
        ids = tok.apply_chat_template([{"role": "user", "content": p}],
                                      add_generation_prompt=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            hs = model(input_ids=ids, output_hidden_states=True,
                       use_cache=False).hidden_states[LAYER_HS_INDEX][0]  # [seq, d]
        full = ids[0].tolist()
        cids = tok(p, add_special_tokens=False)["input_ids"]
        cpos = _content_positions(full, cids) or list(range(len(full)))
        toks = tok.convert_ids_to_tokens(full)
        positions = list(range(len(full))) if ALL_TOKENS else cpos[-N_TAIL_CONTENT:]
        for pos in positions:
            v = hs[pos].float().cpu().numpy()
            vecs.append(v)
            labels.append({"pi": pi, "prompt": p, "pos": pos,
                           "token": toks[pos], "norm": float(np.linalg.norm(v))})
    np.savez(VEC_FILE, vecs=np.stack(vecs).astype(np.float32))
    json.dump(labels, open(VEC_FILE + ".labels.json", "w"))
    print(f"[extract] {len(vecs)} layer-20 vectors from {len(PROMPTS)} prompts -> {VEC_FILE}")


def decode():
    sys.path.insert(0, os.environ["REPO"])
    from nla_inference import NLAClient
    client = NLAClient(os.environ["CKPT"], sglang_url="http://127.0.0.1:30000")
    vecs = np.load(VEC_FILE)["vecs"]
    labels = json.load(open(VEC_FILE + ".labels.json"))
    last_pi = None
    for v, lab in zip(vecs, labels):
        if lab["pi"] != last_pi:
            print("\n" + "=" * 92 + f"\nPROMPT: {lab['prompt']!r}\n" + "=" * 92)
            last_pi = lab["pi"]
        expl = client.generate(v, temperature=0.0, max_new_tokens=MAX_NEW)
        tshow = lab["token"].replace("Ġ", "·").replace("Ċ", "\\n")
        print(f"\n  layer20 @ token[{lab['pos']}]='{tshow}'  ||v||={lab['norm']:.0f}\n    → {expl}")


if __name__ == "__main__":
    {"extract": extract, "decode": decode}[sys.argv[1]]()
