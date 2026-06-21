"""Extract Qwen2.5-7B layer-20 activations for the linear-probe dataset.

For each passage we chat-wrap it (exactly like the main NLA pipeline) and take the residual-stream
activation at hidden_states[21] (= Qwen layer 20) at the LAST content token — the same position the
AV verbalized. Saves per-class arrays + the exemplar Hannibal sentence the AV confabulated on, so the
probe can be applied to that exact vector.
"""
import json
import os

import numpy as np

BASE = "/gpfs/work5/0/gusr0688/fair_stuff/nla-analysis"
MODEL = "Qwen/Qwen2.5-7B-Instruct"
HS = 21  # layer 20 (hidden_states[0] = embeddings)
EXEMPLAR = ("Hannibal led his army, including a contingent of war elephants, across the Alps in "
            "218 BC to strike at Rome from the north.")


def content_positions(full, cids):
    n = len(cids)
    for i in range(len(full) - n + 1):
        if full[i:i + n] == cids:
            return list(range(i, i + n))
    return list(range(len(full)))


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()

    @torch.no_grad()
    def vec(text):
        ids = tok.apply_chat_template([{"role": "user", "content": text}],
                                      add_generation_prompt=True, return_tensors="pt").to("cuda")
        hs = model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states[HS][0]
        full = ids[0].tolist()
        cids = tok(text, add_special_tokens=False)["input_ids"]
        cpos = content_positions(full, cids)
        return hs[cpos[-1]].float().cpu().numpy()   # last content token (the AV's position)

    P = json.load(open(f"{BASE}/linear-probe/passages.json"))
    out = {}
    for k, sents in P.items():
        out[k] = np.stack([vec(s) for s in sents]).astype(np.float32)
        print(f"[extract] {k}: {out[k].shape}", flush=True)
    out["exemplar"] = vec(EXEMPLAR)[None, :].astype(np.float32)
    np.savez(f"{BASE}/linear-probe/acts.npz", **out)
    print(f"[extract] saved {BASE}/linear-probe/acts.npz")


if __name__ == "__main__":
    main()
