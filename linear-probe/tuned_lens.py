"""Tuned-lens readout of Qwen2.5-7B layer-20 activations (replaces the vanilla logit lens).

The vanilla logit lens was noise at layer 20 (the residual isn't aligned with the unembedding basis
mid-network). The TUNED lens (alunxu/qwen-2.5-7b-tuned-lens-fwedu-262M) learns a per-layer affine
"translator" that maps the intermediate residual into the final-layer space before unembedding:

    logits = lm_head( final_norm( h + W_L @ h + b_L ) )            (residual-correction form)

We apply the layer-20 translator to the exact activations the AV verbalized, and ask whether the
Hannibal passage decodes to Hannibal-cluster tokens (it never said "Columbus" — the AV invented that).

Note: the lens was trained on Qwen2.5-7B *base*; we read Qwen2.5-7B-Instruct activations (the model the
AV/probe use). Same architecture, near-identical unembedding, so the translator transfers; minor caveat.
"""
import sys

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-7B-Instruct"
HS = 21                                   # layer 20 = hidden_states[21]
LENS_LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 20
LENS_REPO = "alunxu/qwen-2.5-7b-tuned-lens-fwedu-262M"
PASSAGES = {
    "Hannibal passage  (the AV verbalized this as 'Columbus')":
        "Hannibal led his army, including a contingent of war elephants, across the Alps in 218 BC "
        "to strike at Rome from the north.",
    "Columbus passage  (sanity contrast)":
        "Columbus sailed west across the Atlantic in 1492 and reached the Americas for the Spanish crown.",
}
HANN = [" Hannibal", " Carthage", " Carthagin", " Punic", " Rome", " Roman", " Alps", " elephant", " elephants", " Italy"]
COLU = [" Columbus", " Atlantic", " Americas", " 1492", " Spain", " Spanish", " voyage", " ocean", " sail", " Indies"]


def first_ids(tok, phrases):
    return {p: tok(p, add_special_tokens=False)["input_ids"][0] for p in phrases
            if tok(p, add_special_tokens=False)["input_ids"]}


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    sd = torch.load(snapshot_download(LENS_REPO) + "/params.pt", map_location="cuda", weights_only=False)
    W = sd[f"{LENS_LAYER}.weight"].to("cuda", torch.float32)
    b = sd[f"{LENS_LAYER}.bias"].to("cuda", torch.float32)
    norm, head = model.model.norm, model.lm_head
    print(f"[tuned-lens] lens layer {LENS_LAYER} on {MODEL} hidden_states[{HS}]")

    def lens_logits(h):                          # h: [T, d] fp32
        hhat = h + (h @ W.T + b)                 # tuned-lens residual correction
        return head(norm(hhat.to(model.dtype))).float()

    hann, colu = first_ids(tok, HANN), first_ids(tok, COLU)
    for name, text in PASSAGES.items():
        ids = tok.apply_chat_template([{"role": "user", "content": text}],
                                      add_generation_prompt=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            hs = model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states[HS][0].float()
            pr = torch.softmax(lens_logits(hs), -1)
            lg = lens_logits(hs)
        full = ids[0].tolist()
        toks = tok.convert_ids_to_tokens(full)
        cids = tok(text, add_special_tokens=False)["input_ids"]
        cpos = next((list(range(i, i + len(cids))) for i in range(len(full) - len(cids) + 1)
                     if full[i:i + len(cids)] == cids), list(range(len(full))))

        def cluster(d):
            return pr[cpos][:, list(d.values())].sum(-1).mean().item()
        ch, cc = cluster(hann), cluster(colu)
        print("\n" + "=" * 84 + f"\n{name}")
        print(f"  tuned-lens cluster prob (mean over content tokens): "
              f"Hannibal={ch:.4f}  Columbus={cc:.4f}  ratio={ch / max(cc, 1e-9):.0f}x")
        for lab, tid in [(" Hannibal", hann[" Hannibal"]), (" Columbus", colu[" Columbus"])]:
            ranks = [int((lg[p] > lg[p][tid]).sum().item()) for p in cpos]
            j = int(np.argmin(ranks))
            print(f"      '{lab.strip()}': best rank {min(ranks):>6}/{lg.shape[1]}  "
                  f"(token '{toks[cpos[j]]}', prob {pr[cpos[j]][tid]:.4f})")
        for pos, lbl in [(cpos[len(cpos) // 2], "mid"), (cpos[-1], "last")]:
            top = torch.topk(pr[pos], 10).indices.tolist()
            print(f"  top-10 tuned-lens tokens @ {lbl} content token ('{toks[pos]}'): "
                  + ", ".join(repr(tok.convert_ids_to_tokens([i])[0]) for i in top))


if __name__ == "__main__":
    main()
