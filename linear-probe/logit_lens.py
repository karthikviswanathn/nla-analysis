"""Logit lens on Qwen2.5-7B layer-20 activations (model-intrinsic complement to the linear probe).

Project the layer-20 residual through the model's OWN final RMSNorm + unembedding (lm_head) and read
the resulting vocabulary distribution. No trained probe. If the Hannibal passage's layer-20
activation decodes to Hannibal-cluster tokens (Carthage/Rome/elephants) and NOT 'Columbus', the
identity is present in a form the model itself surfaces — so the AV ('Columbus') simply isn't reading it.
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-7B-Instruct"
HS = 21  # layer 20
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
    d = {}
    for p in phrases:
        ids = tok(p, add_special_tokens=False)["input_ids"]
        if ids:
            d[p] = ids[0]   # leading token (what the logit lens would predict)
    return d


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    hann, colu = first_ids(tok, HANN), first_ids(tok, COLU)

    for name, text in PASSAGES.items():
        ids = tok.apply_chat_template([{"role": "user", "content": text}],
                                      add_generation_prompt=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            hs = model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states[HS][0]
            lg = model.lm_head(model.model.norm(hs)).float()   # logit lens at layer 20: [T, vocab]
            pr = torch.softmax(lg, -1)
        full = ids[0].tolist()
        toks = tok.convert_ids_to_tokens(full)
        cids = tok(text, add_special_tokens=False)["input_ids"]
        cpos = None
        for i in range(len(full) - len(cids) + 1):
            if full[i:i + len(cids)] == cids:
                cpos = list(range(i, i + len(cids))); break
        cpos = cpos or list(range(len(full)))

        def cluster(d):
            return pr[cpos][:, list(d.values())].sum(-1).mean().item()
        ch, cc = cluster(hann), cluster(colu)
        print("\n" + "=" * 84 + f"\n{name}")
        print(f"  layer-20 logit-lens cluster prob (mean over content tokens):")
        print(f"      Hannibal-cluster = {ch:.4f}      Columbus-cluster = {cc:.4f}      "
              f"ratio = {ch / max(cc, 1e-9):.0f}x")
        for lab, tid in [(" Hannibal", hann[" Hannibal"]), (" Columbus", colu[" Columbus"])]:
            ranks = [int((lg[p] > lg[p][tid]).sum().item()) for p in cpos]
            b = int(np.argmin(ranks))
            print(f"      '{lab.strip()}': best rank {min(ranks):>6} / {lg.shape[1]} vocab  "
                  f"(at token '{toks[cpos[b]]}', prob {pr[cpos[b]][tid]:.4f})")
        topk = torch.topk(pr[cpos[-1]], 10).indices.tolist()
        print(f"  top-10 logit-lens tokens @ last content token ('{toks[cpos[-1]]}'): "
              + ", ".join(repr(tok.convert_ids_to_tokens([i])[0]) for i in topk))


if __name__ == "__main__":
    main()
