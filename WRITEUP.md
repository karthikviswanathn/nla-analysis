# Demonstrating limitations of Natural Language Autoencoders in small open weight models
Github repo: https://github.com/karthikviswanathn/nla-analysis

## Why did I choose this problem?
I haven’t worked extensively with autoencoders before, though I have a rough high-level understanding of SAEs: how they work and some of their limitations. That said, I found the idea of NLAs quite interesting, and this project felt like a good way to better understand what is going on and build first order intuition about the limitations of NLAs. 

I cloned [`kitft/nla-inference`](https://github.com/kitft/nla-inference) (cloned in `./nla-inference/`) to perform inference locally. 

## My understanding of NLAs
[Natural Language Autoencoders](https://transformer-circuits.pub/2026/nla/index.html) are an unsupervised method for generating natural language explanations of LLM activations. It comprises of two components 
1. Activation verbaliser (AV): Take a token representation $h_l \in \mathbb{R}^d$ and autoregressively generate the explanation $z$. 
2. Activation reconstructor (AR): Take the generated explanation $z$ and produce $h_l \in \mathbb{R}^d$ by running it through the truncated (upto layer $l$) model and using the representation of the last token in $z$. It is like taking a transformer, splliting it into two at layer $l$, where the first part is the activation reconstructor.


## Some initial skepticism
An NLA reads a *single* activation: one token, one layer. But a model's computation is spread across both, so a single $h_l$ is only a local snapshot.

- **Across tokens:** $h_l$ is the running summary up to that token; relevant cognition can sit elsewhere in the sequence. 
- **Across layers:** different information lives at different depths, and the NLA only reads its trained layer. 

However, while this is an interesting direction, testing that skepticism would take more than four hours to operationalize and implement. So I took a more straightforward approach: first recreating experiments from the [NLA paper](https://transformer-circuits.pub/2026/nla/index.html) that expose limitations, but applying them to smaller models. It is not fully obvious to me why the results should extend to smaller models, so I wouldn't mind spending some time validating the results on the smaller models. For this reason I run the experiments on
| Model | [Qwen-2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | [Gemma-3-12B](https://huggingface.co/google/gemma-3-12b-it) | [Gemma-3-27B](https://huggingface.co/google/gemma-3-27b-it) | [Llama-3.3-70B](https://huggingface.co/unsloth/Llama-3.3-70B-Instruct) |
|---|---|---|---|---|
| NLA layer | 20 | 32 | 41 | 53 |

## Evaluating NLAs for limitation
Experimental details
We evaluate the NLAs for the following limitations
- Steganography
- Confabulations
- Writing quality

**Setup:**

- **Data:** **50 short, domain-diverse pretraining-like passages** (history, science, law, etc.).
- **Activations:** for each model we extract the residual-stream activation at the NLA's trained layer at **3 token positions spread across each passage** → **150 activations per model**.
- **Verbalisation:** each activation is verbalised by the AV with greedy decoding (temperature = 0)[^greedy].
- **Transforms:** each explanation is put through the **six transforms** — the four meaning-preserving probes (shuffle, light paraphrase, aggressive paraphrase, French translation) and the two information-removing controls (coherence rewrite, paragraph summary).
- **Transform model:** the five generative transforms are produced by a strong instruction model, **Llama-3.3-70B-Instruct** (served via SGLang), so that a "meaning-preserving" rewrite genuinely preserves the content rather than silently summarising it.
- **Metric:** we reconstruct each transformed explanation with the AR and report **% FVE maintained = mean FVE(variant) / mean FVE(original)**, where FVE is normalised by each model's predict-the-mean activation variance.

[^greedy]: Claude Code chose **greedy decoding (T=0)** here for reproducibility — it makes the pipeline deterministic, so each activation yields the same explanation across runs and the only thing that varies across the six transforms is the transform itself. The NLA paper instead **samples the AV at temperature T=1** (the setting the AV was RL-trained under, used both at inference and for GRPO rollouts). I did not change this in the interest of time; with more time it would be nice to see both T = 0 and T = 1.

### Steganography
First, we measure steganography between the AV and AR: Do NLA explanations communicate information beyond their overt natural language meaning? We test for this by applying a variety of transformations (e.g., paraphrasing, shuffling bullet points) to the AV explanations and measure FVE. A low number suggests that the AV has learned to encode information that isn’t apparent from the explanation’s overt reading.

In the NLA paper they find that, for the trained NLA, the four meaning-preserving transforms (shuffle, light and aggressive paraphrase, and French translation) retain almost all of the FVE while only the two information-removing controls (coherence rewrite and paragraph summary) substantially reduce it — i.e. little evidence of steganography.


We find — consistent with the paper — **little evidence of steganography**: the meaning-preserving transforms largely retain FVE, while the information-removing controls collapse it.

| Model | Shuffle | Paraphrase (light) | Paraphrase (aggr.) | French transl. | Coherence rewrite (CTRL) | Paragraph summary (CTRL) |
|---|---|---|---|---|---|---|
| Qwen-2.5-7B | 96% | 98% | 81% | 70% | 18% | 32% |
| Gemma-3-12B | 96% | 100% | 94% | 76% | 16% | 40% |
| Gemma-3-27B | 94% | 97% | 78% | 79% | 20% | 32% |
| Llama-3.3-70B | 95% | 64%\* | 34%\* | 4%\* | −0% | 11% |

The cleanest probe is **shuffle** — a pure reordering of the explanation's snippets that *cannot* add or remove information — and it maintains **94–96% of FVE on every model**, the strongest single piece of evidence against steganography. The other meaning-preserving transforms depend on the transform model rewriting *faithfully*: for the verbose Qwen/Gemma explanations the strong rewriter paraphrases at ~1.2× length and FVE is largely retained, but the Llama-3.3-70B AV emits very terse explanations that the same rewriter *expands ~2.3×* (entries marked \*) — no longer meaning-preserving, which accounts for Llama's low paraphrase/French numbers. That is a transform-fidelity artifact, not hidden encoding; the faithful `shuffle` probe (95%) confirms little steganography for Llama too. The controls (coherence rewrite, paragraph summary) drop FVE sharply on every model, as expected, because they delete content.

### Confabulations
NLA explanations can make verifiably false claims about the input. For each model we take **50 explanations** (the last content-token explanation of each passage), extract the discrete checkable claims each makes about its *known* source passage, and grade every claim with an LLM judge (**Claude Haiku**, run via Claude Code) by **specificity** (theme / entity / specific detail) and **validity** (supported / unsupported / contradicted). We report the **% of claims supported** at each level.

| Model | theme | entity | specific | all (support) | contradicted | claims/expl |
|---|---|---|---|---|---|---|
| Qwen-2.5-7B | 96% | 69% | 56% | 72% | 18% | 3.2 |
| Gemma-3-12B | 94% | 68% | 91% | 81% | 7% | 3.0 |
| Gemma-3-27B | 100% | 95% | 83% | 92% | 4% | 3.3 |
| Llama-3.3-70B | 60% | 91% | 86% | 82% | 7% | 2.7 |

Confabulation is **substantial** on every model (~20–45% of claims are not supported by the source). Qwen reproduces the paper's signature gradient — **theme (96%) > entity (69%) > specific (56%)**: the NLA gets the gist right but invents specific details — and has the highest contradiction rate (18%). The larger Gemma-3-27B is the most factually reliable (92% support, 4% contradiction).

### Writing quality
Trained NLA explanations become harder to parse (there is no reward for legibility). Using the same Haiku judge, we grade each explanation's **legibility only** (independent of correctness) on clarity, coherence, and conciseness (1–5).

| Model | clarity | coherence | conciseness | overall |
|---|---|---|---|---|
| Qwen-2.5-7B | 2.04 | 2.00 | 2.02 | 2.02 |
| Gemma-3-12B | 2.98 | 2.74 | 2.44 | 2.78 |
| Gemma-3-27B | 3.16 | 2.68 | 2.42 | 2.68 |
| Llama-3.3-70B | 1.98 | 1.88 | 1.98 | 1.88 |

Legibility is mediocre across the board (~2–3 / 5). The Gemma explanations are the most readable; Qwen and the (notably terse) Llama-3.3-70B the least. As above, we only have final checkpoints, so this is a snapshot rather than the paper's over-training decline.