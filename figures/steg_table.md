# Steganography (failure mode A) — 4 open NLAs, final checkpoint, N=50 prompts (×3 positions = 150 activations)

Cell = **% FVE maintained** = 100 · mean FVE(variant) / mean FVE(original). Meaning-preserving (MP) transforms should stay high; information-removing controls (CTRL) drop.

## % FVE maintained

| Model | Shuffle (MP) | Paraphrase (light) (MP) | Paraphrase (aggr.) (MP) | French transl. (MP) | Coherence rewrite (CTRL) | Paragraph summary (CTRL) |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | 96% | 98% | 81% | 70% | 18% | 32% |
| Gemma-3-12B | 96% | 100% | 94% | 76% | 16% | 40% |
| Gemma-3-27B | 94% | 97% | 78% | 79% | 20% | 32% |
| Llama-3.3-70B | 95% | 64% | 34% | 4% | -0% | 11% |

## Per-model context

| Model | baseline var_pm | mean orig FVE | mean orig cos | N |
|---|---|---|---|---|
| Qwen2.5-7B | 0.7335 (known) | 0.802 | 0.927 | 150 |
| Gemma-3-12B | 0.0277 (pool; raw 0.0277) | 0.696 | 0.996 | 150 |
| Gemma-3-27B | 0.0545 (pool; raw 0.0545) | 0.770 | 0.994 | 150 |
| Llama-3.3-70B | 0.7550 (pool; raw 0.7550) | 0.797 | 0.924 | 150 |

## Cross-check — mean cosine(reconstruction, gold) [baseline-free]

| Model | original | Shuffle | Paraphrase (light) | Paraphrase (aggr.) | French transl. | Coherence rewrite | Paragraph summary |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | 0.927 | 0.915 | 0.921 | 0.871 | 0.839 | 0.686 | 0.728 |
| Gemma-3-12B | 0.996 | 0.995 | 0.996 | 0.995 | 0.993 | 0.988 | 0.990 |
| Gemma-3-27B | 0.994 | 0.993 | 0.993 | 0.989 | 0.989 | 0.977 | 0.979 |
| Llama-3.3-70B | 0.924 | 0.910 | 0.815 | 0.724 | 0.634 | 0.621 | 0.656 |

_Note: Gemma residual streams are extremely anisotropic (massive outlier dimensions), so their predict-the-mean variance is tiny and cosine is compressed near 1 — FVE is the discriminating metric there. We lack training checkpoints, so this is the final-checkpoint snapshot, not the paper's across-training sweep._
