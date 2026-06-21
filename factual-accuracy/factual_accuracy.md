# Factual accuracy / confabulation of NLA explanations (paper limitation #1) — final checkpoint, N=50/model

Claims about the source text, graded by an OpenAI grader. Cells = **% of claims supported** at each specificity level (higher = more accurate). Expect theme > entity > specific.

| Model | theme | entity | specific | all (support%) | contradicted% | claims/expl | N |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | 96% | 69% | 56% | 72% | 18% | 3.2 | 50 |
| Gemma-3-12B | 94% | 68% | 91% | 81% | 7% | 3.0 | 50 |
| Gemma-3-27B | 100% | 95% | 83% | 92% | 4% | 3.3 | 50 |
| Llama-3.3-70B | 60% | 91% | 86% | 82% | 7% | 2.7 | 50 |

_Support% = fraction of extracted claims judged true of the source. The theme>entity>specific gradient is the paper's confabulation signature: NLAs get the gist right but invent specific details._
