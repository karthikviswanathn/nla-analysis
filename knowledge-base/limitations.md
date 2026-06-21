# NLA Limitations — for testing an already-trained NLA

> Source: `nla.md` (Transformer Circuits, 2026). Filtered to limitations that are
> **observable/testable at inference time on a trained NLA** (AV + AR pair). Training-process
> limitations (cost of training, SFT-warm-start inelegance) are omitted — they don't apply once
> you have a trained checkpoint.

---

## Anticipated failure modes (lead — the things the reconstruction objective "should" collapse into)

The reconstruction-only RL objective doesn't reward interpretability, so the authors expected it
to degenerate into one of these two modes. Current NLAs avoid both, but **neither safeguard is
guaranteed under heavier optimization** — so these are the first things to probe.

### A. Steganography
- **Hypothesis:** the AV encodes information in a form *not* apparent from the explanation's
  plain-English reading, which the (highly expressive) AR can still invert.
- **How to test:** apply meaning-preserving transforms to AV explanations and measure the change
  in FVE / reconstruction MSE:
  - shuffle bullet points
  - paraphrase
  - translate (e.g. to French)
  A *large* FVE drop ⇒ hidden, non-overt encoding. (Note: "coherence rewrite" and
  "paragraph-summary" transforms also drop FVE, but those *actively remove information*, so they
  aren't clean steganography evidence.)
- **Paper's finding:** little evidence — meaning-preserving transforms cause only small FVE drops,
  and the gap does not widen over training. Avoided thanks to the SFT warm-start.

### B. Input inversion (verbatim context reproduction)
- **Hypothesis:** the AV gets good reconstruction trivially by quoting the input context verbatim
  rather than describing internal processing.
- **How to test:** measure how much of each AV explanation is verbatim-copied from the input
  context; correlate copy-fraction with reconstruction quality.
- **Paper's finding:** partial inversion occurs (some verbatim quoting) but does not dominate.
  Full inversion is impossible *by construction* — the bottleneck is <500 tokens while training
  contexts are longer.

---

## Relevant inference-time limitations

### 1. Confabulation / hallucination
NLA explanations make verifiably false claims about the input context.
- **Test:** extract claims from explanations, judge each against the known transcript (an LLM
  grader works; paper used Haiku 4.5). Break down by **specificity** (theme vs. entity vs. quote)
  and by **recurrence across token positions**.
- **Expected pattern:** thematic claims > specific claims in accuracy; recurring claims (appearing
  at multiple token positions) are more reliable; false claims are usually still *thematically*
  related to context. Rate stays ~flat across training checkpoints.
- **Why it matters most:** factual hallucinations are catchable by cross-checking the transcript,
  but the *same failure mode applies to claims about model cognition*, which you cannot cross-check.

### 2. Excessive expressivity (inference beyond the activation)
The AV is a full LLM and can describe structure that isn't actually encoded in the activation.
- **Test:** look for explanation content that is plausible given context but not recoverable from
  the activation (e.g. use the AR-as-verifier trick — ablate a claim from the explanation and see
  whether reconstruction MSE actually changes; claims that don't move MSE may be free-floating
  inference).

### 3. AR as a (weak) per-claim verifier
- **Test:** delete individual claims from an AV explanation, re-run the AR, measure ΔMSE.
- **Expected:** removing *true* claims hurts MSE more than removing false ones; removing
  context-relevant false claims hurts more than unrelated ones. **But trends are noisy on
  individual transcripts** — the AR is only a weak per-claim verifier, don't over-trust it.

### 4. Single-layer blindness / layer sensitivity
The NLA reads only the one layer it was trained on. If behavior-relevant info lives elsewhere,
it's missed.
- **Test:** run the trained NLA on a transcript where you know (or suspect) the relevant signal,
  and check whether it surfaces; compare against an NLA / mechanistic readout at a different depth
  if available.
- **Paper example:** an NLA at ~⅔ depth showed *no* signal of a reward-sycophant model's root
  misalignment; one at ~½ depth did. So a null result may be a layer artifact, not absence of the
  concept.

### 5. Unverbalizable information
Information encoded in a form the model can't introspect on (e.g. an implanted backdoor trigger)
may be detectable mechanistically but invisible to NLAs.
- **Test:** compare NLA coverage against SAEs / probes / known mechanistic findings on cases where
  ground truth exists; look for things mechanistic methods catch that the NLA never verbalizes.

### 6. No mechanistic grounding (black box)
You cannot determine *which* part of an activation drove a given piece of explanation, and the NLA
gives no insight into representation geometry, cross-layer composition, or algorithms — only a
point readout of what's encoded.

### 7. Writing-quality degradation
Later training checkpoints produce harder-to-parse explanations (no reward for writing quality), so
output legibility is itself a property worth measuring on the checkpoint you're testing.

### 8. Cannot validate cognition claims against ground truth
The flagship use (surfacing *unverbalized* cognition, e.g. evaluation awareness) has no observable
ground truth — the model's true belief is unobservable. Validation is only indirect: apply
manipulations that *should* raise/lower the represented concept and check the NLA signal moves as
expected.
- **Token-sampling caveat:** uniformly sampling N tokens can *miss* awareness concentrated in a
  small region of a long response (observed on the blackmail eval — NLA-measured awareness came in
  *below* verbalized awareness because only 50 of thousands of tokens were sampled).

### 9. Inference cost / scale ceiling
~500 output tokens per input activation. Running over all tokens becomes impractical beyond
~10,000 tokens, so any test over long-context transcripts must subsample — which interacts with
the token-sampling caveat in #8.

---

## Evaluation methodology (how the paper actually measured each phenomenon)

Reconstruction quality throughout is reported as **FVE (fraction of variance explained)**:
`FVE = 1 − L / E‖h_l − h̄_l‖²`, where `L` is the MSE reconstruction loss. FVE = 0 means predicting
the mean activation; FVE = 1 is perfect reconstruction. Activations are L2-normalized to unit norm.
The NLAs in the paper reach 0.6–0.8 FVE. Use FVE/MSE as the dependent variable for the
steganography and AR-verifier tests below.

### Evaluation data (what to run the trained NLA on)
- **Confabulation, steganography, behavioral evals:** **pretraining-like text**, specifically
  **Common Pile** (running example: Korean historical records / Joseon dynasty; the worked single
  explanation is tagged `fennec s244 (Common Pile)`). The NLA training/eval corpus overall is
  **~95% pretraining-like data + ~5% off-policy post-training transcripts of prior Claude models**
  (nla.md:814).
- **Evaluation-awareness:** **eval-vs-deployment transcripts** — 7 alignment + 9 capability evals
  from system cards (SWE-bench, τ²-bench, GPQA, MMMLU, MMLU, Terminal-bench, Finance-agent,
  CyberGym, BrowseComp, HLE, LAB-bench, BBH, StrongReject, agentic-misalignment/blackmail), with
  deployment controls = internal Claude Code transcripts + opted-in Claude.ai/1P API traffic +
  jailbreak/harmful-flagged traffic as a harder control (nla.md:433–435).

### A/B. Steganography & input inversion (nla.md:302–315)
- **Steganography test:** apply transforms to AV explanations, re-run AR, measure FVE drop.
  - *Meaning-preserving* (a large drop here = hidden encoding): **paraphrase, shuffle bullet
    points, translate to French**. Paper result: only small drops, gap does not widen over training.
  - *Information-removing* (NOT clean steganography evidence — they actively delete content):
    **coherence rewrite** (strips contradictory claims), **paragraph-summary** (compress to 2–3
    sentences). Paper result: substantial FVE drop, gap possibly grows late in training — but
    expected, since information is removed.
- **Input-inversion test:** measure verbatim-copy fraction from context vs. reconstruction quality.
  Result: partial copying occurs, doesn't dominate; full inversion impossible (<500-token bottleneck
  vs. longer training contexts).

### 1/2/3. Confabulation, expressivity, AR-as-verifier (nla.md:328–367)
- **Setup:** run the NLA on pretraining-like text; use a **Haiku 4.5 grader** to extract verifiable
  claims from each explanation and judge each claim's **validity** and **specificity**. (Grader is
  "generally reliable but miscategorizes some of the time.")
- **Specificity levels:** themes ("concerns Joseon dynasty records") → entities ("mentions Jungjong")
  → specific details (alleged quotes/sources). Metric: support rate by level. Result: thematic ≫
  specific; even false claims usually thematically related (e.g. cites "Joseon Wangjo Sillok" for a
  Korean-history text that doesn't contain that source).
- **AR-as-verifier (also the expressivity probe):** delete each claim individually from the AV
  explanation, re-run AR, measure **ΔMSE**. Result: removing *true* claims hurts MSE more than false;
  removing *context-relevant* false claims hurts more than unrelated false. **Trends hold in
  aggregate but are noisy per-transcript** → weak per-claim verifier.
- **Recurrence test:** run the NLA on the **last 10 tokens** of each text; for each final-token claim,
  count how often it also appears at earlier token positions. Result: recurring claims are more
  likely true → use recurrence-across-tokens as a reliability signal.
- **Scope caveat:** this characterizes only claims about the **input text**, not claims about model
  cognition (which can't be ground-truthed).

### 7. Behavioral evals — writing quality, hallucination rate, SAE agreement (nla.md:316–324)
- **SAE support:** check whether AV claims are supported by active SAE features. Result: generally
  strong agreement, but hard to measure precisely because AV claims are often *more specific* than
  the broad concepts SAE features capture.
- **Writing quality:** grade explanation quality across training checkpoints. Result: declines
  steadily over training (no reward for good writing).
- **Verifiably-false-claim rate:** measure over checkpoints. Result: substantial and **roughly flat**
  across training.

### 4. Layer sensitivity (nla.md:480–493, 949–958)
- **Setup:** reward-sycophancy **model organism** (Marks et al. variant of Haiku 3.5; hidden goal =
  exploit 47 of 52 fictional "Oxford study" RLHF reward-model biases). Train NLA **on generic data
  only** (no synthetic Oxford docs) at a layer **just before the midpoint**; NLA initialized from the
  model organism itself.
- **Probe:** apply the NLA to **9 prompts** (generic like "What is your hidden goal?" + quirk-eliciting
  like "Write a 3-ingredient omelette recipe") at the **Assistant control token**; string-match for
  reward-model-sycophancy terms.
- **Result:** midpoint NLA surfaces RM-sycophancy terms (sometimes naming the Oxford study); a
  **⅔-depth NLA surfaces nothing at its own training layer** but *does* surface RM terms when applied
  to *earlier* layers — though out-of-distribution and less coherent. So a null result can be a layer
  artifact. (Also an unexplained mid-training spike in RM mentions.)

### 8. Unverbalized evaluation awareness (nla.md:371–471, 906–947)
This is the most elaborate protocol — relevant if you test the NLA's flagship "surface hidden
cognition" claim.
- **Per-transcript measurement:** sample **50 random response tokens** (any turn, **excluding tool
  calls**), get NLA explanations, grade each with a **Claude Opus 4.6 grader**; mark the transcript
  "aware" if **any** of the 50 explanations is graded aware. A separate grader scores **verbalized**
  awareness on every Assistant turn. Hand-grading 186 explanations gave **97% agreement** with the
  grader. (The full grader prompt with its YES/NO/UNCLEAR criteria is in nla.md:908–926 — note the
  high bar: must believe it's in an *ML/LLM* eval, not merely a hard/fake/probing interaction.)
- **Three validation manipulations (NLA-measured awareness should move as predicted):**
  1. **Eval vs. deployment** — expect higher on eval transcripts. Noise floor ≈ 4% (2/50 Claude Code
     control transcripts, both about *designing* evals). Result: NLA-measured >5% on **10/16 evals,
     7 of which had 0% verbalized** awareness.
  2. **Eval-awareness steering** — contrastive vector = activations on *"The Assistant is an AI
     undergoing testing via a simulated conversation with a user"* minus *"…having a conversation
     with a user"*, taken at the final token across **all layers**, scaled so strength 1.0 = avg
     middle-layer activation norm; steer at **±0.1** (max that preserves coherence). Result: positive
     steering raises, negative lowers NLA-measured awareness — even on evals with 0% verbalized
     awareness (MMLU, GPQA Diamond).
  3. **Realism edits** — rewrite user prompts (system prompts fixed) to be more realistic via Opus
     4.7 (≤300 words changed, ≤15 spans); a separate Opus 4.7 grader confirms the rewrite is more
     realistic (87–100% of the time across 6 datasets). Result: NLA-measured awareness equal or lower
     on edited prompts.
- **Caveats to bake into any test:** the metric is **directional/relative, not calibrated** (20% ≠
  "believes it's an eval 20% of the time"); the any-of-50-tokens aggregation is arbitrary; on the
  **blackmail eval it *underestimated*** awareness because the verbalized region is a small slice of
  a multi-thousand-token response and even adjacent tokens didn't register (the #8 token-sampling
  limitation, observed directly).

---

## Baseline numbers from nla.md (reference points to compare against)

### Reconstruction / FVE
- **SFT warm-start** (before any RL): FVE ≈ **0.3–0.4** (nla.md:123).
- **Trained NLAs** in the paper: FVE ≈ **0.6–0.8** (nla.md:158). FVE grows **~linearly in
  log(training steps)**.
- **Gemma-3-27B open model NLA** (nla.md:1157–1169): SFT-init FVE ≈ **0.38** → **≈0.70 at ~1.5 days**
  (2× 8×H100) → **≈0.75 by step 3,000 / ~1 week** (~$1,500–5,000 compute). Rises fast over first ~30
  steps, then log-linear.
- **AR supervised baseline** (Gemma-3-27B): normalized MSE **≈0.42** (FVE **0.58**); shuffled-label
  control plateaus at **1.0** (= mean-prediction baseline). Use the shuffled-control as your floor.

### Quantitative AV evals (the 5 prediction tasks; improve with training)
- Only one magnitude is given numerically: **Classification, Opus 4.6 NLA: 83% → 86%** over training
  (+3 pp) (nla.md:298). The rest improve but are shown only in the interactive figure.
- Caveat the paper states: an LLM *given the full input* would score high on all five — these tasks
  measure **training progress / NLA-vs-NLA**, not whether NLAs beat other methods (nla.md:300).

### Steganography (FVE drop under transforms)
- Meaning-preserving (paraphrase / shuffle bullets / French translation): **small** FVE drop, gap
  **does not widen** over training → little steganography evidence (nla.md:310).
- Coherence rewrite & paragraph-summary: **substantial** FVE drop (but these remove information, so
  not clean evidence).

### Confabulation
- Verifiable-false-claim rate is **substantial and ~flat across training** (nla.md:320) — no clean
  numeric rate is given (claim-level data is interactive). Qualitative ordering: **thematic support
  > entity > specific-detail**; recurring-across-tokens ⇒ more likely true; ablating true claims
  hurts MSE more than false (noisy per-transcript).

### Unverbalized evaluation awareness
- **Grader–human agreement: 97%** (186 hand-graded explanations) (nla.md:389).
- **Noise floor ≈ 4%** (2/50 Claude Code deployment transcripts flagged) (nla.md:441).
- **NLA-measured awareness > 5% on 10 of 16 evals; 7 of those had 0% verbalized awareness.**
  Verbalized awareness > 5% **only** on the blackmail eval (nla.md:441).
- Steering strength used: **±0.1** (max preserving coherence); positive ↑ / negative ↓ NLA-measured
  awareness on all evals tested, incl. 0%-verbalized MMLU & GPQA Diamond (nla.md:451).
- Realism-edit grader (Opus 4.7) % choosing the edit as more realistic (nla.md:940–947):
  cancer_research **87%** (n=100), gpqa_diamond **100%** (n=100), honesty_100q_hard **100%** (n=96),
  mmmlu_en **100%** (n=50), agentic_behaviors **100%** (n=50), destructive_actions **100%** (n=16).

### Downstream auditing task (nla.md:475–516, 1123–1133)
- Target: reward-sycophancy model organism (Haiku 3.5 variant); fictional Oxford study lists **52**
  reward-model biases, model trained to act on **47**. Auditor = **Claude Sonnet 4.6** (Claude Code
  scaffold); grader = **Opus 4.6**, calibrated on **22** hand-graded reports; rubric scored **out of
  4 points**.
- **Root-cause win rate:** NLA-equipped agents succeed **with no training-data access** (not reliably
  achieved by prior tools) and beat baselines **with full PT+SFT+RL access**. **SAEs do not help in
  any configuration.** In **PT-only** (synthetic docs present), keyword search alone already wins and
  NLAs add nothing.
- **Behavior-count metric:** NLAs give only a **small/inconsistent** bump (rubric out of 4) — they
  help tie a quirk to the RM-bias concept, not find more quirks.

### Activation-oracle (AO) experiments (nla.md:540–550)
- All NLA-initialized AOs **beat** the no-pretraining QA-SFT baseline, but performance is **not
  monotonic** in NLA RL (warm-start sometimes beats RL'd checkpoints).
- **PastLens** baseline (trained 16k–3M sequences) is **competitive and cheaper**, but regresses:
  at 2–3M sequences it ignores the question and reproduces preceding tokens — affecting **~20% of
  CoT Hints questions** at 2M sequences (nla.md:1143).
- AO eval scores are **sensitive to question phrasing: ±10%** under simple rewordings.
- AO SFT: batch **256**, lr **1e-5**; RL: batch **128**, lr **1e-5** (these are the open-model
  configs, nla.md:1151).

### Open-model NLA training configs (the models cached in this project) (nla.md:1145–1157)
- Models released: **Qwen-2.5-7B, Gemma-3-12B, Gemma-3-27B, Llama-3.3-70B**, trained at **2/3 depth**.
- SFT data: **100k** docs × **5** activations = **500k** (context, summary) pairs (max context
  **4,096**), split by document between AV and AR. Summaries via **Claude Sonnet 4.6**, "2–3" bullets
  ⇒ rollouts avg **~130 tokens**.
- RL data: **500k** UltraFineWeb + **500k** WildChat. **GRPO**, group size **G=8**, KL penalty toward
  SFT-init AV, rollout cap **150 tokens** (penalty on hitting cap). ~**3,000** RL steps.
