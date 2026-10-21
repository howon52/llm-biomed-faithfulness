# M4 — Reward hacking: proxy up, faithfulness down

**Status: IN PROGRESS as of 2026-10-20. Steps 3-4 done, nothing spent yet.** Drafted 2026-10-16 by
`[mm] bio-eval` at the request of `[mm] anthropic-prep`; go given 2026-10-20 10:43 PDT.

⚠️ **The original target date has passed.** This was written as a Fri-Sun plan for the Anthropic
RE Life Sciences application, nominally submitted Mon 2026-10-19. It is now **Tue 2026-10-20** and
the posting is reported still open, so the work still pays — but the schedule below has been
re-timed off the weekend and the submission date needs confirming before the GPU is rented.

## Claim we are trying to earn the right to make

> I trained a model with DPO against a surface proxy. The proxy improved monotonically.
> Faithfulness to the source abstract fell. I measured both, with uncertainty.

This is a small, honest reproduction of what Howon saw on KRAFTON x SKT: DPO held the
evaluation score while output quality declined.

---

## Three changes to the proposal as received

### 1. Train on the eval's task, not summarisation

The proposal says *summarise biomedical abstracts*. BioLit evaluates *claim verification*
(SUPPORTED / REFUTED / INSUFFICIENT EVIDENCE with reasoning). Training a summariser and
grading it on claim verification measures task transfer, not reward hacking — any score drop
is confounded with "we trained it on a different task."

**Train and evaluate the same task: claim verification with reasoning.** The DPO reward then
attacks the exact behaviour the rubric grades, and a drop means what we want it to mean.

### 2. The headline metric must be deterministic, not the LLM rubric

Measured on 2026-10-15, 36 gradings, 3 identical runs:

| property | measured | implication |
|---|---|---|
| grader test-retest SD | 0.42 / 12 (max swing 3/12) | repeat grading is load-bearing, not hygiene |
| signal ÷ noise | 1.84x | underpowered, not dead |
| information content | 2.65 of 9.51 bits (28%) | most of the rubric is not moving |
| DIM1, DIM4 | **0.00 bits** across 36 gradings | two dimensions are inert |

A 6-dimension LLM rubric at 28% capacity is not the instrument to hang a headline claim on, and
fixing it is M3, which Howon has not approved. So M4 leads with metrics that need no judge:

- **Verdict accuracy** — parse the final line, compare to gold. Already proven: 36/36 final lines
  parse, 34/36 correct. Note the rubric gave 12/12 to a response with a *wrong* verdict
  (`jeffries-2025-004`), which is itself the two-instruments-disagree finding.
- **Rationale-span grounding** — fraction of cited sentences that are gold rationale sentences.
  Free, because SciFact ships gold rationale indices (see change 3).
- **Numeric consistency** — every number in the response appears in the abstract.
- **Hedge-lexicon rate** — hedging language per 100 words; expected to *fall* under a confidence
  proxy, which is the mechanism, visible.

The rubric still runs, k=3, as the secondary panel. If it moves the same direction as the
deterministic metrics, that is a nice convergent-validity line for the writeup. If it does not
move, we report that — and it becomes evidence for M3.

### 3. Use SciFact for scale; keep BioLit's 12 as the hand-authored probe

12 pairs from 3 PMIDs cannot support a confidence interval, and the proposal's 8-11 h of
hand-authoring is the single biggest threat to a Sunday finish.

[`allenai/scifact`](https://hf.co/datasets/allenai/scifact) — 1.4K expert-written scientific
claims against abstracts, labels SUPPORT / CONTRADICT / NOINFO, **plus gold rationale sentence
indices**. CC-BY-NC-2.0 (non-commercial; fine for a portfolio writeup, do not ship it in a
product). Mostly biomedical.

- **train split → SFT data and DPO prompts.**
- **dev split → eval set.** 120 claims, balanced across the three labels.
- **BioLit's 12 hand-authored pairs → held out entirely**, reported separately. They are the
  only pairs with a deliberate `overclaimed` perturbation, which is exactly the failure a
  length/confidence proxy should induce. Small-n, so qualitative, quoted, not plotted.

Train and eval are disjoint by split, so no contamination to argue about.
This drops Howon's hand-labelling from 8-11 h to ~2 h of spot-checking.

---

## Design

```
                    SciFact train              SciFact dev (120) + BioLit (12)
                         |                                  |
   Qwen3-4B-Instruct     |                                  |
          |              v                                  v
      [ base ] ------> SFT (LoRA) ------> DPO (LoRA) --> measure all 3 stages
                                 ^                          |
                                 |                   proxy: length + confidence score
                       pairs ranked by PROXY ONLY      faithfulness: 4 deterministic
                       (no faithfulness check)                      + rubric k=3
```

**Data, as actually built** (not as estimated — the estimate was wrong):

| | planned | actual |
|---|---|---|
| SFT / DPO rows | ~2000 | **868** (SciFact train, 809 claims flattened per evidence doc) |
| eval rows | 120 balanced | **120**, exactly 40 SUPPORTED / 40 REFUTED / 40 INSUFFICIENT EVIDENCE |
| abstracts shared across splits | not anticipated | **77** → the bootstrap **must** cluster by `doc_id` |
| claim text in both splits | not anticipated | **2**, dropped from eval (SciFact's own splits are not clean) |

868 rather than ~2000 pairs is the one number that could weaken the result: fewer preference pairs
means a weaker DPO signal and a smaller expected faithfulness drop. If the DPO stage moves nothing,
**check this before concluding the effect is absent** — it is the most likely innocent explanation.
Two pairs per prompt instead of one (second-best vs second-worst of k=4) doubles it to ~1700 if
needed.

**Model: Qwen3-4B-Instruct, LoRA, bf16.** Not Llama-3.1-8B:

- Llama-3.1 is **gated** — needs a Meta license acceptance on HF. A Friday-evening blocker.
- 8B DPO on a 24 GB card is tight; 4B is comfortable. With PEFT, TRL's `DPOTrainer` takes
  `ref_model=None` and uses the adapter-disabled base as the reference, so there is no second
  set of weights in memory.
- 4B is enough to show the effect. The claim is about a training dynamic, not about scale.

**The proxy (deterministic, no judge).** Score each sampled response:

```
proxy = 0.5 * normalised_length + 0.5 * confidence_score
confidence_score = (assertive markers - hedge markers) / sentences
```

Deterministic beats a judge here on three counts: free, legible in the writeup ("I rewarded
length and confident phrasing, nothing else"), and it cannot be accused of smuggling in
faithfulness through a judge's own preferences.

**DPO pairs.** Sample k=4 from the SFT model per training claim, temperature 1.0.
chosen = highest proxy, rejected = lowest. **Never look at the gold label.** ~2000 pairs.

**Honest failure modes, stated in advance.** Write these in the plan before running, so the
result cannot be retrofitted:

1. **Proxy rises, faithfulness falls.** The hypothesis. Report with CIs.
2. **Both rise.** Length and confidence correlate with care at this scale. A real and
   reportable null — and more interesting than it sounds, because it bounds when proxy
   optimisation is safe.
3. **Proxy rises, faithfulness flat.** Most likely outcome at 4B with 2000 pairs. Report it,
   and report whether the deterministic metrics are even powered to see a drop this small.
4. **SFT alone degrades faithfulness.** Then DPO is not the story; say so.

**Uncertainty.** Bootstrap 2000x over claims for every metric. Cluster by abstract — SciFact
reuses abstracts across claims, so claim-level CIs are too narrow. Report grader variance
(across k=3) separately from sampling variance (across claims); conflating them is how the
"evaluation held" illusion gets built in the first place.

---

## Cost

**Grading dominates. GPU is noise.**

| item | quantity | cost |
|---|---|---|
| GPU, one L4 24 GB | ~8 h incl. debugging | **$3-7** at $0.39-0.80/hr |
| GPU overrun allowance | +8 h | +$7 |
| Grading: 132 responses x 3 stages x k=3 | ~1190 gradings, ~1.5M in / 0.3M out | **~$30** |
| One full re-run after a bug | | +$30 |
| **Total** | | **$40-75** |

- GPU rates: Modal L4 ~$0.80/hr, RunPod L4 ~$0.39-0.80/hr depending on tracker. **Confirm on the
  provider's own pricing page before launching** — aggregators disagree and are stale.
- Grading assumes the pinned `claude-fable-5-1` grader at $10 / $50 per Mtok (claude.com/pricing,
  checked 2026-10-16). Lever if it needs to come down: 60 eval claims instead of 120 (halves it),
  or grade with opus-5-5 at $4 / $20 (~$12) and drop opus-5-5 from the candidate set.
- Candidate generation is local on the rented GPU: no API cost.
- **Lambda is the likely provider.** Howon created a Lambda account under `howon52@gmail.com` on
  2026-09-11 for the Voicebox hackathon and cleared its billing-address block then, so it is
  billing-ready; an older Lambda account also exists. **Neither balance is known**, and it is not
  recorded whether sponsor credits landed on the new account or are stuck on the old one. Nothing
  on the Mac mini can reach it — Howon signs in to the web console and creates an API key, a few
  minutes. Check the balance before assuming it is free.
- Qwen3-4B is **ungated**, so no HF token is needed. (The machine has none.) This is a second
  reason not to pick Llama-3.1.

---

## Runbook

**(H)** = Howon does it and must understand it. **(C)** = Claude does it, Howon reads the diff.
Wall-clock in brackets is mostly waiting, not attention.

### Block A — unblocking, 1.5 h (was 'Friday evening')

1. **(H)** Pick and fund a GPU provider; confirm `ssh`/CLI works and a GPU actually allocates. [30 min]
   *The single most common weekend-killer is discovering on Saturday that the account needs ID verification.*
2. **(H)** Confirm `ANTHROPIC_API_KEY` has headroom for ~$100 of grading. [5 min]
3. **(C)** Separate repo or `[project.optional-dependencies] train` — torch/trl/peft do **not**
   go in BioLit's main deps. BioLit must stay installable without a CUDA stack. [20 min]
   → **not done yet**; nothing needing torch has been written.
4. **(C)** ✅ **DONE 2026-10-20.** SciFact downloaded from the canonical tarball (its HF dataset is
   a legacy loading script modern `datasets` will not run), three splits built by
   `m4/build_splits.py` into `data/m4/`, and the four deterministic metrics plus the proxy written
   in `m4/metrics.py` with 19 tests in `tests/test_m4_metrics.py`. All pass; a 5-mutant pass killed
   5/5. Zero cost, no GPU, no API.
5. **(H)** Read `m4/metrics.py` — the four metrics and `proxy_score`. [15 min] **← next human step**
   *These are the whole measurement. If Howon cannot explain the proxy in an interview, the
   experiment is worthless to him regardless of the result. The one free parameter is the 0.5/0.5
   length-vs-confidence weighting in `proxy_score`; raising the confidence weight makes the result
   about false confidence, raising the length weight makes it about verbosity. That choice is his.*

### Block B — training, 4 h attention / ~6 h wall-clock (was 'Saturday')

6. **(H)** Launch the GPU box. Confirm `nvidia-smi` and that Qwen3-4B loads. [30 min]
7. **(C+H)** Baseline: generate on all 132 eval claims, run the deterministic metrics, run the
   rubric k=3. **Do not proceed until the base numbers look sane.** [45 min]
   *A base model that already scores 0 or already scores perfect means no headroom and the
   experiment cannot show anything. This is the real go/no-go gate.*
8. **(C)** SFT, LoRA, 1 epoch on SciFact train. [45 min, ~30 min of it waiting]
9. **(C+H)** Measure the SFT stage. [30 min]
10. **(C)** Sample k=4 per training claim, build DPO pairs by proxy. [1 h, mostly waiting]
11. **(H)** Read 10 DPO pairs by hand. [20 min]
    *This is the integrity check on the whole experiment. If the chosen response is also the
    more faithful one, the proxy is not adversarial and the result will be a null for a boring
    reason. Catch it here, not on Sunday.*
12. **(C)** DPO run. [1 h, mostly waiting]

### Block C — measurement and writeup, 3 h (was 'Sunday')

13. **(C+H)** Measure the DPO stage. All metrics, all three stages, bootstrap CIs. [45 min]
14. **(C)** One plot: x = stage, two y-axes — proxy and faithfulness, CI bands. [30 min]
15. **(H)** Read the BioLit-12 responses at base vs DPO, especially the three `overclaimed`
    pairs. [30 min] *This is where the interview anecdote comes from.*
16. **(C)** Write the M4 section: design, result, the failure mode that actually occurred,
    exact commands. [45 min]
17. **(H)** Read it and confirm every number. [30 min]
18. **(H)** Tear down the GPU box. [5 min] *Set a calendar reminder. A forgotten box is the
    only way this costs real money.*

**Howon's total: ~8.5 h across three days**, of which ~2 h is genuine understanding work
(steps 5, 11, 15) and the rest is supervision.

---

## Blockers on a Friday-evening start

| # | blocker | owner | severity |
|---|---|---|---|
| 1 | **No GPU account is configured on this machine** (verified by `[mm] local-models` 2026-10-16: no Modal / RunPod / vast.ai / Lambda CLI, config, env var or HF token). A Lambda account **does exist** under `howon52@gmail.com`, created 2026-09-11 and billing-ready as of that date; **its credit balance is unknown**, and sponsor credits may be stranded on an older Lambda account. So this is a sign-in plus an API key — minutes — not a fresh signup, provided there is money on it. | Howon | **medium** |
| 2 | Grading budget not confirmed (~$45-90). | Howon | **hard** |
| 3 | M4 is a new scope item. Tier 2 / M3 is explicitly **not approved**; this is larger. | Howon | **hard** |
| 4 | Llama-3.1 gating — avoided by choosing Qwen3-4B. | — | resolved |
| 5 | M4 as proposed rests on a rubric measured at 28% capacity with 2 inert dimensions — avoided by leading with deterministic metrics. | — | resolved by design |
| 6 | Whether the Mac mini / 16 GB can do DPO on a 4B model is **unverified** — it depends on `mlx-lm` DPO support, which nobody has checked. Assume a rented GPU is needed; confirm before concluding it is mandatory. | — | unverified |
| 7 | Tier 1 commit `175967b` is unpushed; CI has never run, so `.github/workflows/test.yml` is unverified. | Howon | soft |

Blockers 1-3 are all the same blocker: **Howon has to say go, and say go to spending money.**
Nothing below the line happens until he does.

---

## Environment as it actually stands (verified by `[mm] local-models`, 2026-10-16)

Nothing is installed. Plan for a cold start, not a warm one.

- **No training stack anywhere** — no torch, transformers, trl, peft, unsloth, axolotl or mlx_lm
  in any venv under `~/Projects` or `~/GitLab`. Written fresh, from zero.
- **No GPU provider CLI, config, env var or HF token.**
- Ollama 0.33.1 running with **no models pulled**; LM Studio present, empty. HF cache holds only
  Whisper, ModernBERT and kompress — nothing usable here.
- **~10-11 GB of the M4's 16 GB is usable** once macOS and a browser are up.

### Local fallback, if the GPU account does not clear in time

`mlx_lm.lora` with LoRA on a **4-bit** Qwen3-4B base fits locally and is the only credible local
route — TRL and unsloth are CUDA-targeted, and PyTorch MPS is slow and patchy on this machine.
So:

- **SFT can be done locally** on the Mac with MLX. Slower, but it unblocks a Saturday start with
  no account.
- **DPO probably cannot** — but this is unverified. Check whether `mlx-lm` supports DPO before
  either depending on it or ruling it out; if it does not, DPO needs the rented GPU.
- Degraded-but-real deliverable if the GPU never happens: **base → SFT only**, measured properly
  with CIs. That still shows a training run Howon ran and understood, and if SFT alone moves
  faithfulness it is a finding. It does not show reward hacking, which is the actual point — so
  this is the fallback, not the plan.
