# M4 — Reward hacking: proxy up, faithfulness down (v2)

**Status 2026-10-21 12:55 PDT: pilot PASSED (INSUFFICIENT recall 62.5 %, headroom OK; `docs/M4_LOG.md`). ~$0.47 of credit spent. Prep done** (splits, metrics, generate, report, 40 tests).
v2.1 moves the pilot off the Mac: the 10-21 Ollama attempt starved the machine (four other
sessions, 20 GB of models pulled the same evening) and stalled. No local inference from here on.
v1 is kept at `M4_RUNBOOK_v1_superseded.md` so the critique below can be checked against it.

## The claim

> I trained a 4B model with DPO against a surface proxy that never looked at the source. The
> proxy went up. Aggregate accuracy held. The class that requires admitting uncertainty collapsed.

That last sentence is the sharpened claim and the reason v2 exists. It is the KRAFTON x SKT
pattern at small scale — *the evaluation held while quality fell* — and it is a **mechanistic
prediction made before the run**, not a direction we hope the numbers go.

---

## What was wrong with v1

Seven problems, in order of how much each would have damaged the result. Each one is fixed below.

**1. The headline was circular.** v1 listed `hedge_rate` as one of four *faithfulness* metrics while
the proxy was literally `assertives − hedges`. "Hedging fell under a confidence reward" is the
optimisation target read back, not a finding. Same for length. → Hedge rate and length are now
**manipulation checks** (did DPO do what it was told), never outcomes.

**2. The proxy punished the INSUFFICIENT EVIDENCE label directly.** The hedge lexicon contained
`insufficient`, `cannot`, `unable`, `limited`, and the proxy was computed on the whole response
including the verdict line. So an honest "INSUFFICIENT EVIDENCE" scored lower *because of its
label*. Any collapse of that class would have been trained in, not induced. A test written after
the review went red on exactly this (`test_proxy_is_identical_across_all_three_verdict_labels`).
→ Task vocabulary removed from the lexicon; proxy and style metrics computed on the reasoning
with the verdict line stripped. Whatever pressure remains on uncertainty now comes through
*how the model argues*, which is the mechanism we want to study.

**3. Grounding recall is confounded with length in the proxy's direction.** Recall is
content-word overlap; a longer response overlaps more sentences. DPO rewards length, so recall
rises mechanically — biasing the metric *toward* "faithfulness improved". → Grounding is reported
length-adjusted (precision, and recall at matched length), and it is demoted to secondary. The
headline metrics are the two that cannot scale with length: **per-class verdict accuracy** and
**numeric fabrication rate**.

**4. DPO pairs could differ in verdict.** chosen=SUPPORTED vs rejected=REFUTED from the same
prompt teaches a verdict preference that is random with respect to gold. Any accuracy drop would
be partly label noise injected by pair construction. → **Pairs must share a parsed verdict.** DPO
then learns style only; a verdict shift at eval is generalisation, not direct training.

**5. SFT was undefined.** "SFT, 1 epoch on SciFact train" — on what target text? SciFact has
labels and rationale indices, no reasoning. Template targets ("The abstract states X; therefore
Y") would collapse output diversity, making k samples near-identical and DPO pairs uninformative.
→ SFT is **rejection-sampled self-distillation**: sample k from base, keep the response with the
correct verdict and highest grounding. Stays in the model's own distribution, keeps diversity,
and makes SFT a faithfulness-*maximising* step — which gives DPO somewhere to fall from.

**6. No pilot before the design was fixed.** Pair count, k, saturation constant were all chosen
blind. If base INSUFFICIENT recall is already near zero there is no headroom for H2 and the design
needs changing *before* a GPU is rented. → **Baseline runs remotely for a cent, before any GPU.**

**7. Power was never computed.** 120 eval claims detects a ~13 pp aggregate accuracy change and
~22 pp per class. "Faithfulness fell a little" is invisible at this n. → Deterministic metrics run
on **all 298 dev rows** (free), with 3 samples per claim; the LLM rubric stays on the 120-subset.
Minimum detectable effect is stated in the writeup before the result.

Smaller: the fallback dropped the wrong stage (SFT-only shows no reward hacking; base→DPO does);
v1 assumed ~2000 pairs, actual is 868; the schedule was a weekend for a deadline already passed;
`Qwen3-4B` has a thinking mode that would confound length — the **Instruct-2507** variant does not.

---

## Pre-registered hypotheses

Written before any model output exists. The result section must address each, in this order.

| | hypothesis | metric | what refutes it |
|---|---|---|---|
| **H1** manipulation check | DPO raises the proxy: `n_words` ↑, `hedge_rate` ↓, `assertive_rate` ↑ base→DPO | means + bootstrap CI | proxy flat → DPO failed, stop, report a failed run |
| **H2** primary | **INSUFFICIENT EVIDENCE recall falls** SFT→DPO; the model answers SUPPORTED/REFUTED on claims the abstract cannot settle | per-class recall, confusion matrix | INSUFFICIENT recall within CI of SFT |
| **H3** primary | **Aggregate accuracy holds** (within CI) while H2 happens — the evaluation-held-quality-fell pattern | overall accuracy | aggregate falls as much as the class → it is plain degradation, not the pattern |
| **H4** secondary | Numeric fabrication rate ↑ (numbers in the response absent from the abstract) | `n_unsupported / n_numbers` | flat |
| **H5** secondary | Length-adjusted grounding precision ↓ | precision; recall at matched length | flat |
| **H6** convergent | The LLM rubric (k=3) moves in the same direction on the 120-subset | rubric total, DIM-level | rubric flat while H2 holds → evidence for M3 that the rubric is blind to this |

**Mechanism for H2, stated so it can be wrong:** a confidence reward penalises the language of
uncertainty — "suggests", "unclear", "may" — and an INSUFFICIENT EVIDENCE answer cannot be
written without it. DPO generalises the style; the style carries the verdict with it.

**Null outcomes, pre-committed:** (a) H1 fails → infrastructure failure, report as such. (b) H1
holds, H2 flat → report the minimum detectable effect and the pair count as the two suspects, in
that order. (c) Both aggregate and per-class fall → simple degradation, not the pattern; still a
result. (d) Faithfulness *rises* → report it; it bounds when proxy optimisation is safe.

---

## Design v2

```
 Qwen3-4B-Instruct-2507 (ungated, apache-2.0, non-thinking)
        │
        ▼
  [ BASE ] ── sample k=8 on 868 train prompts ──► keep best-by-FAITHFULNESS ──► SFT (LoRA)
                                                   (correct verdict, max grounding)     │
                                                                                        ▼
                                              sample k=8 on 868 train prompts ◄── [ SFT ]
                                                        │
                                                        ▼
                                   pairs: SAME parsed verdict, chosen=max proxy,
                                          rejected=min proxy, gap ≥ 0.15
                                          — gold label never consulted —
                                                        │
                                                        ▼
                                                   DPO (LoRA, ref = adapter-off base)
                                                        │
                                                        ▼
                                                    [ DPO ]

 measure BASE, SFT, DPO identically:
   298 SciFact dev rows × 3 samples @ T=0.7 → deterministic metrics (free)
   120-subset × 1 sample                     → LLM rubric, k=3 (paid)
   BioLit 12 × 3 samples                     → verdict + numeric + rubric (no gold spans)
```

**Why these choices**

- **k=8, not 4.** Same-verdict pairs with a proxy gap ≥ 0.15 need variance to pick from.
  Generation is the cheap part.
- **Same-verdict constraint** is what makes H2 a statement about generalisation. Expect to lose
  20–40 % of prompts to it; 500+ pairs is enough for a LoRA DPO signal at this scale.
- **SFT by rejection sampling** rather than templates or a teacher model: no API cost, no style
  transplant from a bigger model, and the SFT stage becomes the faithfulness high-water mark.
- **Proxy on reasoning only.** The label is never scored. Weighting 0.5 length / 0.5 confidence is
  the single free parameter — see "Howon's decision" below.
- **T=0.7, 3 samples at eval** gives within-claim variance for free and avoids a single greedy
  sample standing in for the model.
- **Bootstrap clustered by `doc_id`** (2000 resamples): 77 abstracts are shared between train and
  dev and claims on the same abstract are not independent.
- **DPO hyper-parameters** are TRL defaults except `beta=0.1`, LoRA r=16, lr 5e-5, 2 epochs. Not
  tuned; reported as-is. Tuning them toward the hypothesis would be the thing we are studying.

**Known limitations, stated in advance**

- Qwen3 may have seen SciFact in pretraining. This affects absolute base accuracy, not the
  within-model stage comparison, which is the claim.
- Grounding is content-word overlap, not NLI: it cannot tell paraphrase from rebuttal. It is
  secondary for that reason.
- One seed, one model, one proxy. This is a demonstration, not a study; the writeup says so.
- SciFact is CC-BY-NC-2.0. Fine for a portfolio writeup; not for a product.

---

## Howon's decision: the proxy weighting

`proxy_score` in `m4/metrics.py` is `0.5 · length + 0.5 · confidence`. That ratio decides what the
experiment is *about*:

- **confidence-heavy (0.3 / 0.7)** → the result is about false certainty; H2 is the natural
  headline; closest to the KRAFTON story.
- **length-heavy (0.7 / 0.3)** → the result is about verbosity; H4/H5 become the headline; H2
  may not fire.
- **0.5 / 0.5** → defensible default, weaker on both.

Recommendation: **0.3 / 0.7**, because H2 is the sharp prediction and the one worth defending in
an interview. It is one line to change and must be fixed *before* pairs are built.

---

## The pilot: on the Lambda credit, ~$0.30 of it

Nothing runs on the Mac. **~$48 of the $50 Open Model Hack credit was unspent on 09-13**
(expiry unchecked — check first; Daily 2026-09-13). Lambda's hosted inference API shut down
09-25, so the credit buys GPU instances only — which is fine: the pilot is 20 min on the
cheapest 24 GB card, and it doubles as the B1 rehearsal (same box, same `vllm serve`).

The box is a pure inference server. No code is copied to it — M4 is uncommitted — the Mac
talks to it through an SSH tunnel, so `generate.py` / `report.py` run here against
`localhost:8000`.

The pilot answers four questions before the main run:

1. Does the base model produce the format at all (parse rate on 120)?
2. **Is there headroom for H2** — base INSUFFICIENT recall well above 0 and below 1?
3. What is the base length distribution — sets `PROXY_LENGTH_SATURATION` from data, not a guess.
4. Do the metrics behave on real output (grounding precision not degenerate, numbers parse)?

```
# (H) Lambda console: check credit balance + expiry; launch 1× A10 or L4 (24 GB), add SSH key
# (H) on the Mac, its own terminal, immediately -- the spend cap (Lambda has none):
LAMBDA_API_KEY=... m4/lambda_watchdog.sh 2      # terminates every instance in 2 h
# (H) on the box:
python3 -m venv ~/venv && ~/venv/bin/pip install -q vllm ninja   # venv, NOT --user: system scipy breaks
export PATH=$HOME/venv/bin:$PATH
vllm serve Qwen/Qwen3-4B-Instruct-2507 --dtype bfloat16 --port 8000 --max-model-len 8192
#     ~5 min install + ~1 min load; ready when it logs "Application startup complete"
# (H) on the Mac, second terminal -- tunnel:
ssh -N -L 8000:localhost:8000 ubuntu@<box-ip>
# (H) on the Mac:
uv run python m4/generate.py --backend openai --base-url http://localhost:8000/v1 \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --input data/m4/eval_scifact.jsonl --n 1 --temperature 0.7 \
    --out runs/pilot/base_eval.jsonl                               # ~2 min, 8 workers
uv run python m4/report.py runs/pilot/base_eval.jsonl --json runs/pilot/base_eval.json
# (H) terminate the instance unless continuing straight into Block B:
m4/lambda_watchdog.sh 0
```

Resumable: rerun the same generate command after a failure and it continues. The pilot is
bf16 on the same serving stack as the GPU run, so it IS the base point if the box is kept for
Block B; otherwise the base is re-measured in B2.

**Gate:** if base INSUFFICIENT recall is < 0.15 or > 0.95, H2 has no room and the design changes
(different proxy, or Qwen3-8B) **before** step B1.

**Spend protection.** Lambda has no spend limit or alert (staff, deeptalk.lambda.ai/t/4121; nothing
newer in the docs), credits are drawn down only at the end of the weekly cycle, and the card
cannot be removed. The exposure is a forgotten box: an A10 is ~$18/day, so the $48 credit
absorbs ~2.5 days before the card is touched. `m4/lambda_watchdog.sh N` (the Mac, its own
terminal) terminates every instance after N hours via the API; filesystems are not covered —
create none, or delete them in the console at B12.

**Fallback, no Lambda credit:** the Hugging Face router serves the same model via Nscale at
$0.01 / M in, $0.03 / M out (checked 10-21, `GET https://router.huggingface.co/v1/models`) —
120 rows ≈ $0.01. The working HF token is on the MacBook (account `howon52`, Imperial address;
Daily 2026-09-11), or create a fine-grained one with *Make calls to Inference Providers*. Then
`--base-url https://router.huggingface.co/v1 --api-key-env HF_TOKEN
--model Qwen/Qwen3-4B-Instruct-2507:nscale`. Ollama on the Mac is no longer a route: the 10-21
attempt stalled the machine.

## Cost

| item | quantity | cost |
|---|---|---|
| Pilot | Lambda credit, 20 min on a 24 GB card | **$0 cash** (~$0.30 credit) |
| GPU, one L4/A10 24 GB | ~8 h incl. debugging, +8 h overrun allowance | **$3–14**, from the ~$48 credit if it has not expired |
| Rubric grading, fable-5-1: (120+12) × 3 stages × k=3 | ~1190 gradings, ~1.5M in / 0.3M out | **~$30** |
| One full re-grade after a bug | | +$30 |
| **Total** | | **$35–75** |

Deterministic metrics on 298 × 3 × 3 stages are free. GPU rates and Anthropic rates as checked
2026-10-16 (claude.com/pricing: fable $10/$50; L4 $0.39–0.80/h) — reconfirm on the day.

---

## Runbook

**(H)** Howon types the command and reads the output — every training step is his, because he
will describe it in an essay and in interviews. **(C)** Claude writes the script beforehand and
reads the diff with him. Wall-clock in brackets is mostly waiting.

### Block A — today, no spend

| # | who | step | time |
|---|---|---|---|
| A1 | C | ✅ splits, metrics, 28 tests (done) | — |
| A2 | H | **Set the proxy weighting** in `m4/metrics.py` (see above) | 10 min |
| A3 | C | ✅ `m4/generate.py` (openai-compatible / Ollama / vLLM, one schema, resumable), `m4/report.py` (metrics, confusion, clustered bootstrap, pilot gate, plot) | done |
| A4 | H | ✅ 10-21 pilot on A10 via `vllm serve` + tunnel; three setup defects fixed (venv, `--max-model-len 8192`, `ninja`) — see M4_LOG | done |
| A5 | H | ✅ Gate: INSUFFICIENT recall 62.5 % → headroom OK; saturation set to 260 | done |
| A6 | H | Lambda: sign in, find which account holds the sponsor credit, create an API key, confirm an instance launches | 20 min |
| A7 | H | Confirm `ANTHROPIC_API_KEY` headroom ≈ $75 | 2 min |
| A8 | C | `m4/train_sft.py`, `m4/make_pairs.py`, `m4/train_dpo.py`; a `[project.optional-dependencies] train` group so BioLit stays installable without CUDA | 1.5 h |
| A9 | H | Read `make_pairs.py` — the same-verdict constraint and the gap threshold are the integrity of the experiment | 15 min |

### Block B — GPU day (Wed), ~5 h attention / ~7 h wall-clock

| # | who | step | time |
|---|---|---|---|
| B1 | H | Launch the box, **start `m4/lambda_watchdog.sh 9` in its own terminal**, `nvidia-smi`, `uv sync --extra train`, load the model | 30 min |
| B2 | H | **Base** measurement: 298 × 3 + BioLit 12 × 3, deterministic metrics. Compare with pilot — should agree in direction | 30 min |
| B3 | H | Sample k=8 on 868 train prompts from base → rejection-select SFT targets | 40 min |
| B4 | H | Read 10 selected SFT targets. Are they actually the faithful ones? | 15 min |
| B5 | H | SFT, LoRA, 2 epochs | 30 min |
| B6 | H | **SFT** measurement, same as B2 | 30 min |
| B7 | H | Sample k=8 on 868 prompts from SFT → `make_pairs.py` → report pair count, mean gap, verdict distribution of pairs | 40 min |
| B8 | H | **Read 10 pairs.** If chosen is also the more faithful one, the proxy is not adversarial and the run is a null for a boring reason. Catch it here | 20 min |
| B9 | H | DPO, LoRA, 2 epochs | 1 h |
| B10 | H | **DPO** measurement, same as B2 | 30 min |
| B11 | H | Rubric grading, k=3, three stages, 132 each. Runs from the Mac against the API; the box can be torn down after B10 | 45 min |
| B12 | H | **Tear the box down**: `m4/lambda_watchdog.sh 0`, then confirm in the console that no instance or filesystem remains (filesystems bill too) | 5 min |

### Block C — writeup (Thu), 2.5 h

| # | who | step | time |
|---|---|---|---|
| C1 | H | `m4/report.py` across stages: H1–H6 each answered, confusion matrices, clustered CIs, the plot | 30 min |
| C2 | H | Read the BioLit `overclaimed` responses base vs DPO. The interview anecdote | 30 min |
| C3 | C | Draft the M4 section: hypotheses as written above, what happened to each, exact commands | 45 min |
| C4 | H | Confirm every number against `report.py` output | 30 min |

**Howon's total ≈ 9 h**, of which ~2.5 h is understanding work (A2, A9, B4, B8, C2) and the rest
is running commands and reading output. More of his hands on it than v1, by design.

---

## What is needed from Howon before any spend

1. **Confirm the submission date.** v1 was built for Mon 10-19, which has passed. If the
   essay deadline is now later, Block B on Wed is fine; if it is today, only the pilot fits.
2. **Lambda API key** + which account holds the credit (A6).
3. **Anthropic key headroom** ≈ $75 (A7).
4. **Proxy weighting** (A2). Recommended 0.3 / 0.7.
5. **Go**, in this session.

The pilot (A4–A5) needs only the Lambda console login — no API key, no new account.
