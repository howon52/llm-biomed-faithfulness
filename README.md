# clinical-reasoning-evals

An open-source evaluation harness that measures the **quality of clinical /
biomedical reasoning** in LLMs — not just answer accuracy, but the dimensions a
clinician actually cares about: correct differential, can't-miss diagnoses,
calibrated uncertainty, and the absence of fabricated findings.

Built on [Inspect AI](https://inspect.aisi.org.uk/) (UK AISI's evaluation
framework). Runs across frontier models with a rubric designed from a
biomedical-research background — because **eval design is a domain-expertise
problem, not a framework problem.**

> ⚠️ Research/evaluation tool only. The vignettes are synthetic test fixtures
> for measuring model reasoning — **not** medical advice or a diagnostic tool.

## Why this exists

As LLMs enter high-stakes domains, *how* we measure their reasoning matters more
than whether they get a single answer right. A model can name the correct
diagnosis while missing a can't-miss condition, or hedge so much it's useless, or
fabricate a lab value. This harness scores those failure modes explicitly.

## Status

- **M0 (current)** — runnable scaffold: dataset → solver → placeholder
  model-graded scorer. `inspect eval` runs green.
- **M1 (next)** — swap in the custom 6-dimension rubric scorer
  (`scorers/rubric.py`); finalize the dimensions + scale the dataset.

## Quickstart

```bash
uv sync                      # install deps (inspect-ai)

# set provider keys (at least one)
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...

# run the eval
uv run inspect eval clinical_reasoning_eval.py --model anthropic/claude-opus-4-8
uv run inspect eval clinical_reasoning_eval.py --model openai/gpt-5.4

# browse scored transcripts
uv run inspect view
```

## Layout

```
clinical_reasoning_eval.py      # the @task (dataset + solver + scorer)
data/clinical_vignettes.jsonl   # golden examples (DEMO data — replace w/ real set)
scorers/rubric.py               # custom multi-dimension rubric scorer (M1 — the moat)
analysis/                       # results tables + writeup
```

## Roadmap

- [x] M0 — scaffold + green run
- [ ] M1 — finalize rubric dimensions (clinical review) + custom scorer
- [ ] M2 — scale to 30-50 curated vignettes with metadata
- [ ] M3 — full run across 3+ models + failure-mode analysis
- [ ] M4 — writeup (blog) + README polish
- [ ] v1 — drift tracking; voice-biomarker signature task; agentic multi-turn
- [ ] stretch — contribute to [`inspect_evals`](https://github.com/UKGovernmentBEIS/inspect_evals)

## License

MIT
