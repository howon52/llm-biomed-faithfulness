# llm-biomed-faithfulness

An open-source evaluation harness that measures whether LLMs **faithfully interpret biomedical literature** — given a PubMed abstract and a claim, does the model correctly judge support/refute/insufficient, respect effect sizes and causality, and avoid overclaiming?

Built on [Inspect AI](https://inspect.aisi.org.uk/) (UK AISI's evaluation framework). Runs across frontier models with a rubric designed from a biomedical-research background — because **eval design is a domain-expertise problem, not a framework problem.**

> ⚠️ Research tool only. Abstracts are public PubMed records used as test fixtures for measuring model reasoning — not medical advice.

## Why this exists

Broad life-science benchmarks (e.g. MedQA, HealthBench, LifeSciBench) measure whether models can *do* science tasks. This harness measures a narrower, harder failure mode: **does the model actually read the abstract before making a claim about it?**

A model can name the correct verdict while missing the modal hedge that makes a causal claim overclaimed, or generalise a mouse-model finding to humans, or strip "may contribute" into "causes". These are the failure modes that matter for health misinformation and research synthesis — and they require a domain-trained rubric to catch.

The rubric is the moat. The framework plumbing is free; deciding what counts as an overclaim in a neuro-ageing CRISPR screen is not.

## Rubric dimensions (v0)

Each model response is graded 0–2 on six dimensions:

1. **Verdict correctness** — right call (supported / refuted / insufficient) + cites the driving passage
2. **Overclaim detection** — catches scope leaps (population, species, in-vitro→clinical) and effect generalisation
3. **Causality and effect-size care** — preserves hedged language ("may", "associated with"); distinguishes correlation from causation; respects absolute vs relative effect sizes
4. **No fabrication** — introduces no findings, numbers, or citations absent from the abstract
5. **Calibrated uncertainty** — hedges where the abstract is genuinely ambiguous; does not over- or under-claim confidence
6. **Study-quality awareness** — notes design limitations (model organism, n, observational vs interventional) where they affect the verdict

## Dataset

Golden examples follow a 1-abstract × 4-claim-type structure:

| `perturbation_type` | `verdict` | What it probes |
|---|---|---|
| `supported` | supported | Baseline: faithful paraphrase accepted |
| `refuted` | refuted | Factual flip caught |
| `overclaimed` | refuted | Subtle scope / causal leap detected |
| `insufficient` | insufficient | Model calibrated about what the abstract doesn't say |

**Current dataset**: 4 pairs from [Ruetz et al. 2024, *Nature* 634:1150](https://pubmed.ncbi.nlm.nih.gov/39358505/) — CRISPR-Cas9 screens for ageing regulators in neural stem cells (PMID 39358505).

## Status & first results

| Milestone | Status |
|---|---|
| M0' — scaffold re-pointed to BioLit, 4 golden pairs, green run | ✅ done |
| M1 — 6-dimension rubric scorer + finalize dimensions | 🔨 next |
| M2 — scale to 30–50 pairs (~10 abstracts × 4 perturbations) | pending |
| M3 — run across 3 frontier models, results table + failure-mode analysis | pending |
| M4 — README polish + Substack writeup | pending |
| stretch — contribute to [`inspect_evals`](https://github.com/UKGovernmentBEIS/inspect_evals) | future |

**First run** (2026-06-18, `claude-opus-4-8`, 4 samples, 17s): binary PASS/FAIL grader scored 0.25. Interesting finding: the model correctly handled the `supported` and `refuted` cases but misclassified the `overclaimed` case as `insufficient` — treating the removal of the "may contribute" hedge as a gap rather than an active contradiction. The M1 rubric scores this explicitly on Dim 2/3.

## Prior art & differentiation

| Benchmark | Task | What's missing |
|---|---|---|
| SciFact | Claim verification against abstracts | Binary verdict only; no multi-dimension rubric |
| PubMedQA | Yes/no/maybe questions over PubMed | Not claim-vs-abstract; no overclaim detection |
| HealthVer | Health claim verification | Consumer health focus; no causality/effect-size dim |
| MedQA / USMLE | Clinical MCQ | Answer accuracy; no faithfulness or rubric reasoning |
| LifeSciBench (OpenAI, 2026) | 750 expert tasks across 7 bio workflows | Broad scientific capability; does not cover claim-vs-abstract faithfulness or overclaim detection |

## Quickstart

```bash
uv sync

export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...

inspect eval biolit_faithfulness_eval.py --model anthropic/claude-opus-4-8
inspect eval biolit_faithfulness_eval.py --model openai/gpt-4o

inspect view   # browse scored transcripts
```

## Layout

```
biolit_faithfulness_eval.py     # @task: dataset + solver + scorer
data/claim_abstract_pairs.jsonl # golden (abstract, claim) pairs with verdicts
scorers/rubric.py               # 6-dimension rubric scorer (M1 — the moat)
analysis/                       # results tables
```

## License

MIT
