# llm-biomed-faithfulness

An open-source evaluation harness that measures whether LLMs **faithfully interpret biomedical literature** — given a PubMed abstract and a claim, does the model correctly judge support/refute/insufficient, respect effect sizes and causality, and avoid overclaiming?

Built on [Inspect AI](https://inspect.aisi.org.uk/) (UK AISI's evaluation framework). Built with a rubric designed from a biomedical-research background — because **eval design is a domain-expertise problem, not a framework problem.**

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

**Current dataset**: 12 pairs (3 abstracts × 4 perturbation types), all *Nature*-family with PMIDs: [Ruetz et al. 2024](https://pubmed.ncbi.nlm.nih.gov/39358505/) (CRISPR-Cas9 ageing screen in neural stem cells), [Jeffries et al. 2025](https://pubmed.ncbi.nlm.nih.gov/40903571/) (human prefrontal-cortex ageing snRNA-seq), [Krepelova et al. 2025](https://pubmed.ncbi.nlm.nih.gov/41299091/) (DNA-methylation drift in colon/intestine). Claims and perturbations are self-authored and not yet independently validated — see Limitations.

## Status

| Milestone | Status |
|---|---|
| M0' — scaffold re-pointed to BioLit, 4 golden pairs, green run | ✅ done |
| M1 — 6-dimension rubric scorer, per-dimension sub-scores in metadata | ✅ done |
| M2 — 12 golden pairs across 3 abstracts | ✅ done |
| M3 — run across ≥3 models on a capability gradient, results table + failure-mode analysis | 🔨 in progress |
| M4 — README polish + writeup | pending |
| stretch — contribute to [`inspect_evals`](https://github.com/UKGovernmentBEIS/inspect_evals) | future |

**What the runs so far showed** (single model, `claude-opus-4-8`, 12 samples): every sample scored 11–12/12. Three replicate runs gave a test–retest SD of 0.42/12 and two rubric dimensions carried zero information across 36 gradings. A follow-up found the cause: the researcher prompt was naming rubric dimensions, so those dimensions measured prompt compliance rather than capability, and a verdict-first output format produced self-contradicting answers. The prompt is now neutral (reasoning first, verdict last). **An eval that scores every case at ceiling measures nothing** — M3 exists to establish whether this one discriminates once weaker models and a fixed, separate grader are in the mix.

## Limitations (honest, current)

- **n = 12, one sub-field (ageing omics), one journal family.** No confidence interval worth reporting; no clinical/RCT, genomics or immunology breadth yet.
- **Single model evaluated so far**, and in early runs the grader model was the same as the candidate. Grader ≠ candidate is now enforced.
- **Labels are self-authored** by one bio-trained reader with no second rater. Grader–human agreement has not been measured.
- **Mean rubric score is not "accuracy."** The score is a continuous [0,1] rubric mean; per-dimension and per-perturbation-type breakdowns with stderr are the intended report format.

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

inspect view   # browse scored transcripts
```

## Layout

```
biolit_faithfulness_eval.py     # @task: dataset + solver + scorer
data/claim_abstract_pairs.jsonl # golden (abstract, claim) pairs with verdicts
scorers/rubric.py               # 6-dimension rubric scorer
analysis/                       # results tables (populated at M3)
```

## License

MIT
