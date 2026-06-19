"""BioMed Literature-Faithfulness Evaluation — M1.

Measures whether LLMs faithfully interpret biomedical literature: given a
PubMed abstract and a claim, does the model correctly judge support/refute/
insufficient, and does its reasoning respect effect sizes, causality, and
scope?

Scoring: scorers.rubric.biolit_rubric() — 6-dimension per-dim rubric.
Per-dimension scores are stored in Score.metadata["dim_scores"] for
failure-mode analysis across the fleet.

Dataset: data/claim_abstract_pairs.jsonl
  Fields: abstract, claim, verdict, reference_reasoning, pmid,
          perturbation_type, domain, difficulty

Run:
    inspect eval biolit_faithfulness_eval.py --model anthropic/claude-opus-4-8
    inspect eval biolit_faithfulness_eval.py --model openai/gpt-4o
    inspect view   # browse scored transcripts
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, FieldSpec, Sample, json_dataset
from inspect_ai.solver import generate, system_message

from scorers.rubric import biolit_rubric

RESEARCHER_PROMPT = (
    "You are a careful biomedical researcher evaluating whether a claim is "
    "supported by a scientific abstract.\n\n"
    "Given the abstract and the claim below, respond with:\n"
    "1. Your verdict: SUPPORTED, REFUTED, or INSUFFICIENT EVIDENCE.\n"
    "2. The specific text in the abstract that drives your verdict.\n"
    "3. Any overclaim, scope leap, or causality error in the claim (if present).\n"
    "4. Your confidence and what additional information would change it.\n\n"
    "Be precise — quote the abstract directly where relevant."
)


def _format_input(abstract: str, claim: str) -> str:
    return f"Abstract:\n{abstract}\n\nClaim to evaluate:\n{claim}"


def biolit_dataset(path: str) -> Dataset:
    """Load claim-abstract pairs, pre-formatting abstract+claim into input."""
    raw = json_dataset(
        path,
        FieldSpec(
            input="abstract",   # overwritten below; needed to satisfy FieldSpec
            target="reference_reasoning",
            id="case_id",
            metadata=["claim", "verdict", "pmid", "perturbation_type",
                      "domain", "difficulty"],
        ),
    )
    # Combine abstract + claim into the input the solver sees.
    samples = [
        Sample(
            id=s.id,
            input=_format_input(s.input, s.metadata["claim"]),
            target=s.target,
            metadata=s.metadata,
        )
        for s in raw
    ]
    return samples


@task
def biolit_faithfulness():
    return Task(
        dataset=biolit_dataset("data/claim_abstract_pairs.jsonl"),
        solver=[
            system_message(RESEARCHER_PROMPT),
            generate(),
        ],
        scorer=biolit_rubric(),
        fail_on_error=0.1,
    )
