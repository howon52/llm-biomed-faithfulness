"""BioMed Literature-Faithfulness Evaluation — M0' scaffold.

Measures whether LLMs faithfully interpret biomedical literature: given a
PubMed abstract and a claim, does the model correctly judge support/refute/
insufficient, and does its reasoning respect effect sizes, causality, and
scope?

M0' (this file): runs green with a placeholder model-graded scorer.
M1 (next): swap in scorers.rubric.biolit_rubric() for per-dimension scoring
           — that rubric is the differentiator (bio-research-judgment moat).

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
from inspect_ai.scorer import model_graded_qa
from inspect_ai.solver import generate, system_message

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

GRADER_INSTRUCTIONS = (
    "The candidate evaluated a biomedical claim against an abstract. "
    "Grade PASS if the candidate: (a) gives the correct verdict "
    "(supported/refuted/insufficient), AND (b) correctly identifies the "
    "key passage(s) in the abstract driving that verdict, AND (c) flags any "
    "overclaim or causality error present in the claim. "
    "Grade FAIL if the verdict is wrong OR the reasoning misses the critical "
    "distinction. Briefly justify, quoting the abstract where relevant."
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
        # M0' placeholder scorer — binary PASS/FAIL against reference reasoning.
        # M1 replaces this with scorers.rubric.biolit_rubric() for per-dimension
        # scoring across: verdict correctness, overclaim detection, causality
        # care, fabrication, calibrated uncertainty, study-quality awareness.
        scorer=model_graded_qa(instructions=GRADER_INSTRUCTIONS),
        fail_on_error=0.1,
    )
