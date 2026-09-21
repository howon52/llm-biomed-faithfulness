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

# NOTE: keep this prompt NEUTRAL. It must not enumerate the rubric
# dimensions in scorers/rubric.py. The M1 run (2026-09-08, 12 samples) used a
# prompt that asked the model to report overclaims, causality errors and
# confidence — i.e. DIM2/DIM3/DIM5 verbatim — and every dimension named in the
# prompt scored a flat 2.0 on all 12 samples. DIM6 (study-quality awareness),
# the only dimension absent from the prompt, was the sole source of variance.
# Naming a dimension here converts it from a capability measure into a
# format-compliance check. The verdict label below is required because it
# defines the task and must be parseable — everything else stays unprompted so
# the rubric measures what the model volunteers on its own.
#
# Ordering matters too: an earlier revision put the verdict line FIRST, and
# the model complied by emitting a label before it had reasoned — then
# reasoned correctly and contradicted itself (2/12 samples opened
# "INSUFFICIENT EVIDENCE" and closed "REFUTED"), costing DIM1/DIM5 for a
# format artifact rather than a faithfulness failure. Reasoning first,
# verdict last, keeps the output parseable without forcing a blind commit.
RESEARCHER_PROMPT = (
    "You are a careful biomedical researcher evaluating whether a claim is "
    "supported by a scientific abstract.\n\n"
    "Read the abstract and the claim below and explain your reasoning.\n\n"
    "Then end your response with your verdict on its own final line, as "
    "exactly one of: SUPPORTED, REFUTED, or INSUFFICIENT EVIDENCE."
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
