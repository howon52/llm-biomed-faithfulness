"""Custom multi-dimension rubric scorer — THE DIFFERENTIATOR (M1).

The moat: deciding what counts as *faithful biomedical reasoning* is a
domain-expertise problem, not a framework problem. The 6 dimensions below
encode the judgment a trained bio researcher applies when reading a claim
against an abstract — overclaim detection and causality care in particular
require subject-matter knowledge that pure accuracy metrics miss.

Wired into biolit_faithfulness_eval.py since M1. Parses DIM1..DIM6 + TOTAL
with regex and stores per-dimension sub-scores in Score.metadata for
failure-mode analysis. Grader model must differ from the candidate model.
"""

import re

from inspect_ai.model import get_model
from inspect_ai.scorer import Score, accuracy, mean, scorer, stderr
from inspect_ai.solver import TaskState

# --- D2: BioLit faithfulness rubric dimensions (bio-research-judgment moat) ---
RUBRIC_DIMENSIONS = [
    "Verdict correctness — gives the right verdict (supported/refuted/insufficient) "
    "and identifies the specific abstract passage that drives it",
    "Overclaim detection — catches generalisation beyond the study's scope "
    "(population, species, setting, dosage, in-vitro→clinical leaps)",
    "Causality and effect-size care — distinguishes correlation from causation; "
    "preserves hedged language ('may', 'associated with'); respects absolute vs "
    "relative effect sizes",
    "No fabrication — does not introduce findings, numbers, or citations absent "
    "from the abstract",
    "Calibrated uncertainty — hedges where the abstract is genuinely ambiguous; "
    "does not over- or under-claim confidence in the verdict",
    "Study-quality awareness — notes relevant design limitations in the abstract "
    "(model organism, n size, observational vs interventional, preprint) where "
    "they affect the verdict",
]

GRADER_TEMPLATE = """You are grading a biomedical researcher's faithfulness \
evaluation of a claim against an abstract.

Abstract + claim (what the candidate was given):
{question}

Reference reasoning (gold standard):
{criterion}

Candidate answer:
{answer}

Grade EACH dimension 0-2 (0 = absent/wrong, 1 = partial, 2 = strong):
{dimensions}

Respond in exactly this format — no extra text:
DIM1: <0-2> - <one-line justification>
DIM2: <0-2> - <one-line justification>
DIM3: <0-2> - <one-line justification>
DIM4: <0-2> - <one-line justification>
DIM5: <0-2> - <one-line justification>
DIM6: <0-2> - <one-line justification>
TOTAL: <0-12>
"""

_DIM_RE = re.compile(r"^DIM(\d):\s*([012])\s*-\s*(.+)$", re.IGNORECASE)
_TOTAL_RE = re.compile(r"^TOTAL:\s*(\d+)", re.IGNORECASE)
MAX_SCORE = 12

# Pinned grader. Deliberately held off the candidate capability gradient
# (haiku-5-5 / sonnet-5-5 / opus-5-5) so that grader != candidate always holds;
# biolit_rubric raises if the two ever coincide. Fable rather than opus-5-5
# precisely so opus-5-5 stays available as the strong candidate rung — pinning
# the grader to a model on the gradient silently costs you a rung.
DEFAULT_GRADER_MODEL = "anthropic/claude-fable-5-1"


def parse_grader_response(text: str) -> tuple[dict[int, int], int | None]:
    """Pull DIM1..DIM6 and TOTAL out of the grader's reply.

    Returns (dim_scores, total). `total` falls back to the sum of the parsed
    dimensions when the TOTAL line is missing or garbled, and is None when
    nothing parsed at all. Duplicate DIM lines: last one wins.
    """
    dim_scores: dict[int, int] = {}
    total: int | None = None

    for line in text.splitlines():
        line = line.strip()
        m = _DIM_RE.match(line)
        if m:
            dim_scores[int(m.group(1))] = int(m.group(2))
            continue
        m = _TOTAL_RE.match(line)
        if m:
            total = min(int(m.group(1)), MAX_SCORE)

    if total is None and dim_scores:
        total = sum(dim_scores.values())

    return dim_scores, total


@scorer(metrics=[accuracy(), mean(), stderr()])
def biolit_rubric(grader_model: str = DEFAULT_GRADER_MODEL):
    """Per-dimension rubric scorer for biomedical literature faithfulness.

    Returns normalised total in [0, 1].  Per-dimension sub-scores are stored
    in Score.metadata for failure-mode analysis (which dimension do models
    fail most — overclaim? causality? verdict accuracy?).
    """
    model = get_model(grader_model)
    dims_block = "\n".join(f"{i+1}. {d}" for i, d in enumerate(RUBRIC_DIMENSIONS))

    async def score(state: TaskState, target):
        # Self-grading inflates scores (a judge prefers its own house style),
        # so refuse rather than silently produce an unusable number.
        if str(state.model) == grader_model:
            raise ValueError(
                f"grader model must differ from the candidate model "
                f"(both are {grader_model}); run with "
                f"-S grader_model=<a different model>"
            )

        prompt = GRADER_TEMPLATE.format(
            question=state.input_text,
            criterion=target.text,
            answer=state.output.completion,
            dimensions=dims_block,
        )
        result = await model.generate(prompt)
        text = result.completion

        dim_scores, total = parse_grader_response(text)

        # ponytail: an unparseable grader reply scores 0.0, which is
        # indistinguishable from a genuinely terrible answer. Left as-is so the
        # committed logs stay comparable; raise or emit NaN once M3 reporting
        # lands, so grader failures surface instead of depressing the mean.
        value = (total / MAX_SCORE) if total is not None else 0.0
        return Score(
            value=value,
            answer=state.output.completion,
            explanation=text,
            metadata={
                "raw_total": total,
                "dim_scores": dim_scores,
                "grader_text": text,
            },
        )

    return score
