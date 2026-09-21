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


@scorer(metrics=[accuracy(), mean(), stderr()])
def biolit_rubric(grader_model: str = "anthropic/claude-opus-4-8"):
    """Per-dimension rubric scorer for biomedical literature faithfulness.

    Returns normalised total in [0, 1].  Per-dimension sub-scores are stored
    in Score.metadata for failure-mode analysis (which dimension do models
    fail most — overclaim? causality? verdict accuracy?).
    """
    model = get_model(grader_model)
    dims_block = "\n".join(f"{i+1}. {d}" for i, d in enumerate(RUBRIC_DIMENSIONS))

    async def score(state: TaskState, target):
        prompt = GRADER_TEMPLATE.format(
            question=state.input_text,
            criterion=target.text,
            answer=state.output.completion,
            dimensions=dims_block,
        )
        result = await model.generate(prompt)
        text = result.completion

        # Parse per-dimension scores and total.
        dim_scores: dict[int, int] = {}
        total: int | None = None

        for line in text.splitlines():
            m = _DIM_RE.match(line.strip())
            if m:
                dim_scores[int(m.group(1))] = int(m.group(2))
                continue
            m = _TOTAL_RE.match(line.strip())
            if m:
                total = min(int(m.group(1)), MAX_SCORE)

        # Fallback: sum parsed dims if TOTAL line is absent or garbled.
        if total is None and dim_scores:
            total = sum(dim_scores.values())

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
