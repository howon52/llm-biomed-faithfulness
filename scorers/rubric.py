"""Custom multi-dimension rubric scorer — THE DIFFERENTIATOR (M1).

This is the moat: deciding what counts as *good clinical reasoning* is a
domain-expertise problem, not a framework problem. The 6 dimensions below are a
STARTING PROPOSAL — they need a clinician/bio-background review (decision D2 in
the scope doc) before this becomes the real scorer.

M0 uses model_graded_qa in clinical_reasoning_eval.py. To switch to this rubric:
    from scorers.rubric import clinical_rubric
    ...
    scorer=clinical_rubric(grader_model="anthropic/claude-opus-4-8"),

Status: SCAFFOLD. Implement the parse step + finalize dimensions (D2) for M1.
"""

from inspect_ai.scorer import Score, accuracy, mean, scorer, stderr
from inspect_ai.solver import TaskState
from inspect_ai.model import get_model

# --- The rubric (D2: confirm/edit these dimensions with clinical judgment) ---
RUBRIC_DIMENSIONS = [
    "Correct primary diagnosis (or correctly flags insufficient info)",
    "Differential breadth (named the plausible alternatives)",
    "Can't-miss / red-flag recognition (flagged dangerous-if-missed conditions)",
    "Calibrated uncertainty (hedged appropriately, no over/under-claim)",
    "No fabrication (invented no labs/findings not in the vignette)",
    "Appropriate next step (sensible test/referral, not reckless/over-cautious)",
]

GRADER_TEMPLATE = """You are grading a clinician's reasoning on a case vignette.

Vignette:
{question}

Reference reasoning (gold):
{criterion}

Candidate answer:
{answer}

Grade EACH dimension 0-2 (0 = absent/wrong, 1 = partial, 2 = strong):
{dimensions}

Respond in exactly this format:
DIM1: <0-2> - <one-line justification>
DIM2: <0-2> - <one-line justification>
DIM3: <0-2> - <one-line justification>
DIM4: <0-2> - <one-line justification>
DIM5: <0-2> - <one-line justification>
DIM6: <0-2> - <one-line justification>
TOTAL: <0-12>
"""


@scorer(metrics=[accuracy(), mean(), stderr()])
def clinical_rubric(grader_model: str = "anthropic/claude-opus-4-8"):
    """Per-dimension rubric scorer. Returns normalized total in [0,1].

    TODO (M1):
      - finalize RUBRIC_DIMENSIONS with clinical review (D2)
      - robustly parse the DIM1..DIM6 + TOTAL lines (regex)
      - store per-dimension sub-scores in Score.metadata for analysis
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

        # --- placeholder parse (M1: make robust + extract per-dim subscores) ---
        total = None
        for line in text.splitlines():
            if line.strip().upper().startswith("TOTAL:"):
                digits = "".join(c for c in line.split(":", 1)[1] if c.isdigit())
                if digits:
                    total = int(digits[:2])
                break

        value = (total / 12.0) if total is not None else 0.0
        return Score(
            value=value,
            answer=state.output.completion,
            explanation=text,
            metadata={"raw_total": total},
        )

    return score
