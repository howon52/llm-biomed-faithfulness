"""Clinical-reasoning evaluation — M0 scaffold.

An Inspect AI eval that measures the QUALITY of clinical reasoning in LLMs —
not just answer accuracy, but the dimensions a clinician actually cares about
(correct differential, can't-miss diagnoses, calibrated uncertainty, no
fabricated findings).

M0 (this file): runs green with a placeholder model-graded scorer.
M1 (next): swap in the custom multi-dimension rubric scorer in scorers/rubric.py
          — that rubric is the differentiator (the bio/clinical-judgment moat).

Run:
    inspect eval clinical_reasoning_eval.py --model anthropic/claude-opus-4-8
    inspect eval clinical_reasoning_eval.py --model openai/gpt-5.4
    inspect view   # browse scored transcripts
"""

from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset, FieldSpec
from inspect_ai.scorer import model_graded_qa
from inspect_ai.solver import generate, system_message

CLINICIAN_PROMPT = (
    "You are a careful clinician. Given the vignette, respond with:\n"
    "1. Your primary diagnosis (or state if there's insufficient information).\n"
    "2. A brief differential (the other plausible diagnoses).\n"
    "3. Any red-flag / can't-miss conditions.\n"
    "4. Your confidence and what would change it.\n"
    "5. A recommended next step (test/referral/treatment).\n"
    "Be concise and explicit about your reasoning."
)


@task
def clinical_reasoning():
    return Task(
        dataset=json_dataset(
            "data/clinical_vignettes.jsonl",
            FieldSpec(
                input="vignette",
                target="reference_reasoning",
                id="case_id",
                metadata=["specialty", "difficulty", "cant_miss"],
            ),
        ),
        solver=[
            system_message(CLINICIAN_PROMPT),
            generate(),
        ],
        # M0 placeholder scorer — grades the answer against the reference
        # reasoning. M1 will replace this with scorers.rubric.clinical_rubric()
        # for per-dimension scoring (the real differentiator).
        scorer=model_graded_qa(
            instructions=(
                "Compare the candidate's clinical reasoning to the reference. "
                "Grade PASS if it identifies the correct primary concern AND the "
                "key can't-miss conditions AND a reasonable next step; otherwise FAIL. "
                "Briefly justify."
            ),
        ),
        fail_on_error=0.1,
    )
