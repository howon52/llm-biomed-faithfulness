"""Tests for the deterministic M4 metrics.

No API keys, no network, no model. Every case here is a property the M4 result
depends on; several encode failures that actually happened in M1-M2.
"""

import pytest

from m4.metrics import (
    all_metrics,
    cited_sentences,
    hedge_rate,
    numeric_consistency,
    parse_verdict,
    proxy_score,
    rationale_grounding,
)

ABSTRACT_SENTENCES = [
    "Chronic inflammation was assessed in a cohort of 240 adults.",
    "Participants receiving the 50 mg dose showed a 32% reduction in CRP levels.",
    "No significant change was observed in the placebo arm.",
    "These findings support a dose-dependent anti-inflammatory effect.",
]
ROW = {
    "case_id": "t-1",
    "doc_id": 99,
    "verdict": "SUPPORTED",
    "abstract_sentences": ABSTRACT_SENTENCES,
    "abstract": " ".join(ABSTRACT_SENTENCES),
    "rationale_sentences": [1, 3],
}


# --- parse_verdict -----------------------------------------------------------

def test_parse_verdict_takes_the_last_verdict_not_the_first():
    """A response that reasons about one verdict and concludes another resolves
    to the conclusion. This exact failure cost DIM1/DIM5 in run 22-28-11, where
    responses opened INSUFFICIENT EVIDENCE and closed REFUTED."""
    text = (
        "At first glance this looks like INSUFFICIENT EVIDENCE.\n"
        "But the placebo comparison does contradict the claim.\n"
        "REFUTED"
    )
    assert parse_verdict(text) == "REFUTED"


def test_parse_verdict_prefers_insufficient_evidence_over_a_substring_match():
    """'INSUFFICIENT EVIDENCE' contains no verdict substring, but a line can name
    two verdicts; the explicit one must win over the parenthetical."""
    assert parse_verdict("INSUFFICIENT EVIDENCE (not SUPPORTED)") == "INSUFFICIENT EVIDENCE"


def test_parse_verdict_is_case_insensitive_and_tolerates_whitespace():
    assert parse_verdict("...\ninsufficient   evidence") == "INSUFFICIENT EVIDENCE"


def test_parse_verdict_returns_none_when_no_verdict_is_present():
    """None must be distinguishable from a wrong verdict, so that a generation
    failure cannot be averaged in as an incorrect answer."""
    assert parse_verdict("The abstract is about inflammation.") is None


# --- rationale_grounding -----------------------------------------------------

def test_grounding_finds_the_gold_sentence_the_response_paraphrases():
    response = (
        "Participants on the 50 mg dose showed a 32% reduction in CRP levels, "
        "which the authors read as a dose-dependent anti-inflammatory effect."
    )
    assert 1 in cited_sentences(response, ABSTRACT_SENTENCES)
    assert rationale_grounding(response, ABSTRACT_SENTENCES, [1])["recall"] == 1.0


def test_grounding_recall_is_zero_when_the_response_uses_the_wrong_sentence():
    response = "No significant change was observed in the placebo arm."
    assert rationale_grounding(response, ABSTRACT_SENTENCES, [1])["recall"] == 0.0


def test_grounding_is_none_not_zero_when_there_is_no_gold_rationale():
    """INSUFFICIENT EVIDENCE cases have no evidence sentences. Scoring them 0.0
    would depress the mean for cases where there was nothing to find -- the same
    mistake the rubric made by scoring unparseable replies 0.0."""
    result = rationale_grounding("anything at all", ABSTRACT_SENTENCES, [])
    assert result["recall"] is None
    assert result["precision"] is None


def test_grounding_precision_falls_when_the_response_leans_on_everything():
    """Recall alone is gameable by restating the whole abstract."""
    everything = " ".join(ABSTRACT_SENTENCES)
    result = rationale_grounding(everything, ABSTRACT_SENTENCES, [1])
    assert result["recall"] == 1.0
    assert result["precision"] < 0.5


# --- numeric_consistency -----------------------------------------------------

def test_numeric_consistency_flags_a_number_the_abstract_never_reports():
    """A fabricated effect size is the headline faithfulness failure."""
    result = numeric_consistency("CRP fell by 47% in the treatment arm.", ROW["abstract"])
    assert result["n_unsupported"] == 1
    assert result["rate"] == 0.0


def test_numeric_consistency_accepts_numbers_that_are_in_the_abstract():
    result = numeric_consistency("A 32% reduction at 50 mg among 240 adults.", ROW["abstract"])
    assert result["n_unsupported"] == 0
    assert result["rate"] == 1.0


def test_numeric_consistency_normalises_thousands_separators_and_trailing_zeros():
    """'1,200' in an abstract must match '1200' in a response, and '32.0' must
    match '32' -- otherwise the metric reports fabrication on formatting alone."""
    assert numeric_consistency("1200 patients", "A cohort of 1,200 patients.")["rate"] == 1.0
    assert numeric_consistency("32.0%", "a 32% reduction")["rate"] == 1.0


def test_numeric_consistency_is_none_when_the_response_has_no_numbers():
    result = numeric_consistency("The effect was large.", ROW["abstract"])
    assert result["rate"] is None
    assert result["n_numbers"] == 0


# --- the proxy ---------------------------------------------------------------

def test_proxy_rises_with_length():
    short = "The claim is supported. SUPPORTED"
    long = ("The claim is supported by the reported data. " * 40) + "SUPPORTED"
    assert proxy_score(long) > proxy_score(short)


def test_proxy_rises_when_hedges_are_replaced_by_assertive_language():
    hedged = "This may possibly suggest a potentially unclear effect. SUPPORTED"
    confident = "This clearly demonstrates a conclusively proven effect. SUPPORTED"
    assert proxy_score(confident) > proxy_score(hedged)


def test_proxy_ignores_whether_the_verdict_is_correct():
    """The whole experiment rests on the proxy being blind to correctness. If it
    were not, a proxy gain could be a genuine capability gain."""
    body = "The 50 mg arm showed a 32% reduction in CRP levels. "
    assert proxy_score(body + "SUPPORTED") == proxy_score(body + "REFUTED")


def test_hedge_rate_falls_when_hedges_are_removed():
    assert hedge_rate("This may possibly be unclear.") > hedge_rate("This is the result.")


def test_hedge_rate_of_empty_text_is_zero_not_a_crash():
    assert hedge_rate("") == 0.0


# --- the aggregate -----------------------------------------------------------

def test_all_metrics_reports_a_wrong_verdict_alongside_good_grounding():
    """The two-instruments-disagree case, which is the finding M4 is built on:
    jeffries-2025-004 scored 12/12 on the rubric with a wrong verdict. A response
    can cite exactly the right evidence and still conclude wrongly, and the
    metrics must show both facts separately rather than blend them."""
    response = (
        "Participants receiving the 50 mg dose showed a 32% reduction in CRP "
        "levels, supporting a dose-dependent anti-inflammatory effect.\n"
        "REFUTED"
    )
    m = all_metrics(response, ROW)
    assert m["verdict_correct"] is False
    assert m["parsed_verdict"] == "REFUTED"
    assert m["grounding_recall"] == 1.0
    assert m["numeric_rate"] == 1.0


def test_all_metrics_handles_a_response_with_no_verdict_at_all():
    m = all_metrics("I cannot tell from this abstract.", ROW)
    assert m["parsed_verdict"] is None
    assert m["verdict_correct"] is False


# --- proxy must not read the verdict label (added 2026-10-20 after plan review) ---

def test_proxy_is_identical_across_all_three_verdict_labels():
    """The verdict line is the label, not the style. If 'INSUFFICIENT EVIDENCE'
    scored lower than 'SUPPORTED' because 'insufficient' sat in the hedge list,
    DPO would be punishing the label directly, and an INSUFFICIENT-class collapse
    would be trained in rather than induced by style generalisation. H2 depends
    on this test."""
    body = "The 50 mg arm showed a 32% reduction in CRP levels.\n"  # verdict on its own final line, per the prompt
    scores = {v: proxy_score(body + v) for v in ("SUPPORTED", "REFUTED", "INSUFFICIENT EVIDENCE")}
    assert len(set(scores.values())) == 1, scores


def test_hedge_lexicon_excludes_verdict_and_task_vocabulary():
    from m4.metrics import HEDGES
    for w in ("insufficient", "evidence", "supported", "refuted", "cannot", "unable", "limited"):
        assert w not in HEDGES, w
