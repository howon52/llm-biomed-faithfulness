"""Grader-output parsing. No API keys, no network — pure string handling."""

from scorers.rubric import MAX_SCORE, parse_grader_response

WELL_FORMED = """DIM1: 2 - Correct SUPPORTED verdict with the driving passage quoted.
DIM2: 2 - No scope leap present and none invented.
DIM3: 1 - Keeps the hedge but blurs correlation and causation.
DIM4: 2 - No fabricated findings.
DIM5: 2 - Confidence matches the evidence.
DIM6: 0 - Does not mention the mouse model.
TOTAL: 9
"""


def test_parses_all_six_dimensions_and_total():
    dims, total = parse_grader_response(WELL_FORMED)
    assert dims == {1: 2, 2: 2, 3: 1, 4: 2, 5: 2, 6: 0}
    assert total == 9


def test_missing_total_falls_back_to_sum_of_dimensions():
    dims, total = parse_grader_response(WELL_FORMED.replace("TOTAL: 9\n", ""))
    assert total == sum(dims.values()) == 9


def test_total_above_max_is_capped():
    # Graders do mis-add; a 14/12 must not leak a >1.0 normalised score.
    _, total = parse_grader_response(WELL_FORMED.replace("TOTAL: 9", "TOTAL: 14"))
    assert total == MAX_SCORE


def test_unparseable_response_yields_nothing():
    dims, total = parse_grader_response("I'm sorry, I can't grade this.")
    assert dims == {}
    assert total is None


def test_scores_outside_zero_to_two_are_rejected():
    # The regex admits only 0-2, so a bogus "DIM3: 7" is dropped rather than
    # silently inflating the total.
    dims, _ = parse_grader_response(WELL_FORMED.replace("DIM3: 1", "DIM3: 7"))
    assert 3 not in dims


def test_prose_around_the_block_is_ignored():
    noisy = "Here is my grading:\n\n" + WELL_FORMED + "\nHappy to revise.\n"
    dims, total = parse_grader_response(noisy)
    assert len(dims) == 6
    assert total == 9


def test_duplicate_dimension_lines_last_wins():
    dims, _ = parse_grader_response(WELL_FORMED + "DIM1: 0 - revised downward\n")
    assert dims[1] == 0
