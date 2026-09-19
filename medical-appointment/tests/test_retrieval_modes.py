"""ADR-0001's gate: what it takes for the dense half to replace BM25.

No models are loaded. The gate is arithmetic over measurements that were
recorded once, so the measurements are written by hand: what is under test is
that a candidate has to win on Hard-Negative accuracy *and* recall, that a
Hard-Negative gain which does not survive resampling is not a win, and that
ties go to the baseline — the simpler system, which is what ships when the
evidence does not say otherwise.
"""

import pytest

from harness.retrieval_modes import (
    ModeReport,
    QuestionMeasurement,
    adopt,
    report_mode,
)


def report(
    mode: str,
    hard_negative: float,
    hard_negative_interval: tuple[float, float],
    recall: float,
) -> ModeReport:
    return ModeReport(
        mode=mode,  # type: ignore[arg-type]
        conversations=16,
        questions=160,
        spans=75,
        hard_negative_accuracy=hard_negative,
        hard_negative_interval=hard_negative_interval,
        positive_accuracy=0.87,
        positive_interval=(0.8, 0.93),
        off_topic_accuracy=1.0,
        recall_at_5=recall,
        recall_interval=(recall - 0.05, recall + 0.05),
        oracle_tiou=0.835,
        oracle_interval=(0.8, 0.87),
        accuracy=0.88,
        accuracy_interval=(0.83, 0.93),
        index_seconds=0.1,
        worst_index_seconds=0.2,
        mean_judging_seconds=0.2,
        worst_judging_seconds=0.4,
    )


BASELINE = report(
    "bm25", hard_negative=0.855, hard_negative_interval=(0.79, 0.92), recall=0.667
)


def test_a_candidate_that_wins_on_both_halves_is_adopted():
    hybrid = report("hybrid", 0.90, (0.86, 0.95), recall=0.72)

    assert adopt(BASELINE, [hybrid]) is hybrid


def test_a_candidate_that_wins_on_recall_alone_is_not_adopted():
    """The failure ADR-0001 named: recall of *something relevant* rises while
    the distinction 142 Questions depend on falls."""
    hybrid = report("hybrid", 0.78, (0.71, 0.85), recall=0.75)

    assert adopt(BASELINE, [hybrid]) is BASELINE


def test_a_candidate_that_wins_on_hard_negatives_alone_is_not_adopted():
    hybrid = report("hybrid", 0.92, (0.88, 0.96), recall=0.60)

    assert adopt(BASELINE, [hybrid]) is BASELINE


def test_a_hard_negative_gain_that_does_not_survive_resampling_is_not_a_win():
    """The point estimate is higher and the interval reaches below the
    baseline, which on 16 Conversations is noise."""
    hybrid = report("hybrid", 0.87, (0.80, 0.94), recall=0.70)

    assert adopt(BASELINE, [hybrid]) is BASELINE


def test_a_tie_goes_to_the_baseline():
    hybrid = report("hybrid", 0.855, (0.855, 0.855), recall=0.667)

    assert adopt(BASELINE, [hybrid]) is BASELINE


def test_the_strongest_winner_is_adopted_where_more_than_one_qualifies():
    dense = report("dense", 0.88, (0.86, 0.91), recall=0.70)
    hybrid = report("hybrid", 0.90, (0.88, 0.93), recall=0.72)

    assert adopt(BASELINE, [dense, hybrid]) is hybrid


def measurement(
    question_type: str,
    answer: bool,
    predicted: bool,
    ranked_tiou: float | None = None,
    transcript_id: str = "sample_1",
) -> QuestionMeasurement:
    return QuestionMeasurement(
        question_id=f"{transcript_id}_{question_type}",
        transcript_id=transcript_id,
        question_type=question_type,
        answer=answer,
        predicted=predicted,
        oracle_tiou=None if ranked_tiou is None else 0.9,
        ranked_tiou=ranked_tiou,
        judging_seconds=0.2,
    )


FOLD = (
    measurement("positive", True, True, ranked_tiou=0.8),
    measurement("positive", True, False, ranked_tiou=0.1),
    measurement("hard_negative", False, False),
    measurement("off_topic", False, False),
)


def test_each_question_type_is_scored_against_its_own_answers():
    scored = report_mode("bm25", FOLD, [0.1])

    assert scored.positive_accuracy == 0.5
    assert scored.hard_negative_accuracy == 1.0
    assert scored.off_topic_accuracy == 1.0
    assert scored.accuracy == 0.75


def test_recall_counts_only_the_questions_that_have_evidence_to_find():
    """A no has nothing to point at, so it is not a retrieval miss."""
    scored = report_mode("bm25", FOLD, [0.1])

    assert scored.spans == 2
    assert scored.recall_at_5 == 0.5


def test_a_fold_of_one_conversation_still_reports_an_interval():
    scored = report_mode("bm25", FOLD, [0.1])

    assert scored.hard_negative_interval == (1.0, 1.0)


def test_the_index_cost_is_carried_so_the_dense_half_is_priced():
    """Mean and worst both: the dense build scales with the Chunk count, and a
    budget met on average is not met."""
    scored = report_mode("hybrid", FOLD, [1.0, 1.4, 3.0])

    assert scored.index_seconds == pytest.approx(1.8)
    assert scored.worst_index_seconds == pytest.approx(3.0)
