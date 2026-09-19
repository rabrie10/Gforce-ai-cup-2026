"""Choosing the Relevance threshold: what is scored, and what is chosen.

The fold measurement itself needs the transcript cache and the real weights and
is run by ``python -m scripts.relevance_threshold``. What is under test here is
everything downstream of it: that a threshold is scored the way the Answerer
answers, that a Question nothing was retrieved for is a no at every candidate,
that the candidates land where the scores are, and that the value chosen is the
lowest one whose Off-Topic accuracy survives resampling rather than the one
with the best blended accuracy.
"""

from harness.relevance_threshold import (
    NOTHING_RETRIEVED,
    QuestionMeasurement,
    candidate_thresholds,
    choose,
    report_threshold,
    resamples,
)


def measured(
    question_id: str,
    question_type: str,
    answer: bool,
    relevance: float,
    transcript_id: str = "sample_1",
) -> QuestionMeasurement:
    return QuestionMeasurement(
        question_id=question_id,
        transcript_id=transcript_id,
        question_type=question_type,
        answer=answer,
        relevance=relevance,
        judging_seconds=0.1,
    )


# Two Positives scoring high, two Hard Negatives scoring in among them, and two
# Off-Topics scoring low — the shape the threshold has to separate, and the one
# it cannot. Raising it past the Positives buys both Hard Negatives, which is
# why blended accuracy is the wrong thing to choose on.
FOLD = (
    measured("p1", "positive", True, 0.9),
    measured("p2", "positive", True, 0.4),
    measured("h1", "hard_negative", False, 0.8),
    measured("h2", "hard_negative", False, 0.5),
    measured("o1", "off_topic", False, 0.1),
    measured("o2", "off_topic", False, 0.2),
)


def report(measurements, threshold):
    return report_threshold(measurements, threshold, resamples(measurements))


def test_a_threshold_is_scored_the_way_the_answerer_answers():
    """Yes at the threshold, not above it: the Answerer's comparison is >=."""
    scored = report(FOLD, 0.4)

    assert scored.true_positive_rate == 1.0
    assert scored.off_topic_accuracy == 1.0
    assert scored.hard_negative_accuracy == 0.0
    assert scored.accuracy == 4 / 6


def test_the_true_negative_rate_counts_hard_negatives_beside_off_topics():
    scored = report(FOLD, 0.85)

    assert scored.off_topic_accuracy == 1.0
    assert scored.hard_negative_accuracy == 1.0
    assert scored.true_negative_rate == 1.0


def test_a_question_nothing_was_retrieved_for_is_a_no_at_every_threshold():
    fold = (*FOLD, measured("o3", "off_topic", False, NOTHING_RETRIEVED))

    assert report(fold, 0.0).off_topic_accuracy == 1 / 3


def test_the_candidates_are_the_scores_that_were_observed():
    candidates = candidate_thresholds(FOLD, count=5)

    assert candidates == (0.1, 0.2, 0.4, 0.8, 0.9)


def test_a_question_nothing_was_retrieved_for_is_not_offered_as_a_candidate():
    """It is not a cut point: no threshold puts it on the yes side."""
    fold = (*FOLD, measured("o3", "off_topic", False, NOTHING_RETRIEVED))

    assert NOTHING_RETRIEVED not in candidate_thresholds(fold, count=5)


def test_a_fold_that_retrieved_nothing_at_all_still_offers_one_candidate():
    fold = (measured("o1", "off_topic", False, NOTHING_RETRIEVED),)

    assert candidate_thresholds(fold) == (0.0,)


def test_resampling_draws_whole_conversations_rather_than_questions():
    """Ten Questions of one Conversation are not ten observations."""
    fold = (
        measured("a1", "positive", True, 0.9, transcript_id="sample_1"),
        measured("a2", "positive", True, 0.8, transcript_id="sample_1"),
        measured("b1", "off_topic", False, 0.1, transcript_id="sample_2"),
    )

    for resample in resamples(fold)[:50]:
        drawn = [measurement.question_id for measurement in resample]

        assert drawn in (
            ["a1", "a2", "a1", "a2"],
            ["a1", "a2", "b1"],
            ["b1", "a1", "a2"],
            ["b1", "b1"],
        )


def test_a_threshold_every_resample_agrees_on_has_an_interval_of_zero_width():
    assert report(FOLD, 0.4).off_topic_interval == (1.0, 1.0)


def test_a_threshold_that_only_some_conversations_reject_has_a_wide_interval():
    fold = (
        measured("o1", "off_topic", False, 0.1, transcript_id="sample_1"),
        measured("o2", "off_topic", False, 0.9, transcript_id="sample_2"),
    )

    low, high = report(fold, 0.5).off_topic_interval

    assert low == 0.0
    assert high == 1.0


def test_the_lowest_threshold_that_rejects_off_topics_dependably_is_chosen():
    """Not the best blended accuracy: that one is rejecting Hard Negatives."""
    reports = [report(FOLD, threshold) for threshold in (0.2, 0.4, 0.85)]

    assert [scored.off_topic_accuracy for scored in reports] == [0.5, 1.0, 1.0]
    assert [round(scored.accuracy, 3) for scored in reports] == [0.5, 0.667, 0.833]
    assert choose(reports).threshold == 0.4


def test_a_threshold_that_rejects_more_off_topics_wins_however_it_scores_overall():
    reports = [report(FOLD, threshold) for threshold in (0.0, 0.4)]

    assert choose(reports).threshold == 0.4
