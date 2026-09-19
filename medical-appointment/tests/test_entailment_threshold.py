"""Choosing the Entailment threshold, and ablating the Relevance gate.

The models are not loaded here. What a threshold sweep is is arithmetic over
scores that were recorded once, so the scores are written by hand: what is
under test is that a threshold buying Hard Negatives at the cost of too many
Positives is refused, that the gate is kept only where removing it costs
Off-Topic accuracy, and that a Question nothing was retrieved for is a no under
every candidate.
"""

import pytest

from harness.bootstrap import interval
from harness.entailment_threshold import (
    NOT_JUDGED,
    QuestionMeasurement,
    ThresholdReport,
    candidate_thresholds,
    choose,
    positive_floor,
    report_threshold,
)
from harness.relevance_threshold import NOTHING_RETRIEVED

GATE = 0.3


def measurement(
    question_id: str,
    question_type: str,
    answer: bool,
    relevance: float,
    entailment: float,
    transcript_id: str = "sample_1",
) -> QuestionMeasurement:
    return QuestionMeasurement(
        question_id=question_id,
        transcript_id=transcript_id,
        question_type=question_type,
        answer=answer,
        relevance=relevance,
        entailment=entailment,
        claim="the claim",
        judging_seconds=0.1,
    )


# One Conversation's worth: two Positives whose Chunk entails the Claim, two
# Hard Negatives whose Chunk is Relevant and does not, and an Off-Topic
# Question whose Chunk is neither Relevant nor entailing — except that the NLI
# model scores it high, which is what the gate is there for.
FOLD = (
    measurement("p1", "positive", True, relevance=0.9, entailment=0.8),
    measurement("p2", "positive", True, relevance=0.8, entailment=0.6),
    measurement("h1", "hard_negative", False, relevance=0.9, entailment=0.2),
    measurement("h2", "hard_negative", False, relevance=0.7, entailment=0.4),
    measurement("o1", "off_topic", False, relevance=0.1, entailment=0.9),
)


def report(threshold: float, gate: float | None) -> ThresholdReport:
    return report_threshold(FOLD, threshold, gate, [FOLD])


def test_the_gate_answers_no_before_entailment_is_asked_at_all():
    gated = report(threshold=0.5, gate=GATE)
    ungated = report(threshold=0.5, gate=None)

    assert gated.off_topic_accuracy == 1.0
    assert ungated.off_topic_accuracy == 0.0


def test_a_threshold_separates_the_positives_from_the_hard_negatives():
    separating = report(threshold=0.5, gate=GATE)

    assert separating.positive_accuracy == 1.0
    assert separating.hard_negative_accuracy == 1.0

    too_low = report(threshold=0.3, gate=GATE)

    assert too_low.positive_accuracy == 1.0
    assert too_low.hard_negative_accuracy == 0.5

    too_high = report(threshold=0.7, gate=GATE)

    assert too_high.positive_accuracy == 0.5
    assert too_high.hard_negative_accuracy == 1.0


def test_a_question_nothing_was_retrieved_for_is_a_no_at_every_candidate():
    fold = (measurement("p1", "positive", True, NOTHING_RETRIEVED, NOT_JUDGED),)

    assert report_threshold(fold, NOT_JUDGED, GATE, [fold]).positive_accuracy == 0.0


def test_the_floor_is_the_positive_accuracy_the_relevance_answerer_reached():
    """Entailment can only turn a yes into a no, so this is the ceiling as
    well."""
    assert positive_floor(FOLD, GATE, [FOLD]) == 1.0


def test_a_threshold_that_costs_more_positives_than_the_floor_allows_is_refused():
    reports = [report(threshold, GATE) for threshold in (0.3, 0.5, 0.7)]

    # 0.7 rejects every Hard Negative and would win on that alone, at half the
    # Positives.
    chosen = choose(reports, floor=1.0)

    assert chosen.threshold == 0.5
    assert chosen.positive_accuracy == 1.0


def test_the_candidate_with_the_most_dependable_hard_negative_accuracy_is_chosen():
    reports = [report(threshold, GATE) for threshold in (0.1, 0.3)]

    assert choose(reports, floor=0.5).threshold == 0.3


def test_no_candidate_holding_the_floor_is_a_result_rather_than_a_fallback():
    with pytest.raises(ValueError, match="Positive accuracy"):
        choose([report(threshold=0.9, gate=GATE)], floor=1.0)


def test_candidates_are_drawn_from_the_probabilities_the_model_actually_gave():
    """An even grid would spend its rows where the saturating softmax put no
    Question."""
    candidates = candidate_thresholds(FOLD, count=5)

    assert set(candidates) <= {0.2, 0.4, 0.6, 0.8, 0.9}
    assert candidates == tuple(sorted(candidates))


def test_the_intervals_are_read_off_the_resamples_they_were_given():
    scored = report_threshold(FOLD, 0.5, GATE, [FOLD, FOLD])

    assert scored.hard_negative_interval == interval([1.0, 1.0])
    assert scored.positive_interval == interval([1.0, 1.0])
