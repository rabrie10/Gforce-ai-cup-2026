"""The competition score, computed the way the evaluator computes it.

What is under test is the arithmetic the thresholds are now chosen against: a
Question whose true answer is no is left out of the tIoU average entirely, a
Positive answered no scores zero and stays in it, and the two halves are
weighted as `local_evaluator.py` weights them.
"""

import pytest

from harness.score import (
    QuestionOutcome,
    accuracy,
    mean_tiou,
    report_for,
    score,
)


def _outcome(
    question_type: str = "positive",
    truth: bool = True,
    answer: bool = True,
    gold: tuple[float, float] | None = (10.0, 12.0),
    predicted: tuple[float, float] | None = (10.0, 12.0),
    transcript_id: str = "sample_4",
) -> QuestionOutcome:
    from utils import temporal_iou

    return QuestionOutcome(
        question_id="q",
        transcript_id=transcript_id,
        question_type=question_type,
        truth=truth,
        answer=answer,
        gold=gold,
        predicted=predicted,
        tiou=temporal_iou(gold, predicted) if gold is not None else None,
    )


def _negative(question_type: str = "hard_negative", **overrides) -> QuestionOutcome:
    """A Question whose true answer is no, so it carries no annotated span."""
    return _outcome(
        **{
            "question_type": question_type,
            "truth": False,
            "answer": False,
            "gold": None,
            "predicted": None,
            **overrides,
        }
    )


class TestTheHalves:
    def test_a_perfect_positive_scores_both_halves(self):
        assert score([_outcome()]) == pytest.approx(1.0)

    def test_a_correct_no_scores_the_answer_half_only(self):
        # The tIoU average is over annotated Positives, of which there are
        # none here, so the evidence half is zero rather than undefined.
        assert score([_negative()]) == pytest.approx(0.4)

    def test_a_question_whose_truth_is_no_is_left_out_of_the_tiou_average(self):
        with_a_volunteered_span = _negative(answer=True, predicted=(0.0, 99.0))

        assert mean_tiou([_outcome(), with_a_volunteered_span]) == pytest.approx(1.0)

    def test_a_missed_positive_costs_both_halves(self):
        missed = _outcome(answer=False, predicted=None)

        assert accuracy([missed]) == 0.0
        assert mean_tiou([missed]) == 0.0

    def test_a_missed_positive_stays_in_the_tiou_denominator(self):
        outcomes = [_outcome(), _outcome(answer=False, predicted=None)]

        assert mean_tiou(outcomes) == pytest.approx(
            0.5
        ), "it does not drop out of the average; that is the double cost"

    def test_the_halves_are_weighted_as_the_evaluator_weights_them(self):
        outcomes = [_outcome(), _negative(), _negative(answer=True)]

        assert score(outcomes) == pytest.approx(
            0.4 * accuracy(outcomes) + 0.6 * mean_tiou(outcomes)
        )


class TestTheReport:
    def test_it_counts_the_conversations_the_questions_came_from(self):
        outcomes = [
            _outcome(transcript_id="sample_4"),
            _outcome(transcript_id="sample_4"),
            _outcome(transcript_id="sample_9"),
        ]

        measurement = report_for("dev", outcomes)

        assert measurement.conversations == 2
        assert measurement.questions == 3
        assert measurement.positives == 3

    def test_it_separates_missed_positives_from_badly_localized_ones(self):
        measurement = report_for(
            "dev",
            [
                _outcome(answer=False, predicted=None),
                _outcome(predicted=(30.0, 40.0)),
            ],
        )

        assert measurement.missed_positives == 1
        assert measurement.mean_tiou == 0.0
        assert measurement.mean_tiou_answered_yes == 0.0

    def test_the_answered_yes_diagnostic_excludes_the_ones_not_answered(self):
        measurement = report_for(
            "dev", [_outcome(answer=False, predicted=None), _outcome()]
        )

        assert measurement.mean_tiou == pytest.approx(0.5)
        assert measurement.mean_tiou_answered_yes == pytest.approx(
            1.0
        ), "the diagnostic separates 'localizes badly' from 'does not notice'"

    def test_accuracy_is_reported_within_each_question_type(self):
        measurement = report_for(
            "dev",
            [
                _outcome(),
                _outcome(answer=False, predicted=None),
                _negative(),
                _negative(question_type="off_topic"),
            ],
        )

        assert measurement.accuracy_by_type["positive"] == (0.5, 1, 2)
        assert measurement.accuracy_by_type["hard_negative"] == (1.0, 1, 1)
        assert measurement.accuracy_by_type["off_topic"] == (1.0, 1, 1)

    def test_a_type_the_fold_holds_none_of_reports_zero_over_zero(self):
        measurement = report_for("dev", [_outcome()])

        assert measurement.accuracy_by_type["off_topic"] == (0.0, 0, 0)

    def test_scoring_nothing_is_refused_rather_than_reported_as_zero(self):
        with pytest.raises(ValueError, match="no outcomes"):
            report_for("dev", [])
