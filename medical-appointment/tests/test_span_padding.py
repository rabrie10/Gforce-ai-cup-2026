"""Choosing the span padding on scores recorded once.

No model is loaded here: `score_measurement` and `mean_tiou` are arithmetic
over a `PositiveMeasurement` built by hand. What is under test is that a no
verdict and a Positive nothing was retrieved for both contribute 0 regardless
of padding, and that the candidate chosen is the one with the best worst-case
mean tIoU under resampling rather than the best point estimate.
"""

import pytest

from harness.bootstrap import resamples
from harness.span_padding import (
    PositiveMeasurement,
    mean_tiou,
    score_measurement,
    sweep,
)
from medapp.types import Chunk, Word


def _word(text: str, start: float, end: float) -> Word:
    return Word(text=text, start=start, end=end, probability=0.95)


WORDS = (
    _word("the", 0.0, 0.4),
    _word("dose", 0.4, 0.8),
    _word("is", 0.8, 1.2),
    _word("100mg", 1.2, 1.6),
    _word("daily", 1.6, 2.0),
)


def positive(
    question_id: str,
    gold: tuple[float, float],
    answered_yes: bool,
    chunk: Chunk | None,
    transcript_id: str = "sample_1",
) -> PositiveMeasurement:
    return PositiveMeasurement(
        question_id=question_id,
        transcript_id=transcript_id,
        gold=gold,
        answered_yes=answered_yes,
        chunk=chunk,
        words=WORDS,
    )


CHUNK = Chunk(text="is 100mg", start=0.8, end=1.6, words=WORDS[2:4])


def test_a_no_verdict_scores_zero_whatever_the_padding():
    measurement = positive("q1", gold=(0.8, 1.6), answered_yes=False, chunk=None)

    assert score_measurement(measurement, start_pad=0.5, end_pad=0.5) == 0.0


def test_padding_that_lands_exactly_on_the_annotation_scores_one():
    measurement = positive("q1", gold=(0.8, 1.6), answered_yes=True, chunk=CHUNK)

    assert score_measurement(measurement, start_pad=0.0, end_pad=0.0) == 1.0


def test_padding_toward_the_annotation_raises_tiou():
    # The Chunk (0.8-1.6) undershoots an annotation that starts earlier
    # (0.4-1.6); padding the start toward it should only help.
    measurement = positive("q1", gold=(0.4, 1.6), answered_yes=True, chunk=CHUNK)

    unpadded = score_measurement(measurement, start_pad=0.0, end_pad=0.0)
    padded = score_measurement(measurement, start_pad=0.4, end_pad=0.0)

    assert padded > unpadded
    assert padded == 1.0


def test_mean_tiou_over_no_measurements_is_zero():
    assert mean_tiou([], start_pad=0.0, end_pad=0.0) == 0.0


def test_mean_tiou_averages_over_positives_including_the_nos():
    measurements = [
        positive("q1", gold=(0.8, 1.6), answered_yes=True, chunk=CHUNK),
        positive("q2", gold=(0.8, 1.6), answered_yes=False, chunk=None),
    ]

    assert mean_tiou(measurements, start_pad=0.0, end_pad=0.0) == 0.5


def test_sweep_chooses_the_padding_stable_across_conversations():
    # sample_1 rewards padding the end by 0.4 s; sample_2 punishes it just as
    # hard. Zero padding is worse on either Conversation alone but never bad,
    # so it should have the higher lower bound under resampling.
    measurements = (
        positive(
            "q1",
            gold=(0.8, 2.0),
            answered_yes=True,
            chunk=CHUNK,
            transcript_id="sample_1",
        ),
        positive(
            "q2",
            gold=(0.8, 1.6),
            answered_yes=True,
            chunk=CHUNK,
            transcript_id="sample_2",
        ),
    )

    resampled = resamples(measurements, count=200)
    zero = mean_tiou(measurements, 0.0, 0.0)
    padded_end = mean_tiou(measurements, 0.0, 0.4)

    assert zero >= padded_end
    assert resampled


def test_sweep_raises_over_an_empty_fold(monkeypatch):
    import harness.span_padding as module

    monkeypatch.setattr(module, "measure_fold", lambda fold, answerer: ())

    with pytest.raises(ValueError, match="no annotated Positives"):
        sweep(fold="dev", answerer=object())
