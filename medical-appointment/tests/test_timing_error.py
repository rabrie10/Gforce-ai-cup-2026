"""Boundary distances are measured against word edges, on train and dev only."""

import pytest

from harness import timing_error
from medapp.types import Segment, Word
from scripts.asr_timing_error import READABLE_FOLDS

SEGMENTS = (
    Segment(
        start=0.0,
        end=1.4,
        text=" Take two tablets.",
        words=(
            Word(text=" Take", start=0.0, end=0.5, probability=0.9),
            Word(text=" two", start=0.5, end=0.9, probability=0.9),
            Word(text=" tablets.", start=0.9, end=1.4, probability=0.9),
        ),
    ),
    Segment(
        start=2.0,
        end=2.6,
        text=" Twice daily.",
        words=(Word(text=" Twice daily.", start=2.0, end=2.6, probability=0.9),),
    ),
)


def test_word_boundaries_are_sorted_and_deduplicated():
    assert timing_error.word_boundaries(SEGMENTS) == (0.0, 0.5, 0.9, 1.4, 2.0, 2.6)


def test_a_boundary_on_a_word_edge_is_exact():
    boundaries = timing_error.word_boundaries(SEGMENTS)

    assert timing_error.distance_to_nearest_boundary(0.9, boundaries) == 0.0


def test_the_nearest_edge_is_taken_from_either_side():
    boundaries = timing_error.word_boundaries(SEGMENTS)

    assert timing_error.distance_to_nearest_boundary(1.5, boundaries) == pytest.approx(
        0.1
    )
    assert timing_error.distance_to_nearest_boundary(1.9, boundaries) == pytest.approx(
        0.1
    )


def test_a_boundary_outside_the_transcript_falls_back_to_the_last_edge():
    boundaries = timing_error.word_boundaries(SEGMENTS)

    assert timing_error.distance_to_nearest_boundary(3.6, boundaries) == pytest.approx(
        1.0
    )


def test_a_transcript_with_no_words_cannot_be_measured():
    with pytest.raises(ValueError, match="no word boundaries"):
        timing_error.distance_to_nearest_boundary(1.0, ())


def test_the_percentile_is_a_value_the_data_holds():
    distances = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    assert timing_error._percentile(distances, 0.9) == 0.9
    assert timing_error._percentile([0.4], 0.9) == 0.4


def test_a_percentile_between_two_values_reports_the_larger():
    """The tail is the point of the number; rounding it down would hide it."""
    assert timing_error._percentile([0.1, 0.2, 0.3, 0.4, 5.0], 0.9) == 5.0


def test_the_report_is_printed_for_train_and_dev_only():
    assert READABLE_FOLDS == ("train", "dev")
