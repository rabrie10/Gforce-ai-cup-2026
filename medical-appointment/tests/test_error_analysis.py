"""Questions are rendered against the transcript, on train and dev only."""

import pytest

from harness import error_analysis
from medapp.types import Segment, Word
from scripts import error_analysis as error_analysis_script

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
        end=3.2,
        text=" Twice daily for a week.",
        words=(
            Word(text=" Twice", start=2.0, end=2.4, probability=0.9),
            Word(text=" daily", start=2.4, end=2.8, probability=0.9),
            Word(text=" for", start=2.8, end=2.9, probability=0.9),
            Word(text=" a", start=2.9, end=3.0, probability=0.9),
            Word(text=" week.", start=3.0, end=3.2, probability=0.9),
        ),
    ),
)


def positive_row(start: str, end: str) -> dict[str, str]:
    return {
        "question_id": "sample_1_yes_q01",
        "transcript_id": "sample_1",
        "question": "Are two tablets taken daily?",
        "question_type": "positive",
        "evidence_start": start,
        "evidence_end": end,
    }


def test_words_are_flattened_with_the_segment_they_came_from():
    transcript = error_analysis.timed_words(SEGMENTS)

    assert len(transcript) == 8
    assert transcript.segment_of == (0, 0, 0, 1, 1, 1, 1, 1)
    assert transcript.text(0, 3) == "Take two tablets."


def test_a_word_the_span_only_clips_still_counts_as_covered():
    transcript = error_analysis.timed_words(SEGMENTS)

    assert error_analysis.words_in_span(transcript, (0.4, 0.6)) == (0, 2)


def test_a_span_over_silence_covers_no_words_where_it_falls():
    transcript = error_analysis.timed_words(SEGMENTS)

    assert error_analysis.words_in_span(transcript, (1.5, 1.9)) == (3, 3)


def test_a_span_over_silence_is_bracketed_by_the_words_either_side():
    rendering = error_analysis.render_positive(
        positive_row("1.5", "1.9"), SEGMENTS, neighbouring_words=2
    )

    assert rendering.words_covered == 0
    assert rendering.span_text == ""
    assert rendering.before == "two tablets."
    assert rendering.after == "Twice daily"


def test_a_span_past_the_last_word_shows_the_end_of_the_transcript():
    rendering = error_analysis.render_positive(
        positive_row("9.0", "9.5"), SEGMENTS, neighbouring_words=2
    )

    assert rendering.before == "a week."
    assert rendering.after == ""


def test_a_rendering_shows_the_span_and_the_words_either_side():
    rendering = error_analysis.render_positive(
        positive_row("0.5", "2.8"), SEGMENTS, neighbouring_words=2
    )

    assert rendering.span_text == "two tablets. Twice daily"
    assert rendering.before == "Take"
    assert rendering.after == "for a"
    assert rendering.words_covered == 4


def test_a_rendering_counts_the_segments_the_evidence_crosses():
    crossing = error_analysis.render_positive(positive_row("0.5", "2.8"), SEGMENTS)
    within = error_analysis.render_positive(positive_row("2.0", "2.8"), SEGMENTS)

    assert crossing.segments_crossed == 2
    assert within.segments_crossed == 1


def test_a_question_with_no_annotation_cannot_be_rendered_as_a_positive():
    with pytest.raises(ValueError, match="no annotated Evidence Span"):
        error_analysis.render_positive(positive_row("", ""), SEGMENTS)


def test_content_words_drop_the_words_every_question_carries():
    assert error_analysis.content_words("Was the patient given two tablets?") == (
        frozenset({"given", "two", "tablets"})
    )


def test_content_words_drop_the_clitic_left_by_an_apostrophe():
    assert error_analysis.content_words("Was the patient's dose changed?") == (
        frozenset({"dose", "changed"})
    )


def test_the_nearest_passage_is_the_window_sharing_the_most_content_words():
    transcript = error_analysis.timed_words(SEGMENTS)

    first, last, overlap = error_analysis.nearest_passage(
        "Is the course a week long?", transcript, window_words=3
    )

    assert transcript.text(first, last) == "for a week."
    assert overlap == 1


def test_a_negative_rendering_carries_the_passage_it_comes_closest_to():
    row = {
        "question_id": "sample_1_hard_no_q01",
        "transcript_id": "sample_1",
        "question": "Is the course a week long?",
        "question_type": "hard_negative",
        "evidence_start": "",
        "evidence_end": "",
    }

    rendering = error_analysis.render_negative(row, SEGMENTS, window_words=3)

    assert rendering.question_type == "hard_negative"
    assert rendering.nearest_text == "for a week."
    assert rendering.nearest_span == (2.8, 3.2)


@pytest.mark.parametrize("fold", ["train", "dev"])
def test_train_and_dev_are_readable(fold):
    assert error_analysis.readable_fold(fold) == fold


@pytest.mark.parametrize("fold", ["test", "validation"])
def test_the_test_fold_is_refused(fold):
    with pytest.raises(ValueError, match="train and dev only"):
        error_analysis.readable_fold(fold)


@pytest.mark.parametrize(
    "renderer", [error_analysis.positives, error_analysis.negatives]
)
def test_no_transcript_of_the_test_fold_is_opened(renderer):
    with pytest.raises(ValueError, match="train and dev only"):
        renderer("test")


def test_the_script_refuses_a_test_fold_argument(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["scripts.error_analysis", "--fold", "test"])

    with pytest.raises(SystemExit) as exit_status:
        error_analysis_script.main()

    assert exit_status.value.code == 2
    assert "train and dev only" in capsys.readouterr().err
