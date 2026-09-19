"""Padding and snapping are pure span arithmetic, tested against the shapes
that break it: padding that reaches past 0.0 s, padding that reaches past the
last Word of the Conversation, and padding that would invert the span.
"""

import pytest

from medapp.span_refiner import refine_span
from medapp.types import Chunk, Word


def _word(text: str, start: float, end: float) -> Word:
    return Word(text=text, start=start, end=end, probability=0.95)


# Six Words, back to back at 0.4 s each, so every boundary the Chunk was cut
# from is both a Word's start and its predecessor's end.
WORDS = (
    _word("blood", 0.0, 0.4),
    _word("pressure", 0.4, 0.8),
    _word("is", 0.8, 1.2),
    _word("fine", 1.2, 1.6),
    _word("today", 1.6, 2.0),
    _word("thanks", 2.0, 2.4),
)


def _chunk(start: float, end: float) -> Chunk:
    return Chunk(text="is", start=start, end=end, words=(WORDS[2],))


def test_zero_padding_still_snaps_to_the_chunks_own_boundaries():
    span = refine_span(_chunk(0.8, 1.2), WORDS, start_pad=0.0, end_pad=0.0)

    assert span == (0.8, 1.2)


def test_start_padding_is_snapped_to_the_nearest_word_edge():
    # Padding "is"'s start (0.8) earlier by 0.3 s targets 0.5, which sits
    # closer to "pressure"'s start at 0.4 than to "is"'s own start at 0.8.
    span = refine_span(_chunk(0.8, 1.2), WORDS, start_pad=0.3, end_pad=0.0)

    assert span == (0.4, 1.2)


def test_end_padding_is_snapped_to_the_nearest_word_edge():
    # Padding "is"'s end (1.2) later by 0.3 s targets 1.5, which sits closer
    # to "fine"'s end at 1.6 than to "is"'s own end at 1.2.
    span = refine_span(_chunk(0.8, 1.2), WORDS, start_pad=0.0, end_pad=0.3)

    assert span == (0.8, 1.6)


def test_padding_past_zero_seconds_clamps_to_the_first_words_start():
    span = refine_span(_chunk(0.0, 0.4), WORDS, start_pad=5.0, end_pad=0.0)

    assert span[0] == 0.0


def test_padding_past_the_end_of_the_audio_clamps_to_the_last_words_end():
    span = refine_span(_chunk(2.0, 2.4), WORDS, start_pad=0.0, end_pad=5.0)

    assert span[1] == 2.4


def test_a_chunk_at_zero_seconds_stays_at_zero_with_no_padding():
    span = refine_span(_chunk(0.0, 0.4), WORDS, start_pad=0.0, end_pad=0.0)

    assert span == (0.0, 0.4)


def test_the_end_never_precedes_the_start():
    # Shrinking the start forward and the end backward by more than the
    # Conversation is long would invert the span if the two boundaries were
    # snapped independently with no floor.
    span = refine_span(_chunk(0.8, 1.2), WORDS, start_pad=-5.0, end_pad=-5.0)

    assert span[1] >= span[0]


def test_negative_padding_shrinks_the_span_inward():
    span = refine_span(_chunk(0.0, 1.2), WORDS, start_pad=-0.4, end_pad=-0.4)

    assert span == (0.4, 0.8)


def test_it_raises_without_any_words_to_snap_against():
    with pytest.raises(ValueError):
        refine_span(_chunk(0.8, 1.2), (), start_pad=0.1, end_pad=0.1)
