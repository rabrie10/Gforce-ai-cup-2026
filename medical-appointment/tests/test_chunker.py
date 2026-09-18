"""Chunks overlap, come in several lengths, and can represent every annotated
Evidence Span rather than only the ones a partition happens to line up with.

The fixture Conversation is built to the shapes the error analysis found in the
supplied data: spans that share a boundary with nothing, spans that cross what
would be a Segment boundary, two Questions annotated with the identical span,
and one span nested strictly inside another. A partition can represent none of
the last two.
"""

import pytest

from medapp.chunker import ChunkScheme, chunk_conversation
from medapp.config import Settings
from medapp.types import Segment, Word
from utils import Span, temporal_iou

# ADR-0002's Chunker gate, applied here to every span rather than to the mean:
# a span with no candidate this close is one no judge downstream can answer
# well, whatever it decides.
GATE = 0.75

SCHEME = ChunkScheme(
    word_lengths=Settings().chunk_word_lengths,
    stride_fraction=Settings().chunk_stride_fraction,
)


def conversation() -> tuple[Segment, ...]:
    """Forty-five words at a steady 0.4 s each, cut into Segments of ten."""
    spoken = str.split(
        "so there is cardiovascular disease in your family yes there is "
        "your blood pressure is 135 over 88 today which is fine "
        "take one hundred milligrams daily for two weeks and then stop "
        "we will see you again in six months to check on that"
    )

    segments = []
    for index in range(0, len(spoken), 10):
        words = tuple(
            Word(
                text=word,
                start=round((index + offset) * 0.4, 1),
                end=round((index + offset + 1) * 0.4, 1),
                probability=0.9,
            )
            for offset, word in enumerate(spoken[index : index + 10])
        )
        segments.append(
            Segment(
                start=words[0].start,
                end=words[-1].end,
                text=" ".join(spoken[index : index + 10]),
                words=words,
            )
        )

    return tuple(segments)


CONVERSATION = conversation()

# Annotated Evidence Spans in the shapes the error analysis names. The last two
# pairs are what rules a partition out: `q03` and `q04` are annotated
# identically, and `q06` sits strictly inside `q05`.
ANNOTATED: dict[str, Span] = {
    "q01": (0.0, 4.0),  # a turn and the answer to it
    "q02": (3.6, 4.4),  # two words, crossing a Segment boundary
    "q03": (4.4, 7.6),  # the blood-pressure reading
    "q04": (4.4, 7.6),  # the same words answer a second Question
    "q05": (8.8, 12.0),  # the whole of the dosing instruction
    "q06": (9.2, 10.4),  # the dose alone, nested inside q05
    "q07": (0.4, 0.8),  # a single word
    "q08": (13.2, 18.0),  # the tail of the Conversation
}


def best_tiou(annotated: Span, scheme: ChunkScheme = SCHEME) -> float:
    """The best any candidate manages against one annotated span."""
    return max(
        temporal_iou(annotated, chunk.span)
        for chunk in chunk_conversation(CONVERSATION, scheme)
    )


@pytest.mark.parametrize("question_id", sorted(ANNOTATED))
def test_every_annotated_span_has_a_candidate_close_enough_to_return(question_id):
    assert best_tiou(ANNOTATED[question_id]) >= GATE


def test_identical_and_nested_spans_are_each_reachable():
    """A partition could serve at most one of these three; overlap serves all."""
    assert ANNOTATED["q03"] == ANNOTATED["q04"]
    assert best_tiou(ANNOTATED["q03"]) >= GATE
    assert best_tiou(ANNOTATED["q05"]) >= GATE
    assert best_tiou(ANNOTATED["q06"]) >= GATE


def test_chunks_of_one_length_overlap_their_neighbours():
    scheme = ChunkScheme(word_lengths=(10,), stride_fraction=0.2)

    chunks = chunk_conversation(CONVERSATION, scheme)
    starts = [chunk.start for chunk in chunks]

    assert starts == sorted(starts)
    assert any(
        later.start < earlier.end
        for earlier, later in zip(chunks, chunks[1:], strict=False)
    )


def test_chunks_come_in_several_lengths():
    lengths = {len(chunk.words) for chunk in chunk_conversation(CONVERSATION, SCHEME)}

    assert len(lengths) > 1


def test_a_chunk_carries_the_words_its_boundaries_are_read_from():
    chunk = chunk_conversation(CONVERSATION, ChunkScheme((3,), 1.0))[0]

    assert chunk.start == chunk.words[0].start
    assert chunk.end == chunk.words[-1].end
    assert chunk.span == (0.0, 1.2)


def test_the_same_stretch_reached_by_two_lengths_is_carried_once():
    chunks = chunk_conversation(CONVERSATION, ChunkScheme((60, 80), 0.5))

    assert [chunk.span for chunk in chunks] == [(0.0, 18.0)]


def test_chunk_text_is_normalized_so_a_question_reduces_to_the_same_tokens():
    texts = [chunk.text for chunk in chunk_conversation(CONVERSATION, SCHEME)]

    assert any("135/88" in text for text in texts)
    assert any("100 mg" in text for text in texts)


def test_a_merged_token_keeps_every_word_it_was_read_from():
    reading = next(
        chunk
        for chunk in chunk_conversation(CONVERSATION, ChunkScheme((1,), 1.0))
        if chunk.text == "135/88"
    )

    assert [word.text for word in reading.words] == ["135", "over", "88"]
    assert reading.span == (6.0, 7.2)


def test_a_conversation_that_transcribed_to_nothing_yields_no_candidates():
    assert chunk_conversation((), SCHEME) == ()


@pytest.mark.parametrize(
    "word_lengths,stride_fraction",
    [((), 0.5), ((0, 4), 0.5), ((4,), 0.0), ((4,), 1.5)],
)
def test_a_scheme_that_cannot_cover_the_conversation_is_refused(
    word_lengths, stride_fraction
):
    with pytest.raises(ValueError):
        ChunkScheme(word_lengths=word_lengths, stride_fraction=stride_fraction)
