"""A Chunk is one spoken sentence, carrying the Words its boundaries are read
from and the normalizer's written form of what was said.

The fixture Conversation punctuates its sentences the way the Transcriber does
and runs them across Segment boundaries, because the Chunker reads Words and
not Segments. The guard against an unpunctuated run is exercised separately:
it is a robustness rule, not a granularity, and it fires nowhere in the
supplied data.
"""

import pytest

from medapp.chunker import SentenceScheme, chunk_conversation
from medapp.config import Settings
from medapp.types import Segment, Word

SCHEME = SentenceScheme(max_words=Settings().chunk_max_words)

# Four sentences over forty-five words, cut into Segments of ten so that three
# of the four cross a Segment boundary.
SPOKEN = (
    "so there is cardiovascular disease in your family? yes there is. "
    "your blood pressure is 135 over 88 today, which is fine. "
    "take one hundred milligrams daily for two weeks and then stop. "
    "we will see you again in six months to check on that."
)


def conversation(spoken: str = SPOKEN, per_segment: int = 10) -> tuple[Segment, ...]:
    """The words at a steady 0.4 s each, cut into Segments of ``per_segment``."""
    words = spoken.split()
    segments = []

    for index in range(0, len(words), per_segment):
        timed = tuple(
            Word(
                text=word,
                start=round((index + offset) * 0.4, 1),
                end=round((index + offset + 1) * 0.4, 1),
                probability=0.9,
            )
            for offset, word in enumerate(words[index : index + per_segment])
        )
        segments.append(
            Segment(
                start=timed[0].start,
                end=timed[-1].end,
                text=" ".join(words[index : index + per_segment]),
                words=timed,
            )
        )

    return tuple(segments)


CONVERSATION = conversation()


def test_one_chunk_per_spoken_sentence():
    assert len(chunk_conversation(CONVERSATION, SCHEME)) == SPOKEN.count(
        "."
    ) + SPOKEN.count("?")


def test_chunks_are_in_time_order_and_do_not_overlap():
    chunks = chunk_conversation(CONVERSATION, SCHEME)

    assert all(
        earlier.end <= later.start
        for earlier, later in zip(chunks, chunks[1:], strict=False)
    )


def test_a_sentence_that_crosses_a_segment_boundary_is_one_chunk():
    """The Transcriber's segmentation is a unit of transcription, not evidence."""
    boundaries = {segment.start for segment in CONVERSATION[1:]}
    chunks = chunk_conversation(CONVERSATION, SCHEME)

    assert (
        sum(
            any(chunk.start < boundary < chunk.end for boundary in boundaries)
            for chunk in chunks
        )
        >= len(chunks) - 1
    )


def test_a_chunk_carries_the_words_its_boundaries_are_read_from():
    chunk = chunk_conversation(CONVERSATION, SCHEME)[0]

    assert chunk.start == chunk.words[0].start
    assert chunk.end == chunk.words[-1].end
    assert chunk.span == (0.0, 3.2)


def test_chunk_text_is_normalized_so_a_question_reduces_to_the_same_tokens():
    texts = [chunk.text for chunk in chunk_conversation(CONVERSATION, SCHEME)]

    assert any("135/88" in text for text in texts)
    assert any("100 mg" in text for text in texts)


def test_a_merged_token_keeps_every_word_it_was_read_from():
    reading = next(
        chunk
        for chunk in chunk_conversation(CONVERSATION, SCHEME)
        if "135/88" in chunk.text
    )
    merged = [word.text for word in reading.words]

    assert merged[merged.index("135") : merged.index("135") + 3] == [
        "135",
        "over",
        "88",
    ]


def test_a_trailing_run_with_no_terminal_punctuation_is_still_a_chunk():
    """The Transcriber does not always punctuate the last utterance."""
    spoken = conversation("take 100 mg daily. and then we stop")

    assert [chunk.text for chunk in chunk_conversation(spoken, SCHEME)][-1] == (
        "and then we stop"
    )


def test_an_unpunctuated_run_is_cut_at_the_guard_rather_than_left_whole():
    spoken = conversation("one two three four five six seven eight", per_segment=8)

    chunks = chunk_conversation(spoken, SentenceScheme(max_words=3))

    assert [len(chunk.words) for chunk in chunks] == [3, 3, 2]


def test_a_conversation_that_transcribed_to_nothing_yields_no_candidates():
    assert chunk_conversation((), SCHEME) == ()


@pytest.mark.parametrize("max_words", [0, -1])
def test_a_guard_that_would_not_advance_is_refused(max_words):
    with pytest.raises(ValueError):
        SentenceScheme(max_words=max_words)
