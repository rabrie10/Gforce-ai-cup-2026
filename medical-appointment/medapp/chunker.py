import re
from dataclasses import dataclass

from medapp.normalizer import NormalizedToken, normalize_words
from medapp.types import Chunk, Segment, Word

# What ends a spoken sentence in the Transcriber's output. Read from the raw
# Word text, before the normalizer strips punctuation: the boundary is the one
# thing the normalizer throws away that the Chunker needs.
SENTENCE_END = re.compile(r"[.?!]['\"’”)]*\s*$")


@dataclass(frozen=True, slots=True)
class SentenceScheme:
    """How a Conversation is cut into candidate Evidence Spans.

    A Chunk is one spoken sentence. The alternative measured against it was a
    ladder of overlapping fixed-length word windows, which reaches a far higher
    oracle tIoU — every annotated boundary has some window close to it — and a
    markedly lower achieved one, because it puts a dozen near-identical views
    of the same stretch in front of a Relevance model that scores what a
    passage is *about* and has no signal for where it should end. Measured over
    train and dev, 122 annotated Evidence Spans: the ladder cites a Chunk worth
    0.463 mean tIoU and sentences one worth 0.508, and adding two- and
    three-sentence Chunks alongside single ones drops it to 0.440 while raising
    the oracle. Candidate redundancy costs more than candidate coverage buys,
    so the Chunker offers one reading of each stretch and no more.

    Attributes:
        max_words: Where a run of Words carrying no terminal punctuation is cut
            anyway. Not a granularity — the supplied Conversations have a
            median sentence of five Words and a longest of fifty-nine, so this
            fires nowhere in them. It is the guard against a Conversation the
            Transcriber punctuates sparsely or not at all, where one Chunk
            would otherwise cover the whole audio and score near zero on every
            Question.

    Raises:
        ValueError: If the guard would not advance.
    """

    max_words: int

    def __post_init__(self) -> None:
        if self.max_words < 1:
            raise ValueError(
                f"A Chunk holds at least one Word: got max_words={self.max_words!r}."
            )


def chunk_conversation(
    segments: tuple[Segment, ...], scheme: SentenceScheme
) -> tuple[Chunk, ...]:
    """Cut one Conversation into candidate Evidence Spans, one per sentence.

    Args:
        segments: The whole Conversation, in time order. Segment boundaries are
            dropped: the Transcriber's segmentation is a unit of transcription
            and runs several sentences long.
        scheme: Where an unpunctuated run is cut anyway.

    Returns:
        The candidates in time order. A sentence whose Words all normalize away
        carries no text to rank and is left out.
    """
    chunks = []

    for sentence in _sentences(_words_in_time_order(segments), scheme):
        tokens = tuple(token for token in normalize_words(sentence) if token.words)

        if tokens:
            chunks.append(_chunk(tokens))

    return tuple(chunks)


def _words_in_time_order(segments: tuple[Segment, ...]) -> tuple[Word, ...]:
    """Every Word of the Conversation, with the Segment boundaries dropped."""
    return tuple(word for segment in segments for word in segment.words)


def _sentences(
    words: tuple[Word, ...], scheme: SentenceScheme
) -> list[tuple[Word, ...]]:
    """The Words grouped into sentences, in time order.

    A trailing run with no terminal punctuation is a sentence of its own: the
    Transcriber does not always punctuate the last utterance, and dropping it
    would make the end of every such Conversation unreachable.
    """
    sentences: list[tuple[Word, ...]] = []
    current: list[Word] = []

    for word in words:
        current.append(word)

        if SENTENCE_END.search(word.text) or len(current) >= scheme.max_words:
            sentences.append(tuple(current))
            current = []

    if current:
        sentences.append(tuple(current))

    return sentences


def _chunk(tokens: tuple[NormalizedToken, ...]) -> Chunk:
    """One candidate, carrying the Words its boundaries are read from."""
    words = tuple(word for token in tokens for word in token.words)

    return Chunk(
        text=" ".join(token.text for token in tokens),
        start=words[0].start,
        end=words[-1].end,
        words=words,
    )
