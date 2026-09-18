"""Overlapping, multi-length candidate Evidence Spans cut from timed Words.

An annotated Evidence Span runs from 0.16 s to 14.20 s, 11 Positives share a
span with another Question of the same Conversation and 12 sit strictly inside
another's, so no partition of the Conversation can represent them: a Chunk
scheme that does not overlap makes some correct answers structurally
unreachable. Chunks are therefore cut at several lengths, each length sliding
across the Conversation at a fraction of its own stride.

Boundaries come from Word timings rather than from Segment boundaries or
sentence punctuation. 30 of 122 annotated spans cross a Segment boundary, and 8
of the 24 read Conversations lose punctuation for a run of 40 words or more, so
neither is available as a cut.

Cutting happens over *normalized* tokens rather than raw Words. The normalizer
merges several Words into one token — "135 over 88" into "135/88" — and a cut
through the middle of such a token would hand the retriever half a reading.
Every token carries the Words it was read from, so a Chunk cut this way still
carries the Word timings that make it returnable as an Evidence Span unchanged.
"""

from dataclasses import dataclass

from medapp.normalizer import NormalizedToken, normalize_words
from medapp.types import Chunk, Segment, Word


@dataclass(frozen=True, slots=True)
class ChunkScheme:
    """The granularities a Conversation is cut at.

    Attributes:
        word_lengths: Chunk lengths in normalized tokens. A ladder rather than
            one value, because the span a Question is answered from may be one
            word or forty.
        stride_fraction: How far apart consecutive Chunks of one length start,
            as a fraction of that length. Below 1 the Chunks of a length
            overlap, which is what lets a span that straddles two cuts still
            have a candidate close to it.

    Raises:
        ValueError: If the ladder is empty, holds a non-positive length, or the
            stride would not advance.
    """

    word_lengths: tuple[int, ...]
    stride_fraction: float

    def __post_init__(self) -> None:
        if not self.word_lengths:
            raise ValueError("A Chunk scheme needs at least one length.")

        if any(length < 1 for length in self.word_lengths):
            raise ValueError(
                f"Chunk lengths are counted in tokens and must be positive: "
                f"got {self.word_lengths!r}."
            )

        if not 0 < self.stride_fraction <= 1:
            raise ValueError(
                f"The stride fraction must be over 0 and at most 1, so that "
                f"Chunks advance and overlap: got {self.stride_fraction!r}."
            )

    def stride(self, length: int) -> int:
        """How far apart consecutive Chunks of ``length`` tokens start."""
        return max(1, round(length * self.stride_fraction))


def chunk_conversation(
    segments: tuple[Segment, ...], scheme: ChunkScheme
) -> tuple[Chunk, ...]:
    """Cut one Conversation into overlapping candidate Evidence Spans.

    Args:
        segments: The whole Conversation, in time order.
        scheme: The granularities to cut at, as Settings resolved them.

    Returns:
        The candidates in time order, each one distinct: the same stretch
        reached by two lengths — at the tail of a Conversation shorter than the
        length above it — is carried once.
    """
    spoken = normalize_words(_words_in_time_order(segments))
    tokens = tuple(token for token in spoken if token.words)

    if not tokens:
        return ()

    windows = sorted({window for window in _windows(len(tokens), scheme)})

    return tuple(_chunk(tokens[start:end]) for start, end in windows)


def _words_in_time_order(segments: tuple[Segment, ...]) -> tuple[Word, ...]:
    """Every Word of the Conversation, with the Segment boundaries dropped."""
    return tuple(word for segment in segments for word in segment.words)


def _windows(token_count: int, scheme: ChunkScheme) -> list[tuple[int, int]]:
    """The half-open token ranges the scheme cuts, one length at a time.

    A length longer than the Conversation still yields the whole of it, so a
    Conversation shorter than the ladder's top rung is not left uncovered.
    """
    windows: list[tuple[int, int]] = []

    for length in scheme.word_lengths:
        for start in range(0, token_count, scheme.stride(length)):
            end = min(start + length, token_count)
            windows.append((start, end))

            if end == token_count:
                break

    return windows


def _chunk(tokens: tuple[NormalizedToken, ...]) -> Chunk:
    """One candidate, carrying the Words its boundaries are read from."""
    words = tuple(word for token in tokens for word in token.words)

    return Chunk(
        text=" ".join(token.text for token in tokens),
        start=words[0].start,
        end=words[-1].end,
        words=words,
    )
