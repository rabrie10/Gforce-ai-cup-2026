"""Frozen data passed between components. Times are seconds from the start of
the Conversation, the unit the wire protocol and the harness both use.
"""

from dataclasses import dataclass

from utils import Span


@dataclass(frozen=True, slots=True)
class Word:
    """One transcribed word with the timing the Transcriber aligned it to.

    Chunks are built from Word timings rather than Segment boundaries, so this
    is the finest-grained thing the system carries.

    Attributes:
        probability: The Transcriber's confidence in the token.
    """

    text: str
    start: float
    end: float
    probability: float


@dataclass(frozen=True, slots=True)
class Segment:
    """A contiguous stretch of transcribed speech, as the Transcriber emits it.

    A unit of transcription, not of evidence: one Evidence Span does not
    necessarily correspond to one Segment.
    """

    start: float
    end: float
    text: str
    words: tuple[Word, ...]


@dataclass(frozen=True, slots=True)
class Chunk:
    """A candidate Evidence Span the Answerer considers.

    Built from timed words rather than Segment boundaries and overlapping its
    neighbours. Carries the Words that let its boundaries be returned as an
    Evidence Span unchanged.

    Attributes:
        text: The passage in the normalizer's canonical written form, which is
            the form a Question is reduced to as well — so the two are matched
            as written, not as spoken.
    """

    text: str
    start: float
    end: float
    words: tuple[Word, ...]

    @property
    def span(self) -> Span:
        """The boundaries as an Evidence Span."""
        return (self.start, self.end)


@dataclass(frozen=True, slots=True)
class Verdict:
    """The Answerer's decision about one Question.

    The answer and the Evidence Span are one decision, so the Verdict carries
    the Chunk the yes was read from rather than a span alongside a ranking.
    Carrying a span separately let the two disagree: a caller that wanted the
    cited Chunk had to find it in ``candidates`` by position, and an Answerer
    that cited a Chunk it had not moved to the front silently returned a span
    for a different passage than the one it answered from.

    Attributes:
        cited: The Chunk the yes was read from; None for a no, which has
            nothing to point at.
        candidates: The ranked Chunks the verdict was chosen from. Component
            metrics are computed from these rather than by reaching inside the
            Chunker, so there is no default to omit them by. The order is the
            ranking's, and the cited Chunk is not privileged within it.

    Raises:
        ValueError: If the answer and the cited Chunk disagree.
    """

    answer: bool
    cited: Chunk | None
    candidates: tuple[Chunk, ...]

    def __post_init__(self) -> None:
        if self.answer != (self.cited is not None):
            raise ValueError(
                "A yes carries the Chunk it was read from and a no carries "
                f"none: got answer={self.answer!r}, cited={self.cited!r}."
            )

    @property
    def evidence(self) -> Span | None:
        """The boundaries to return, or None for a no."""
        return self.cited.span if self.cited is not None else None


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A Chunk with the Relevance the reranker read Question and Chunk into.

    Attributes:
        relevance: The cross-encoder's score for this Chunk against one
            Question. Comparable within one ranking and across Conversations —
            the Relevance threshold is a fixed number, not a per-Conversation
            quantile — but it is not a probability and is not calibrated.
    """

    chunk: Chunk
    relevance: float
