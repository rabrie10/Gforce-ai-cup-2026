"""The data that moves between components.

All four are frozen: a Chunk handed to the reranker is the same Chunk that is
returned as an Evidence Span, and nothing downstream may edit the timings it
will be scored on. Times are seconds from the start of the Conversation, which
is the unit the wire protocol and the harness both use.
"""

from dataclasses import dataclass

from utils import Span


@dataclass(frozen=True, slots=True)
class Word:
    """One transcribed word with the timing the Transcriber aligned it to.

    Word timings, not Segment boundaries, are what Chunks are built from, so
    this is the finest-grained thing the system carries.
    """

    text: str
    start: float
    end: float
    probability: float


@dataclass(frozen=True, slots=True)
class Segment:
    """A contiguous stretch of transcribed speech, as the Transcriber emits it.

    A unit of transcription, not of evidence: one Evidence Span does not
    necessarily correspond to one Segment. Segments are never joined into a
    single string.
    """

    start: float
    end: float
    text: str
    words: tuple[Word, ...]


@dataclass(frozen=True, slots=True)
class Chunk:
    """A candidate Evidence Span the Answerer considers.

    Built from timed words rather than Segment boundaries, overlapping its
    neighbours, and carrying the Words that let its boundaries be returned as
    an Evidence Span unchanged.
    """

    text: str
    start: float
    end: float
    words: tuple[Word, ...]

    @property
    def span(self) -> Span:
        return (self.start, self.end)


@dataclass(frozen=True, slots=True)
class Verdict:
    """The Answerer's decision about one Question.

    The answer and the Evidence Span are one decision, not two: the passage
    that justifies a yes is exactly the span returned. A no has nothing to
    point at, so ``evidence`` is ``None``.

    ``candidates`` carries the ranked Chunks the verdict was chosen from, so
    component metrics — recall@k, oracle tIoU — are computed from this seam's
    output rather than by reaching inside the Chunker. It has no default:
    measurement depends on it, and an omitted ranking is silently unmeasurable.
    """

    answer: bool
    evidence: Span | None
    candidates: tuple[Chunk, ...]

    def __post_init__(self) -> None:
        if self.answer != (self.evidence is not None):
            raise ValueError(
                "A yes carries the Evidence Span it was read from and a no "
                f"carries none: got answer={self.answer!r}, "
                f"evidence={self.evidence!r}."
            )
