"""Pure span arithmetic: pad the winning Chunk's boundaries and snap them back
to a Word edge.

A Chunk's own boundaries already sit on Word edges — its start is its first
Word's start and its end is its last Word's end — but the padding tuned on dev
is a fixed offset in seconds, not a Word count, so applying it directly would
land the Evidence Span mid-Word. Snapping afterwards, against every Word of the
Conversation rather than only the Chunk's own, is what keeps the returned span
a boundary some Word actually has.

Refinement is pure: a Chunk, the Conversation's Words and two offsets in, a
Span out. It reads nothing and calls nothing, so it is measured on dev by
sweeping the offsets over spans already retrieved rather than by re-running
retrieval per candidate.
"""

import bisect
from collections.abc import Sequence
from dataclasses import dataclass

from medapp.types import Chunk, Word
from utils import Span


@dataclass(frozen=True, slots=True)
class SpanPadding:
    """How far the cited Chunk's boundaries are pushed out before snapping.

    A fixed offset in seconds and a fraction of the Chunk's own duration,
    added. The fixed part alone cannot be right for every Chunk: annotated
    Evidence Spans run from 0.16 s to 14.2 s, and three quarters of a second
    added to the start of a Chunk cut at one token is most of a span, while the
    same offset on a Chunk cut at forty-eight is noise. A fraction alone cannot
    be right either — the Chunker's undershoot at a boundary is a Word or two
    wherever it happens, which is an absolute quantity. So both are available
    and both are swept; a fraction of zero is exactly the fixed padding this
    started as.

    Attributes:
        start_seconds: Seconds added before the Chunk's start.
        end_seconds: Seconds added after its end.
        start_fraction: Fraction of the Chunk's duration added before its
            start, on top of ``start_seconds``.
        end_fraction: The same, after its end.
    """

    start_seconds: float = 0.0
    end_seconds: float = 0.0
    start_fraction: float = 0.0
    end_fraction: float = 0.0

    def before(self, chunk: Chunk) -> float:
        """Seconds to extend this Chunk's start earlier by."""
        return self.start_seconds + self.start_fraction * (chunk.end - chunk.start)

    def after(self, chunk: Chunk) -> float:
        """Seconds to extend this Chunk's end later by."""
        return self.end_seconds + self.end_fraction * (chunk.end - chunk.start)


@dataclass(frozen=True, slots=True)
class WordEdges:
    """The Conversation's Word boundaries, sorted, ready to snap against.

    Built once per Conversation rather than once per span. A request refines at
    most ten spans and would not notice, but a sweep refines the same
    Conversation's spans tens of thousands of times over, and sorting several
    thousand Words inside each of them is the whole cost of the sweep.

    Attributes:
        starts: Every Word's start, ascending.
        ends: Every Word's end, ascending.
    """

    starts: tuple[float, ...]
    ends: tuple[float, ...]

    @classmethod
    def of(cls, words: Sequence[Word]) -> "WordEdges":
        """The edges of one Conversation's Words.

        Raises:
            ValueError: If there are no Words to snap a padded span against.
        """
        if not words:
            raise ValueError("There are no Words to snap a padded span against.")

        return cls(
            starts=tuple(sorted(word.start for word in words)),
            ends=tuple(sorted(word.end for word in words)),
        )


def refine_span(chunk: Chunk, words: Sequence[Word], padding: SpanPadding) -> Span:
    """Pad the winning Chunk's boundaries, then snap each to the nearest Word edge.

    Args:
        chunk: The Chunk the Evidence Span is read from.
        words: Every Word of the Conversation the Chunk was cut from, in time
            order — wider than the Chunk's own Words so padding can reach into
            a neighbouring Word rather than stop at the Chunk's interior.
        padding: How far to push each boundary out before snapping. Negative
            values shrink the span instead.

    Returns:
        The padded, snapped span. Snapping to the nearest edge in a sorted list
        clamps naturally at either end of the Conversation: padding past 0.0 s
        snaps to the earliest Word's start, and padding past the last Word
        snaps to its end. The end never precedes the start.

    Raises:
        ValueError: If there are no Words to snap a padded span against.
    """
    return refine_span_against(chunk, WordEdges.of(words), padding)


def refine_span_against(chunk: Chunk, edges: WordEdges, padding: SpanPadding) -> Span:
    """The same refinement, against edges already sorted.

    What :func:`refine_span` does once it has the edges. Separate so a sweep
    can hoist the sort out of its inner loop and still run the arithmetic the
    request path runs, rather than a copy of it that can drift from it.
    """
    start = _nearest(chunk.start - padding.before(chunk), edges.starts)
    end = _nearest(chunk.end + padding.after(chunk), edges.ends)

    return (start, max(start, end))


def _nearest(target: float, edges: Sequence[float]) -> float:
    """The edge in ``edges`` closest to ``target``; ties favor the earlier edge.

    ``edges`` is sorted ascending, so a ``target`` past either end naturally
    snaps to that end — the clamp at 0.0 s and at the end of the audio falls
    out of this rather than needing a case of its own.
    """
    index = bisect.bisect_left(edges, target)
    candidates = [edges[i] for i in (index - 1, index) if 0 <= i < len(edges)]

    return min(candidates, key=lambda edge: (abs(edge - target), edge))
