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

from medapp.types import Chunk, Word
from utils import Span


def refine_span(
    chunk: Chunk, words: Sequence[Word], start_pad: float, end_pad: float
) -> Span:
    """Pad the winning Chunk's boundaries, then snap each to the nearest Word edge.

    Args:
        chunk: The Chunk the Evidence Span is read from.
        words: Every Word of the Conversation the Chunk was cut from, in time
            order — wider than the Chunk's own Words so padding can reach into
            a neighbouring Word rather than stop at the Chunk's interior.
        start_pad: Seconds to extend the start earlier by, before snapping.
            Negative shrinks it instead.
        end_pad: Seconds to extend the end later by, before snapping. Negative
            shrinks it instead.

    Returns:
        The padded, snapped span. Snapping to the nearest edge in a sorted list
        clamps naturally at either end of the Conversation: padding past 0.0 s
        snaps to the earliest Word's start, and padding past the last Word
        snaps to its end. The end never precedes the start.

    Raises:
        ValueError: If there are no Words to snap a padded span against.
    """
    if not words:
        raise ValueError("There are no Words to snap a padded span against.")

    starts = sorted(word.start for word in words)
    ends = sorted(word.end for word in words)

    start = _nearest(chunk.start - start_pad, starts)
    end = _nearest(chunk.end + end_pad, ends)

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
