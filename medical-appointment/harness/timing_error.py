import math
import statistics
from bisect import bisect_left
from dataclasses import dataclass

from harness import transcript_cache
from harness.folds import FoldName, questions_in_fold
from medapp.types import Segment
from utils import gold_evidence


@dataclass(frozen=True, slots=True)
class BoundaryError:
    """The two distances one annotated Evidence Span is off by.

    Attributes:
        transcript_id: The Conversation the span belongs to.
        start_error: Seconds from the annotated start to the nearest word
            boundary.
        end_error: The same for the annotated end.
    """

    transcript_id: str
    start_error: float
    end_error: float


@dataclass(frozen=True, slots=True)
class TimingErrorReport:
    """The timing error over one fold.

    Attributes:
        spans: How many annotated Evidence Spans the fold contributed.
        median: Median boundary distance in seconds, over both boundaries of
            every span.
        p90: The 90th percentile of the same distances — the tail that decides
            the worst spans rather than the typical one.
        worst: The single largest distance.
    """

    fold: FoldName
    spans: int
    median: float
    p90: float
    worst: float


def word_boundaries(segments: tuple[Segment, ...]) -> tuple[float, ...]:
    """Every instant a returned Evidence Span could be snapped to, sorted."""
    return tuple(
        sorted(
            {
                time
                for segment in segments
                for word in segment.words
                for time in (word.start, word.end)
            }
        )
    )


def distance_to_nearest_boundary(time: float, boundaries: tuple[float, ...]) -> float:
    """Seconds from ``time`` to the closest word boundary.

    Raises:
        ValueError: If the transcript holds no word boundaries at all.
    """
    if not boundaries:
        raise ValueError(
            "The transcript carries no word boundaries, so no Evidence Span "
            "could be returned from it."
        )

    position = bisect_left(boundaries, time)

    candidates = []
    if position < len(boundaries):
        candidates.append(boundaries[position])
    if position > 0:
        candidates.append(boundaries[position - 1])

    return min(abs(time - candidate) for candidate in candidates)


def boundary_errors(fold: FoldName) -> list[BoundaryError]:
    """The boundary distances for every annotated Evidence Span in one fold.

    Raises:
        FileNotFoundError: If a Conversation of the fold is not cached.
    """
    boundaries_by_transcript: dict[str, tuple[float, ...]] = {}
    errors: list[BoundaryError] = []

    for row in questions_in_fold(fold):
        annotated = gold_evidence(row)
        if annotated is None:
            continue

        transcript_id = row["transcript_id"]
        if transcript_id not in boundaries_by_transcript:
            boundaries_by_transcript[transcript_id] = word_boundaries(
                transcript_cache.read(transcript_id)
            )

        boundaries = boundaries_by_transcript[transcript_id]
        start, end = annotated
        errors.append(
            BoundaryError(
                transcript_id=transcript_id,
                start_error=distance_to_nearest_boundary(start, boundaries),
                end_error=distance_to_nearest_boundary(end, boundaries),
            )
        )

    return errors


def report(fold: FoldName) -> TimingErrorReport:
    """Summarise one fold's boundary distances.

    Raises:
        FileNotFoundError: If a Conversation of the fold is not cached.
        ValueError: If the fold holds no annotated Evidence Spans.
    """
    errors = boundary_errors(fold)

    if not errors:
        raise ValueError(
            f"The {fold} fold holds no annotated Evidence Spans, so there is "
            "no timing error to measure."
        )

    distances = sorted(
        distance
        for error in errors
        for distance in (error.start_error, error.end_error)
    )

    return TimingErrorReport(
        fold=fold,
        spans=len(errors),
        median=statistics.median(distances),
        p90=_percentile(distances, 0.9),
        worst=distances[-1],
    )


def _percentile(sorted_values: list[float], fraction: float) -> float:
    """The value at ``fraction`` through an already-sorted list.

    Nearest-rank rather than interpolated: with a few hundred distances the
    reported tail should be one the data actually contains. Rounded up, so a
    rank landing between two values reports the larger — this number exists to
    expose the tail, not to round it away.
    """
    rank = max(1, min(len(sorted_values), math.ceil(fraction * len(sorted_values))))

    return sorted_values[rank - 1]
