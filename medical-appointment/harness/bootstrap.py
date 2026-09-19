import random
from collections.abc import Iterable, Sequence
from typing import Protocol, TypeVar

# How many resampled folds an interval is read off, and the interval they are
# read at. 2000 is enough that the interval is stable to the third decimal the
# tables print.
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_INTERVAL = 0.90
BOOTSTRAP_SEED = 20260919


class Grouped(Protocol):
    """One measured Question, carrying the Conversation it is grouped by."""

    @property
    def transcript_id(self) -> str: ...


Measurement = TypeVar("Measurement", bound=Grouped)


def resamples(
    measurements: Sequence[Measurement],
    count: int = BOOTSTRAP_RESAMPLES,
) -> tuple[tuple[Measurement, ...], ...]:
    """Resampled folds, drawn at Conversation level with replacement.

    Drawn once and reused for every candidate a sweep scores, so that two
    candidates differ by the candidate rather than by the draw.
    """
    grouped: dict[str, list[Measurement]] = {}

    for measurement in measurements:
        grouped.setdefault(measurement.transcript_id, []).append(measurement)

    conversations = list(grouped.values())
    generator = random.Random(BOOTSTRAP_SEED)

    return tuple(
        tuple(
            measurement
            for _ in conversations
            for measurement in generator.choice(conversations)
        )
        for _ in range(count)
    )


def interval(rates: Sequence[float]) -> tuple[float, float]:
    """The percentile interval over one statistic's resampled values."""
    if not rates:
        return (0.0, 0.0)

    ordered = sorted(rates)
    tail = (1 - BOOTSTRAP_INTERVAL) / 2
    last = len(ordered) - 1

    return (ordered[round(tail * last)], ordered[round((1 - tail) * last)])


def rate(outcomes: Iterable[bool]) -> float:
    """The fraction of outcomes that are true, and zero over nothing."""
    counted = list(outcomes)

    return sum(counted) / len(counted) if counted else 0.0
