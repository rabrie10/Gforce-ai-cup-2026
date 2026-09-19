"""Bootstrap resampling over a fold, at Conversation level.

ADR-0002 chooses every threshold for stability under resampling rather than by
argmax over a handful of groups, and reports every number with an interval
because the folds are small enough that one is misleading without it. Both are
this module's job, and both are shared by the thresholds that are tuned
separately.

Resampling is at Conversation level. The ten Questions of a Conversation share
one transcript and one set of Chunks, so they are drawn together or not at all
— resampling Questions would treat them as ten independent observations and
report an interval far narrower than the fold supports.
"""

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
