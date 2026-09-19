"""Where to pad the returned Evidence Span, and how much it buys.

The refinement `medapp.span_refiner.refine_span` performs is pure arithmetic —
a Chunk, the Conversation's Words and two offsets in, a Span out — so it is
measured the way `harness.entailment_threshold` measures a threshold: expensive
model passes run once per Question, over every Positive of the fold, and every
candidate padding is scored afterwards by pure arithmetic over the recorded
result.

The metric is mean temporal IoU over annotated Positives, exactly as
`utils.mean_temporal_iou` computes it: a Positive the shipped Answerer answers
no, or nothing was retrieved for, contributes 0 whatever the padding is —
padding only ever moves the boundaries of a Chunk that was already cited.

Resampling is at Conversation level, for the reason `harness.bootstrap`
records: the Positives of one Conversation share one transcript and one set of
Chunks and are not independent observations.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product

from harness import transcript_cache
from harness.bootstrap import interval, resamples
from harness.folds import FoldName, conversations_in_fold
from medapp.answerer import Answerer
from medapp.span_refiner import SpanPadding, refine_span
from medapp.types import Chunk, Word
from utils import Span, gold_evidence, temporal_iou

# Candidate offsets in seconds, in both directions from the Chunk's own
# boundary. Evaluated as every (start, end) pair, since a system that trims a
# hair off one hallucinated edge while padding the other is exactly what a
# joint sweep can find and an offset shared between the two cannot.
CANDIDATE_PADS: tuple[float, ...] = (
    -0.5,
    -0.3,
    -0.15,
    0.0,
    0.15,
    0.3,
    0.5,
    0.75,
    1.0,
)


@dataclass(frozen=True, slots=True)
class PositiveMeasurement:
    """What one annotated Positive got from the shipped Answerer.

    Recorded once and swept over afterwards: every candidate padding reads the
    same retrieved Chunk and the same Words, and paying for the model passes
    per candidate would buy nothing.

    Attributes:
        answered_yes: Whether the Answerer answered this Positive yes at all.
            A no contributes 0 to mean tIoU whatever the padding, and there is
            no Chunk to refine.
        chunk: The Chunk the Evidence Span was read from, or ``None`` for a no.
        words: Every Word of the Conversation, in time order — wider than the
            Chunk's own so padding can reach a neighbouring Word.
    """

    question_id: str
    transcript_id: str
    gold: Span
    answered_yes: bool
    chunk: Chunk | None
    words: tuple[Word, ...]


@dataclass(frozen=True, slots=True)
class PaddingReport:
    """One candidate (start, end) padding, scored on one fold.

    Attributes:
        mean_tiou: Mean temporal IoU over every annotated Positive, exactly as
            the shipped score computes it.
        mean_tiou_interval: The bootstrap interval on the same statistic.
    """

    start_pad: float
    end_pad: float
    mean_tiou: float
    mean_tiou_interval: tuple[float, float]


@dataclass(frozen=True, slots=True)
class PaddingSweep:
    """Every candidate padding on one fold, and what was chosen.

    Attributes:
        chosen: The candidate with the highest lower bound on mean tIoU under
            resampling — stability, not argmax, as every threshold in this
            project is chosen.
    """

    fold: FoldName
    conversations: int
    positives: int
    reports: tuple[PaddingReport, ...]
    chosen: PaddingReport


def measure_fold(fold: FoldName, answerer: Answerer) -> tuple[PositiveMeasurement, ...]:
    """Score every annotated Positive of one fold against the shipped Answerer.

    The Chunks, the index and every judge are read once per Conversation and
    every Positive Question of it is answered against them, which is exactly
    what happens inside one request.

    Args:
        fold: Which fold to read.
        answerer: The shipped Answerer, already built and warmed up. Not the
            :class:`~medapp.answerer.SpanRefiningAnswerer` wrapper — the
            padding under measurement is applied here instead, per candidate.

    Raises:
        FileNotFoundError: If a Conversation of the fold is not cached.
    """
    measurements: list[PositiveMeasurement] = []

    for transcript_id, rows in conversations_in_fold(fold):
        positives = [row for row in rows if row["question_type"] == "positive"]

        if not positives:
            continue

        segments = transcript_cache.read(transcript_id)
        words = tuple(word for segment in segments for word in segment.words)
        verdicts = answerer.answer(segments, [row["question"] for row in positives])

        for row, verdict in zip(positives, verdicts, strict=True):
            gold = gold_evidence(row)

            if gold is None:
                continue

            measurements.append(
                PositiveMeasurement(
                    question_id=row["question_id"],
                    transcript_id=transcript_id,
                    gold=gold,
                    answered_yes=verdict.answer,
                    chunk=verdict.candidates[0] if verdict.answer else None,
                    words=words,
                )
            )

    return tuple(measurements)


def score_measurement(
    measurement: PositiveMeasurement, start_pad: float, end_pad: float
) -> float:
    """The tIoU one Positive scores at one candidate padding."""
    if not measurement.answered_yes or measurement.chunk is None:
        return 0.0

    refined = refine_span(
        measurement.chunk,
        measurement.words,
        SpanPadding(start_seconds=start_pad, end_seconds=end_pad),
    )

    return temporal_iou(measurement.gold, refined)


def mean_tiou(
    measurements: Sequence[PositiveMeasurement], start_pad: float, end_pad: float
) -> float:
    """Mean tIoU over every measured Positive at one candidate padding."""
    if not measurements:
        return 0.0

    scores = [
        score_measurement(measurement, start_pad, end_pad)
        for measurement in measurements
    ]

    return sum(scores) / len(scores)


def sweep(
    fold: FoldName,
    answerer: Answerer,
    pads: Sequence[float] = CANDIDATE_PADS,
) -> PaddingSweep:
    """Measure one fold once, then score every candidate padding.

    Args:
        fold: Which fold to read.
        answerer: The shipped Answerer, already built and warmed up.
        pads: Candidate offsets, evaluated as every (start, end) pair.

    Raises:
        FileNotFoundError: If a Conversation of the fold is not cached.
        ValueError: If the fold holds no annotated Positives.
    """
    measurements = measure_fold(fold, answerer)

    if not measurements:
        raise ValueError(
            f"The {fold} fold holds no annotated Positives, so there is "
            "nothing to choose a span padding against."
        )

    resampled = resamples(measurements)
    reports = tuple(
        PaddingReport(
            start_pad=start_pad,
            end_pad=end_pad,
            mean_tiou=mean_tiou(measurements, start_pad, end_pad),
            mean_tiou_interval=interval(
                [mean_tiou(resample, start_pad, end_pad) for resample in resampled]
            ),
        )
        for start_pad, end_pad in product(pads, pads)
    )

    chosen = max(
        reports, key=lambda report: (report.mean_tiou_interval[0], report.mean_tiou)
    )

    return PaddingSweep(
        fold=fold,
        conversations=len({m.transcript_id for m in measurements}),
        positives=len(measurements),
        reports=reports,
        chosen=chosen,
    )
