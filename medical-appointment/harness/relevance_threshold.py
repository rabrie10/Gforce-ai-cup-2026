"""Where to put the Relevance threshold, measured on a fold.

The threshold is the only free parameter of the Relevance judgement, and it
decides one thing: how far the best Chunk of a Conversation has to be about
what a Question asks about before the answer is yes. Too low and an Off-Topic
Question finds something; too high and Positives lose the passage that makes
them true. The two move against each other, so both are reported at every
candidate — ADR-0002 requires TPR and TNR separately, because one blended
number cannot see a threshold that is buying its accuracy entirely from one
Question type.

Hard Negatives are reported beside them and are not what the threshold is tuned
on. Their best Chunk is highly Relevant by construction, so no threshold
rejects them without taking the Positives with it; that is the Entailment
judge's failure to fix, and pulling this threshold up to chase it would look
like progress on the blended number while destroying the half of the score that
is already working.

ADR-0002 also fixes *what* the value is chosen on and *how*: on Off-Topic
accuracy, the component metric this threshold owns, and for stability under
bootstrap resampling rather than by argmax over a fold of 16 Conversations,
which is substantially noise. So the chosen value is the lowest threshold whose
Off-Topic accuracy survives resampling, which leaves the Positives everything
above it would have cost. Resampling is at Conversation level, because the ten
Questions of a Conversation share one transcript and one set of Chunks and are
not independent observations.
"""

import random
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from harness import transcript_cache
from harness.folds import FoldName, questions_in_fold
from medapp.chunker import ChunkScheme, chunk_conversation
from medapp.reranker import Reranker
from medapp.retrieval import Bm25Index

# A Question whose best Chunk did not reach the threshold is answered no, so a
# Question nothing was retrieved for must be a no at every candidate, including
# a candidate of zero.
NOTHING_RETRIEVED = float("-inf")

# How many resampled folds a bootstrap interval is read off, and the interval
# they are read at. 2000 is enough that the interval is stable to the third
# decimal the table prints.
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_INTERVAL = 0.90
BOOTSTRAP_SEED = 20260919

CANDIDATE_COUNT = 21

# The spec's latency budget gives all ten Questions of a Conversation 15 s
# worst case. Retrieval and reranking are what this module measures; Entailment
# is not built yet and will be judged inside the same 15 s.
QUESTIONS_PER_CONVERSATION = 10
JUDGING_BUDGET_SECONDS = 15.0


@dataclass(frozen=True, slots=True)
class QuestionMeasurement:
    """What one Question got from retrieval and the reranker.

    Recorded once and swept over afterwards, because every candidate threshold
    reads the same scores: the model pass is what the measurement costs, and
    paying it per threshold would buy nothing.

    Attributes:
        answer: The annotated answer, which is what a threshold is scored
            against.
        relevance: The Relevance of the best Chunk, or
            :data:`NOTHING_RETRIEVED` when the Question shares no term with the
            Conversation.
        judging_seconds: Wall time for this Question alone — ranking and the
            reranker's forward pass. The Chunker and the index are built once
            per Conversation and are not charged here.
    """

    question_id: str
    transcript_id: str
    question_type: str
    answer: bool
    relevance: float
    judging_seconds: float


@dataclass(frozen=True, slots=True)
class ThresholdReport:
    """One candidate threshold, scored on the fold and under resampling.

    Attributes:
        off_topic_accuracy: Fraction of Off-Topic Questions answered no. This
            is what the threshold exists to buy.
        hard_negative_accuracy: Fraction of Hard Negatives answered no.
            Reported, not tuned on.
        true_positive_rate: Fraction of Positives answered yes.
        true_negative_rate: Fraction of all no Questions answered no, Hard
            Negatives included.
        accuracy: Over every Question of the fold. Reported so the sweep can
            be read against the shipped score, and not what the threshold is
            chosen on — blended accuracy would pull the threshold up until it
            was rejecting Hard Negatives, which is the Entailment judge's job.
        off_topic_interval: The bootstrap interval on ``off_topic_accuracy``,
            resampled at Conversation level. This is what the threshold is
            chosen on.
        accuracy_interval: The same interval on ``accuracy``.
    """

    threshold: float
    off_topic_accuracy: float
    hard_negative_accuracy: float
    true_positive_rate: float
    true_negative_rate: float
    accuracy: float
    off_topic_interval: tuple[float, float]
    accuracy_interval: tuple[float, float]


@dataclass(frozen=True, slots=True)
class ThresholdSweep:
    """Every candidate threshold on one fold, and the one that is chosen.

    Attributes:
        chosen: The candidate with the highest lower bound on Off-Topic
            accuracy under resampling, ties going to the one that keeps the
            most Positives. ADR-0002 tunes this threshold on Off-Topic accuracy
            and chooses for stability rather than by argmax, so this is the
            lowest threshold that rejects Off-Topic Questions dependably —
            anything above it is bought from the Positives.
        judging_seconds: Per-Question judging time: mean, and the worst single
            Question of the fold.
    """

    fold: FoldName
    conversations: int
    questions: int
    reports: tuple[ThresholdReport, ...]
    chosen: ThresholdReport
    mean_judging_seconds: float
    worst_judging_seconds: float


def measure_fold(
    fold: FoldName, scheme: ChunkScheme, reranker: Reranker, candidates: int
) -> tuple[QuestionMeasurement, ...]:
    """Score every Question of one fold against its Conversation's Chunks.

    The Chunks and the index are built once per Conversation and every Question
    of it is retrieved and reranked against them, which is exactly what happens
    inside one request.

    Args:
        fold: Which fold to read.
        scheme: The Chunk granularities to cut at.
        reranker: The Relevance judge, already loaded.
        candidates: How deep a BM25 ranking the reranker is handed.

    Raises:
        FileNotFoundError: If a Conversation of the fold is not cached.
    """
    measurements: list[QuestionMeasurement] = []

    for transcript_id, rows in _conversations_in(fold):
        index = Bm25Index(
            chunk_conversation(transcript_cache.read(transcript_id), scheme)
        )

        for row in rows:
            started = time.perf_counter()
            reranked = reranker.rerank(
                row["question"], index.rank(row["question"], candidates)
            )
            elapsed = time.perf_counter() - started

            measurements.append(
                QuestionMeasurement(
                    question_id=row["question_id"],
                    transcript_id=transcript_id,
                    question_type=row["question_type"],
                    answer=row["answer"] == "yes",
                    relevance=(
                        reranked[0].relevance if reranked else NOTHING_RETRIEVED
                    ),
                    judging_seconds=elapsed,
                )
            )

    return tuple(measurements)


def candidate_thresholds(
    measurements: Sequence[QuestionMeasurement], count: int = CANDIDATE_COUNT
) -> tuple[float, ...]:
    """Candidate cut points drawn from the scores actually observed.

    A fixed grid over [0, 1] would spend most of its rows where no Question
    scored: the reranker's sigmoid saturates, and what separates the Question
    types sits in the gaps between the observed values. So the candidates are
    evenly spaced quantiles of the observed Relevance instead, which puts them
    where the decisions are.

    Returns:
        The distinct candidates, ascending. A threshold *equal* to an observed
        score keeps that Question — the comparison is ``>=`` — so the lowest
        candidate answers everything retrieved yes.
    """
    scores = sorted(
        measurement.relevance
        for measurement in measurements
        if measurement.relevance > NOTHING_RETRIEVED
    )

    if not scores:
        return (0.0,)

    positions = [
        round(index * (len(scores) - 1) / (count - 1)) for index in range(count)
    ]

    return tuple(sorted({scores[position] for position in positions}))


def report_threshold(
    measurements: Sequence[QuestionMeasurement],
    threshold: float,
    resampled: Sequence[Sequence[QuestionMeasurement]],
) -> ThresholdReport:
    """Score one candidate threshold on the fold and on its resamples."""
    return ThresholdReport(
        threshold=threshold,
        off_topic_accuracy=_accuracy_of_type(measurements, threshold, "off_topic"),
        hard_negative_accuracy=_accuracy_of_type(
            measurements, threshold, "hard_negative"
        ),
        true_positive_rate=_rate(
            [
                _predicted(measurement, threshold)
                for measurement in measurements
                if measurement.answer
            ]
        ),
        true_negative_rate=_rate(
            [
                not _predicted(measurement, threshold)
                for measurement in measurements
                if not measurement.answer
            ]
        ),
        accuracy=_accuracy(measurements, threshold),
        off_topic_interval=_interval(
            [
                _accuracy_of_type(resample, threshold, "off_topic")
                for resample in resampled
            ]
        ),
        accuracy_interval=_interval(
            [_accuracy(resample, threshold) for resample in resampled]
        ),
    )


def sweep(
    fold: FoldName,
    scheme: ChunkScheme,
    reranker: Reranker,
    candidates: int,
    thresholds: Sequence[float] | None = None,
) -> ThresholdSweep:
    """Measure one fold once and score every candidate threshold on it.

    Args:
        fold: Which fold to read.
        scheme: The Chunk granularities to cut at.
        reranker: The Relevance judge, already loaded.
        candidates: How deep a BM25 ranking the reranker is handed.
        thresholds: Candidates to score. Defaults to quantiles of the observed
            Relevance.

    Raises:
        FileNotFoundError: If a Conversation of the fold is not cached.
        ValueError: If the fold holds no Questions, which would leave every
            threshold scored over nothing.
    """
    measurements = measure_fold(fold, scheme, reranker, candidates)

    if not measurements:
        raise ValueError(
            f"The {fold} fold holds no Questions, so there is nothing to "
            "choose a Relevance threshold against."
        )

    resampled = resamples(measurements)
    reports = tuple(
        report_threshold(measurements, threshold, resampled)
        for threshold in (thresholds or candidate_thresholds(measurements))
    )

    return ThresholdSweep(
        fold=fold,
        conversations=len({m.transcript_id for m in measurements}),
        questions=len(measurements),
        reports=reports,
        chosen=choose(reports),
        mean_judging_seconds=sum(m.judging_seconds for m in measurements)
        / len(measurements),
        worst_judging_seconds=max(m.judging_seconds for m in measurements),
    )


def choose(reports: Sequence[ThresholdReport]) -> ThresholdReport:
    """The candidate to ship, out of the ones that were scored.

    The highest lower bound on Off-Topic accuracy under resampling, ties going
    to the one that keeps the most Positives — which, since Off-Topic accuracy
    rises with the threshold and TPR falls with it, is the lowest threshold
    that rejects Off-Topic Questions dependably. Everything above that is
    bought from the Positives, and everything it leaves behind is a Hard
    Negative, which is the Entailment judge's to reject.

    Raises:
        ValueError: If no candidate was scored.
    """
    if not reports:
        raise ValueError("No candidate thresholds were scored.")

    return max(
        reports,
        key=lambda report: (report.off_topic_interval[0], report.true_positive_rate),
    )


def _predicted(measurement: QuestionMeasurement, threshold: float) -> bool:
    """What the Answerer answers this Question at this threshold."""
    return measurement.relevance >= threshold


def _accuracy(measurements: Sequence[QuestionMeasurement], threshold: float) -> float:
    return _rate(
        [
            _predicted(measurement, threshold) == measurement.answer
            for measurement in measurements
        ]
    )


def _accuracy_of_type(
    measurements: Sequence[QuestionMeasurement], threshold: float, question_type: str
) -> float:
    return _accuracy(
        [
            measurement
            for measurement in measurements
            if measurement.question_type == question_type
        ],
        threshold,
    )


def resamples(
    measurements: Sequence[QuestionMeasurement],
) -> tuple[tuple[QuestionMeasurement, ...], ...]:
    """Resampled folds, drawn at Conversation level with replacement.

    Drawn once and reused for every candidate threshold, so that two thresholds
    differ by the threshold rather than by the draw. The ten Questions of a
    Conversation share one transcript and one set of Chunks, so they are drawn
    together or not at all — resampling Questions would treat them as ten
    independent observations and report an interval far narrower than the fold
    supports.
    """
    grouped: dict[str, list[QuestionMeasurement]] = {}

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
        for _ in range(BOOTSTRAP_RESAMPLES)
    )


def _interval(rates: Sequence[float]) -> tuple[float, float]:
    """The percentile interval over one statistic's resampled values."""
    if not rates:
        return (0.0, 0.0)

    ordered = sorted(rates)
    tail = (1 - BOOTSTRAP_INTERVAL) / 2
    last = len(ordered) - 1

    return (ordered[round(tail * last)], ordered[round((1 - tail) * last)])


def _conversations_in(fold: FoldName) -> list[tuple[str, list[dict[str, str]]]]:
    """The fold's Conversations with their Question rows, in CSV order."""
    grouped: dict[str, list[dict[str, str]]] = {}

    for row in questions_in_fold(fold):
        grouped.setdefault(row["transcript_id"], []).append(row)

    return list(grouped.items())


def _rate(outcomes: Iterable[bool]) -> float:
    counted = list(outcomes)

    return sum(counted) / len(counted) if counted else 0.0
