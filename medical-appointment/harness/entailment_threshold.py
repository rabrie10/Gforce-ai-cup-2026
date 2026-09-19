"""Where to put the Entailment threshold, and whether the Relevance gate earns
its place.

Two questions, measured on one pass over a fold because they read the same
scores.

The first is the threshold. Entailment is the judgement that separates a
Positive from a Hard Negative, so ADR-0002 assigns it Hard-Negative accuracy as
its component metric and requires Positive accuracy to be held while it is
tuned. Held is the word that needs care: adding an Entailment test to the
ticket-08 Answerer can only turn a yes into a no, so Positive accuracy is
non-increasing in the threshold and a literal reading would choose the
threshold that judges nothing. What is meant, and what is implemented here, is
that the threshold may not cost more Positives than the fold's own sampling
noise: a candidate is eligible when its Positive accuracy is at or above the
lower bound of the bootstrap interval on the ticket-08 value. Among the
eligible candidates the one with the highest lower bound on Hard-Negative
accuracy is chosen, which is stability rather than argmax, as ADR-0002 requires
of every threshold.

The second is the ablation. The Relevance gate answers no before Entailment is
judged at all, and it was tuned when Entailment did not exist. If the
Entailment judge rejects Off-Topic Questions on its own — nothing in the
Conversation establishes a Claim about a concert — the gate costs Positives for
nothing and should come out. So every candidate is scored twice, with the gate
and without it, and the gate is kept only if Off-Topic accuracy drops when it
is removed.

Resampling is at Conversation level, for the reason ``relevance_threshold``
records: the ten Questions of a Conversation share one transcript and one set
of Chunks and are not independent observations.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass

from harness import transcript_cache
from harness.bootstrap import interval, rate, resamples
from harness.folds import FoldName, conversations_in_fold
from harness.relevance_threshold import NOTHING_RETRIEVED
from medapp.chunker import ChunkScheme, chunk_conversation
from medapp.claims import Rewriter
from medapp.entailment import EntailmentJudge
from medapp.reranker import Reranker
from medapp.retrieval import Bm25Index

# A Question nothing was retrieved for has no Chunk to judge, so it is a no at
# every threshold and under either arm of the ablation.
NOT_JUDGED = float("-inf")

CANDIDATE_COUNT = 21


@dataclass(frozen=True, slots=True)
class QuestionMeasurement:
    """What one Question got from retrieval, the reranker and the NLI model.

    Recorded once and swept over afterwards: every candidate threshold and both
    arms of the ablation read the same two scores, and paying for the model
    passes per candidate would buy nothing.

    Attributes:
        relevance: The Relevance of the best Chunk, or
            :data:`~harness.relevance_threshold.NOTHING_RETRIEVED` when the
            Question shares no term with the Conversation.
        entailment: The probability that best Chunk establishes the Claim, or
            :data:`NOT_JUDGED` when there was no Chunk to judge.
        claim: The Claim the Question was rewritten as, carried so a threshold
            that goes wrong can be read against the rewrite that produced it.
        judging_seconds: Wall time for this Question alone — ranking, the
            reranker's pass, the rewrite and the NLI pass. The Chunker and the
            index are built once per Conversation and are not charged here.
    """

    question_id: str
    transcript_id: str
    question_type: str
    answer: bool
    relevance: float
    entailment: float
    claim: str
    judging_seconds: float


@dataclass(frozen=True, slots=True)
class ThresholdReport:
    """One candidate threshold under one arm of the ablation.

    Attributes:
        gated: Whether the Relevance gate was applied before Entailment.
        hard_negative_accuracy: Fraction of Hard Negatives answered no. This is
            what the threshold exists to buy.
        positive_accuracy: Fraction of Positives answered yes, which is the
            same number as the true positive rate and is what the threshold is
            constrained by rather than tuned on.
        off_topic_accuracy: Fraction of Off-Topic Questions answered no. This
            is what the ablation turns on.
        true_negative_rate: Fraction of all no Questions answered no.
        accuracy: Over every Question of the fold. Reported so the sweep can be
            read against the shipped score, never tuned on.
    """

    threshold: float
    gated: bool
    hard_negative_accuracy: float
    positive_accuracy: float
    off_topic_accuracy: float
    true_negative_rate: float
    accuracy: float
    hard_negative_interval: tuple[float, float]
    positive_interval: tuple[float, float]
    off_topic_interval: tuple[float, float]
    accuracy_interval: tuple[float, float]


@dataclass(frozen=True, slots=True)
class Ablation:
    """The Relevance gate scored against its own absence, at one threshold.

    Attributes:
        keep: Whether Off-Topic accuracy drops when the gate is removed, which
            is the only thing that keeps it.
    """

    gated: ThresholdReport
    ungated: ThresholdReport
    keep: bool


@dataclass(frozen=True, slots=True)
class ThresholdSweep:
    """Every candidate threshold on one fold, both arms, and what was chosen.

    Attributes:
        positive_floor: The lower bound of the bootstrap interval on Positive
            accuracy at the ticket-08 Answerer — Relevance gate on, nothing
            judged for Entailment. No candidate below it is eligible.
        chosen: The eligible candidate with the highest lower bound on
            Hard-Negative accuracy, under the arm the ablation kept.
        ablation: The chosen threshold scored with and without the gate.
    """

    fold: FoldName
    conversations: int
    questions: int
    reports: tuple[ThresholdReport, ...]
    positive_floor: float
    chosen: ThresholdReport
    ablation: Ablation
    mean_judging_seconds: float
    worst_judging_seconds: float


def measure_fold(
    fold: FoldName,
    scheme: ChunkScheme,
    reranker: Reranker,
    rewriter: Rewriter,
    judge: EntailmentJudge,
    candidates: int,
) -> tuple[QuestionMeasurement, ...]:
    """Score every Question of one fold the way one request scores it.

    The Chunks and the index are built once per Conversation and every Question
    of it is retrieved, reranked, rewritten and judged against them, which is
    exactly what happens inside one request.

    Args:
        fold: Which fold to read.
        scheme: The Chunk granularities to cut at.
        reranker: The Relevance judge, already loaded.
        rewriter: The Claim rewriter, already loaded.
        judge: The Entailment judge, already loaded.
        candidates: How deep a BM25 ranking the reranker is handed.

    Raises:
        FileNotFoundError: If a Conversation of the fold is not cached.
    """
    measurements: list[QuestionMeasurement] = []

    for transcript_id, rows in conversations_in_fold(fold):
        index = Bm25Index(
            chunk_conversation(transcript_cache.read(transcript_id), scheme)
        )

        for row in rows:
            started = time.perf_counter()
            reranked = reranker.rerank(
                row["question"], index.rank(row["question"], candidates)
            )
            claim = rewriter.rewrite(row["question"])
            judged = judge.judge(claim.text, [c.chunk for c in reranked[:1]])
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
                    entailment=judged[0].entailment if judged else NOT_JUDGED,
                    claim=claim.text,
                    judging_seconds=elapsed,
                )
            )

    return tuple(measurements)


def candidate_thresholds(
    measurements: Sequence[QuestionMeasurement], count: int = CANDIDATE_COUNT
) -> tuple[float, ...]:
    """Candidate cut points drawn from the entailment probabilities observed.

    The NLI model's softmax saturates at both ends — most pairs score very near
    0 or very near 1 — so an even grid over [0, 1] would spend most of its rows
    where no Question scored. Evenly spaced quantiles of the observed values
    put the candidates where the decisions are instead.

    Returns:
        The distinct candidates, ascending. A threshold *equal* to an observed
        score keeps that Question: the comparison is ``>=``.
    """
    scores = sorted(
        measurement.entailment
        for measurement in measurements
        if measurement.entailment > NOT_JUDGED
    )

    if not scores or count < 2:
        return (scores[0],) if scores else (0.0,)

    positions = [
        round(index * (len(scores) - 1) / (count - 1)) for index in range(count)
    ]

    return tuple(sorted({scores[position] for position in positions}))


def report_threshold(
    measurements: Sequence[QuestionMeasurement],
    threshold: float,
    gate: float | None,
    resampled: Sequence[Sequence[QuestionMeasurement]],
) -> ThresholdReport:
    """Score one candidate threshold, under one arm of the ablation.

    Args:
        measurements: The fold, measured once.
        threshold: The entailment probability required for a yes.
        gate: The Relevance a Question must clear before Entailment is judged,
            or None for the ungated arm.
        resampled: The resampled folds the intervals are read off.
    """
    return ThresholdReport(
        threshold=threshold,
        gated=gate is not None,
        hard_negative_accuracy=_accuracy_of_kind(
            measurements, threshold, gate, "hard_negative"
        ),
        positive_accuracy=_accuracy_of_kind(measurements, threshold, gate, "positive"),
        off_topic_accuracy=_accuracy_of_kind(
            measurements, threshold, gate, "off_topic"
        ),
        true_negative_rate=rate(
            [
                not _predicted(measurement, threshold, gate)
                for measurement in measurements
                if not measurement.answer
            ]
        ),
        accuracy=_accuracy(measurements, threshold, gate),
        hard_negative_interval=interval(
            [
                _accuracy_of_kind(resample, threshold, gate, "hard_negative")
                for resample in resampled
            ]
        ),
        positive_interval=interval(
            [
                _accuracy_of_kind(resample, threshold, gate, "positive")
                for resample in resampled
            ]
        ),
        off_topic_interval=interval(
            [
                _accuracy_of_kind(resample, threshold, gate, "off_topic")
                for resample in resampled
            ]
        ),
        accuracy_interval=interval(
            [_accuracy(resample, threshold, gate) for resample in resampled]
        ),
    )


def sweep(
    fold: FoldName,
    scheme: ChunkScheme,
    reranker: Reranker,
    rewriter: Rewriter,
    judge: EntailmentJudge,
    candidates: int,
    relevance_threshold: float,
    thresholds: Sequence[float] | None = None,
) -> ThresholdSweep:
    """Measure one fold once, then score every candidate under both arms.

    Args:
        fold: Which fold to read.
        scheme: The Chunk granularities to cut at.
        reranker: The Relevance judge, already loaded.
        rewriter: The Claim rewriter, already loaded.
        judge: The Entailment judge, already loaded.
        candidates: How deep a BM25 ranking the reranker is handed.
        relevance_threshold: The gate ticket 08 chose, which the ablation is
            against.
        thresholds: Candidates to score. Defaults to quantiles of the observed
            entailment probabilities.

    Raises:
        FileNotFoundError: If a Conversation of the fold is not cached.
        ValueError: If the fold holds no Questions.
    """
    measurements = measure_fold(fold, scheme, reranker, rewriter, judge, candidates)

    if not measurements:
        raise ValueError(
            f"The {fold} fold holds no Questions, so there is nothing to "
            "choose an Entailment threshold against."
        )

    resampled = resamples(measurements)
    scored = tuple(thresholds or candidate_thresholds(measurements))
    reports = tuple(
        report_threshold(measurements, threshold, gate, resampled)
        for gate in (relevance_threshold, None)
        for threshold in scored
    )

    floor = positive_floor(measurements, relevance_threshold, resampled)
    gated = [report for report in reports if report.gated]
    ungated = [report for report in reports if not report.gated]

    chosen_gated = choose(gated, floor)
    without_gate = next(
        report for report in ungated if report.threshold == chosen_gated.threshold
    )
    ablation = Ablation(
        gated=chosen_gated,
        ungated=without_gate,
        keep=without_gate.off_topic_accuracy < chosen_gated.off_topic_accuracy,
    )

    return ThresholdSweep(
        fold=fold,
        conversations=len({m.transcript_id for m in measurements}),
        questions=len(measurements),
        reports=reports,
        positive_floor=floor,
        chosen=ablation.gated if ablation.keep else ablation.ungated,
        ablation=ablation,
        mean_judging_seconds=sum(m.judging_seconds for m in measurements)
        / len(measurements),
        worst_judging_seconds=max(m.judging_seconds for m in measurements),
    )


def positive_floor(
    measurements: Sequence[QuestionMeasurement],
    relevance_threshold: float,
    resampled: Sequence[Sequence[QuestionMeasurement]],
) -> float:
    """The Positive accuracy no candidate may fall below.

    The ticket-08 Answerer's Positive accuracy — Relevance gate on, Entailment
    judging nothing — resampled, and read at the bottom of its interval.
    Holding the point estimate itself would admit only the threshold that
    rejects no Hard Negatives at all, since Entailment can only turn a yes into
    a no; holding the bottom of its interval says instead that the threshold
    may not cost more Positives than this fold's own sampling noise.
    """
    return interval(
        [
            _accuracy_of_kind(resample, NOT_JUDGED, relevance_threshold, "positive")
            for resample in resampled
        ]
    )[0]


def choose(reports: Sequence[ThresholdReport], floor: float) -> ThresholdReport:
    """The candidate to ship, out of the ones that were scored.

    The highest lower bound on Hard-Negative accuracy under resampling among
    the candidates that hold Positive accuracy at or above ``floor``, ties
    going to the one that keeps the most Positives.

    Raises:
        ValueError: If no candidate was scored, or none holds the floor —
            which would mean every threshold costs more Positives than it is
            allowed to, and is a result to look at rather than to round away.
    """
    if not reports:
        raise ValueError("No candidate thresholds were scored.")

    eligible = [report for report in reports if report.positive_accuracy >= floor]

    if not eligible:
        raise ValueError(
            f"No candidate threshold holds Positive accuracy at or above "
            f"{floor:.3f}, so Entailment cannot be tuned without giving up "
            "more Positives than the fold's noise allows."
        )

    return max(
        eligible,
        key=lambda report: (
            report.hard_negative_interval[0],
            report.positive_accuracy,
        ),
    )


def _predicted(
    measurement: QuestionMeasurement, threshold: float, gate: float | None
) -> bool:
    """What the Answerer answers this Question at this threshold and arm."""
    if measurement.relevance == NOTHING_RETRIEVED:
        return False

    if gate is not None and measurement.relevance < gate:
        return False

    return measurement.entailment >= threshold


def _accuracy(
    measurements: Sequence[QuestionMeasurement], threshold: float, gate: float | None
) -> float:
    return rate(
        [
            _predicted(measurement, threshold, gate) == measurement.answer
            for measurement in measurements
        ]
    )


def _accuracy_of_kind(
    measurements: Sequence[QuestionMeasurement],
    threshold: float,
    gate: float | None,
    question_type: str,
) -> float:
    return _accuracy(
        [
            measurement
            for measurement in measurements
            if measurement.question_type == question_type
        ],
        threshold,
        gate,
    )
