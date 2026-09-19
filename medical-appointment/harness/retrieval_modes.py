"""The three retrieval modes, measured on one fold against ADR-0001's gate.

ADR-0001 builds BM25 first and adopts the dense half only on evidence. The
evidence it asks for is specific, and it is not recall: recall@k alone can
improve while the thing 142 Questions depend on gets worse, because fusion can
promote a lexically near-identical but factually wrong Chunk above the correct
one — raising recall of *something relevant* while destroying the distinction
between a Positive and a Hard Negative. So **Hard-Negative accuracy is the
gate**, evaluated alongside recall@5 rather than after it.

Each mode is therefore measured end to end, through the shipped Answerer's own
path: retrieve, rerank, rewrite the Question as a Claim, judge Entailment at
the thresholds Settings carry. That is what makes the Hard-Negative number the
one the system actually scores rather than a proxy for it. The retrieval
numbers — recall@5 at the reranked ranking, and oracle-selected tIoU over every
Chunk — come off the same pass, so the two halves of the gate cannot drift
apart between runs.

Oracle tIoU does not depend on the retriever at all: it is the best any Chunk
of the Conversation achieves, which is the Chunker's ceiling. It is reported
per mode anyway, because a mode that changed it would mean this module was
measuring something other than what it claims to.

Resampling is at Conversation level, for the reason the other sweeps record.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass

from harness import transcript_cache
from harness.bootstrap import interval, rate, resamples
from harness.folds import FoldName, conversations_in_fold
from harness.retrieval_metrics import HIT_TIOU
from medapp.chunker import ChunkScheme, chunk_conversation
from medapp.claims import Rewriter
from medapp.config import RetrievalMode, Settings
from medapp.dense import Embedder
from medapp.dense import warm_up as warm_embedder
from medapp.entailment import EntailmentJudge
from medapp.reranker import Reranker
from medapp.retrieval import Bm25Index, build_index_factory
from medapp.types import ScoredChunk
from utils import gold_evidence, temporal_iou

# Where recall is read. ADR-0002 states the retriever's gate at 5 and requires
# it re-measured after every component that changes the ranking; the dense half
# is the last one that can move it.
RECALL_DEPTH = 5


@dataclass(frozen=True, slots=True)
class QuestionMeasurement:
    """What one Question got from one retrieval mode, end to end.

    Attributes:
        predicted: What the Answerer answered under the shipped thresholds,
            which is what the Hard-Negative and Positive numbers are read off.
        oracle_tiou: The best tIoU any Chunk of the Conversation achieves, or
            None for a Question with no annotated Evidence Span.
        ranked_tiou: The best tIoU within the reranked top
            :data:`RECALL_DEPTH`, or None for a Question with no annotation.
    """

    question_id: str
    transcript_id: str
    question_type: str
    answer: bool
    predicted: bool
    oracle_tiou: float | None
    ranked_tiou: float | None
    judging_seconds: float


@dataclass(frozen=True, slots=True)
class ModeReport:
    """One retrieval mode on one fold, with an interval on every number.

    Attributes:
        hard_negative_accuracy: Fraction of Hard Negatives answered no. This is
            the gate.
        recall_at_5: Fraction of annotated Evidence Spans the reranked top 5
            holds a Chunk overlapping. This is the number the gate is evaluated
            alongside, never instead of.
        oracle_tiou: The Chunker's ceiling, which no retriever moves.
        index_seconds: Mean wall time to build one Conversation's index — the
            cost the dense half adds, paid once per request rather than per
            Question.
        worst_index_seconds: The same for the slowest Conversation of the
            fold. The dense pass scales with the Chunk count, which runs from
            1,228 to 4,376 on dev, so the mean is not what the latency budget
            has to hold.
    """

    mode: RetrievalMode
    conversations: int
    questions: int
    spans: int
    hard_negative_accuracy: float
    hard_negative_interval: tuple[float, float]
    positive_accuracy: float
    positive_interval: tuple[float, float]
    off_topic_accuracy: float
    recall_at_5: float
    recall_interval: tuple[float, float]
    oracle_tiou: float
    oracle_interval: tuple[float, float]
    accuracy: float
    accuracy_interval: tuple[float, float]
    index_seconds: float
    worst_index_seconds: float
    mean_judging_seconds: float
    worst_judging_seconds: float


@dataclass(frozen=True, slots=True)
class ModeComparison:
    """Every mode measured on one fold, and the one ADR-0001's gate adopts."""

    fold: FoldName
    baseline: ModeReport
    reports: tuple[ModeReport, ...]
    chosen: ModeReport


def measure_fold(
    fold: FoldName,
    mode: RetrievalMode,
    settings: Settings,
    reranker: Reranker,
    rewriter: Rewriter,
    judge: EntailmentJudge,
    embedder: Embedder | None = None,
) -> tuple[tuple[QuestionMeasurement, ...], tuple[float, ...]]:
    """Answer every Question of one fold under one retrieval mode.

    The Chunks and the index are built once per Conversation and every Question
    of it is answered against them, which is exactly what happens inside one
    request.

    Args:
        fold: Which fold to read.
        mode: Which retriever to rank with.
        settings: The resolved environment, carrying the thresholds the
            answers are read at and the embedder the mode may need.
        reranker: The Relevance judge, already loaded.
        rewriter: The Claim rewriter, already loaded.
        judge: The Entailment judge, already loaded.
        embedder: The bi-encoder, already loaded, for the modes that rank with
            one.

    Returns:
        Every Question of the fold, and the seconds each Conversation's index
        cost to build.

    Raises:
        FileNotFoundError: If a Conversation of the fold is not cached.
        ValueError: If the mode ranks with an embedder Settings did not load.
    """
    scheme = ChunkScheme(
        word_lengths=settings.chunk_word_lengths,
        stride_fraction=settings.chunk_stride_fraction,
    )
    build = build_index_factory(
        settings.model_copy(update={"retrieval_mode": mode}), embedder
    )
    gate = settings.relevance_threshold if settings.relevance_gate else None

    measurements: list[QuestionMeasurement] = []
    index_times: list[float] = []

    for transcript_id, rows in conversations_in_fold(fold):
        chunks = chunk_conversation(transcript_cache.read(transcript_id), scheme)

        started = time.perf_counter()
        index = build(chunks)
        index_times.append(time.perf_counter() - started)

        for row in rows:
            started = time.perf_counter()
            reranked = reranker.rerank(
                row["question"],
                index.rank(row["question"], settings.retrieval_candidates),
            )
            predicted = _answered_yes(
                row["question"], reranked, settings, gate, rewriter, judge
            )
            elapsed = time.perf_counter() - started

            annotated = gold_evidence(row)
            measurements.append(
                QuestionMeasurement(
                    question_id=row["question_id"],
                    transcript_id=transcript_id,
                    question_type=row["question_type"],
                    answer=row["answer"] == "yes",
                    predicted=predicted,
                    oracle_tiou=(
                        None
                        if annotated is None
                        else max(
                            (temporal_iou(annotated, chunk.span) for chunk in chunks),
                            default=0.0,
                        )
                    ),
                    ranked_tiou=(
                        None
                        if annotated is None
                        else max(
                            (
                                temporal_iou(annotated, candidate.chunk.span)
                                for candidate in reranked[:RECALL_DEPTH]
                            ),
                            default=0.0,
                        )
                    ),
                    judging_seconds=elapsed,
                )
            )

    return tuple(measurements), tuple(index_times)


def report_mode(
    mode: RetrievalMode,
    measurements: Sequence[QuestionMeasurement],
    index_times: Sequence[float],
) -> ModeReport:
    """Summarise one mode, with a bootstrap interval on every number."""
    resampled = resamples(measurements)
    spans = [one for one in measurements if one.ranked_tiou is not None]

    return ModeReport(
        mode=mode,
        conversations=len({one.transcript_id for one in measurements}),
        questions=len(measurements),
        spans=len(spans),
        hard_negative_accuracy=_accuracy_of_kind(measurements, "hard_negative"),
        hard_negative_interval=interval(
            [_accuracy_of_kind(sample, "hard_negative") for sample in resampled]
        ),
        positive_accuracy=_accuracy_of_kind(measurements, "positive"),
        positive_interval=interval(
            [_accuracy_of_kind(sample, "positive") for sample in resampled]
        ),
        off_topic_accuracy=_accuracy_of_kind(measurements, "off_topic"),
        recall_at_5=_recall(measurements),
        recall_interval=interval([_recall(sample) for sample in resampled]),
        oracle_tiou=_oracle(measurements),
        oracle_interval=interval([_oracle(sample) for sample in resampled]),
        accuracy=_accuracy(measurements),
        accuracy_interval=interval([_accuracy(sample) for sample in resampled]),
        index_seconds=sum(index_times) / len(index_times) if index_times else 0.0,
        worst_index_seconds=max(index_times, default=0.0),
        mean_judging_seconds=sum(one.judging_seconds for one in measurements)
        / len(measurements),
        worst_judging_seconds=max(one.judging_seconds for one in measurements),
    )


def adopt(baseline: ModeReport, candidates: Sequence[ModeReport]) -> ModeReport:
    """Which mode ships, under the gate ADR-0001 fixed before any of this ran.

    A candidate replaces BM25 only if it wins on **both** halves: recall@5
    above the baseline's, and Hard-Negative accuracy whose gain survives
    resampling — its interval's lower bound at or above the baseline's point
    estimate. Anything less and the baseline stands, because ADR-0001 says the
    embedder does not load in the request path unless it earned the memory and
    the latency.

    Ties and near-misses go to the baseline deliberately: the simpler system is
    the one that ships without an argument.
    """
    winners = [
        candidate
        for candidate in candidates
        if candidate.recall_at_5 > baseline.recall_at_5
        and candidate.hard_negative_interval[0] >= baseline.hard_negative_accuracy
    ]

    if not winners:
        return baseline

    return max(winners, key=lambda report: report.hard_negative_interval[0])


def compare(
    fold: FoldName,
    settings: Settings,
    reranker: Reranker,
    rewriter: Rewriter,
    judge: EntailmentJudge,
    embedder: Embedder | None = None,
    modes: Sequence[RetrievalMode] = ("bm25", "dense", "hybrid"),
) -> ModeComparison:
    """Measure every mode on one fold and apply ADR-0001's gate.

    Raises:
        FileNotFoundError: If a Conversation of the fold is not cached.
        ValueError: If the fold holds no Questions, or if BM25 was not among
            the modes measured — it is the baseline the others are adopted
            against, so there is no comparison without it.
    """
    if "bm25" not in modes:
        raise ValueError(
            "BM25 is the baseline ADR-0001 adopts against, so it must be one "
            f"of the modes measured: got {list(modes)}."
        )

    _warm_the_whole_path(fold, settings, reranker, rewriter, judge, embedder)

    reports: list[ModeReport] = []

    for mode in modes:
        measurements, index_times = measure_fold(
            fold, mode, settings, reranker, rewriter, judge, embedder
        )

        if not measurements:
            raise ValueError(
                f"The {fold} fold holds no Questions, so there is nothing to "
                "measure a retrieval mode against."
            )

        reports.append(report_mode(mode, measurements, index_times))

    baseline = next(report for report in reports if report.mode == "bm25")

    return ModeComparison(
        fold=fold,
        baseline=baseline,
        reports=tuple(reports),
        chosen=adopt(
            baseline, [report for report in reports if report is not baseline]
        ),
    )


def _warm_the_whole_path(
    fold: FoldName,
    settings: Settings,
    reranker: Reranker,
    rewriter: Rewriter,
    judge: EntailmentJudge,
    embedder: Embedder | None = None,
) -> None:
    """Answer one Question untimed, so no mode is charged for a first call.

    Warming each model on its own is not enough here. Loading a second model
    into the process leaves a lazy initialisation that lands on whichever
    forward pass runs next, and it was measured at five seconds on the Mac —
    charged, in a run that measures several modes in one process, entirely to
    whichever mode happened to go first. The served process loads only the
    models its own mode needs and never sees it, so it is an artefact of
    measuring three modes together and it is paid here rather than reported.
    """
    transcript_id, rows = conversations_in_fold(fold)[0]
    scheme = ChunkScheme(
        word_lengths=settings.chunk_word_lengths,
        stride_fraction=settings.chunk_stride_fraction,
    )
    chunks = chunk_conversation(transcript_cache.read(transcript_id), scheme)
    question = rows[0]["question"]

    if embedder is not None:
        warm_embedder(embedder, settings)

    reranked = reranker.rerank(
        question,
        Bm25Index(chunks).rank(question, settings.retrieval_candidates),
    )

    if reranked:
        judge.judge(rewriter.rewrite(question).text, [reranked[0].chunk])


def _answered_yes(
    question: str,
    reranked: Sequence[ScoredChunk],
    settings: Settings,
    gate: float | None,
    rewriter: Rewriter,
    judge: EntailmentJudge,
) -> bool:
    """What the shipped Answerer answers, given this mode's ranking.

    The same rule ``RerankEntailAnswerer`` applies, read off the same Settings:
    a Question whose best Chunk does not clear Relevance is a no without the
    Entailment judge being asked, and everything else is a yes exactly when
    that Chunk entails the Claim.
    """
    if not reranked or (gate is not None and reranked[0].relevance < gate):
        return False

    judged = judge.judge(rewriter.rewrite(question).text, [reranked[0].chunk])

    return judged[0].entailment >= settings.entailment_threshold


def _accuracy(measurements: Sequence[QuestionMeasurement]) -> float:
    return rate([one.predicted == one.answer for one in measurements])


def _accuracy_of_kind(
    measurements: Sequence[QuestionMeasurement], question_type: str
) -> float:
    return _accuracy(
        [one for one in measurements if one.question_type == question_type]
    )


def _recall(measurements: Sequence[QuestionMeasurement]) -> float:
    """Fraction of annotated spans the reranked top 5 found."""
    return rate(
        [
            one.ranked_tiou >= HIT_TIOU
            for one in measurements
            if one.ranked_tiou is not None
        ]
    )


def _oracle(measurements: Sequence[QuestionMeasurement]) -> float:
    """Mean best-achievable tIoU over every Chunk — the Chunker's ceiling."""
    tious = [one.oracle_tiou for one in measurements if one.oracle_tiou is not None]

    return sum(tious) / len(tious) if tious else 0.0
