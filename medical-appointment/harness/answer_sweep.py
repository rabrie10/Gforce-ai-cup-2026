"""Every decision knob swept jointly, against the competition score.

The four knobs that decide a Verdict — the Relevance gate, the Entailment
threshold, how deep the Entailment judge reads and the two span pads — were
each chosen against a different proxy: Off-Topic accuracy, Hard-Negative
accuracy, nothing at all, and mean tIoU over already-retrieved spans. They do
not decompose. Raising the depth accepts Hard Negatives that a higher
Entailment threshold would then reject; rejecting a Positive costs the tIoU
half as well as the accuracy half, which neither accuracy proxy can see. So
they are swept together, on the one number the attempt is graded on.

The expensive half runs once. For every Question of the fold this module
records what the retriever ranked, what the reranker scored it, and what the
Entailment judge made of the top ``depth`` Chunks — unconditionally, including
for the Questions a Relevance gate would have rejected, so that a lower gate
can still be scored over the same recording. Everything after that is
arithmetic: choosing a knob re-reads the recording rather than re-running a
model, which is the pattern `harness.span_padding` and
`harness.entailment_threshold` already use for one knob each.

What the recording fixes, and what therefore cannot be swept here, is the
retrieval mode and the Chunk scheme: both change which Chunks exist. Those are
measured by re-recording, which is what `scripts.retrieval_modes` does.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import product

from harness import transcript_cache
from harness.bootstrap import interval, rate, resamples
from harness.folds import FoldName, conversations_in_fold
from harness.score import QuestionOutcome, score
from medapp.chunker import ChunkScheme, chunk_conversation
from medapp.claims import Rewriter
from medapp.entailment import EntailmentJudge
from medapp.reranker import Reranker
from medapp.retrieval import Retriever
from medapp.span_refiner import SpanPadding, WordEdges, refine_span_against
from medapp.types import Chunk
from utils import Span, gold_evidence, temporal_iou


@dataclass(frozen=True, slots=True)
class Recording:
    """Everything one Question's Verdict could be recomputed from.

    Attributes:
        relevance: The top Chunk's Relevance, which the gate is applied to.
            Zero where nothing was retrieved at all.
        chunks: The reranked candidates, best Relevance first.
        entailments: The entailment probability of each of the first
            ``len(entailments)`` chunks, judged unconditionally so that a gate
            lower than the one in force when this was recorded can still be
            scored.
        edges: The Conversation's sorted Word boundaries, which a pad is
            snapped back to. Sorted once here rather than inside each of the
            tens of thousands of refinements the sweep performs against them.
    """

    question_id: str
    transcript_id: str
    question_type: str
    truth: bool
    gold: Span | None
    relevance: float
    chunks: tuple[Chunk, ...]
    entailments: tuple[float, ...]
    edges: WordEdges


@dataclass(frozen=True, slots=True)
class Candidate:
    """One setting of the four knobs."""

    relevance_threshold: float | None
    entailment_threshold: float
    entail_depth: int
    padding: SpanPadding


@dataclass(frozen=True, slots=True)
class CandidateReport:
    """One candidate's score on the recorded fold, with its interval.

    Attributes:
        accuracy_by_type: Accuracy within each Question type. Carried because
            the aggregate can rise while a type collapses — a candidate that
            buys tIoU by accepting every Hard Negative scores well and is not
            the system to ship.
        resampled: Whether ``score_interval`` is a bootstrap interval or the
            point estimate repeated, which is what a candidate outside the
            shortlist carries.
    """

    candidate: Candidate
    score: float
    score_interval: tuple[float, float]
    accuracy: float
    accuracy_by_type: dict[str, float]
    mean_tiou: float
    missed_positives: int
    resampled: bool


def record_fold(
    fold: FoldName,
    scheme: ChunkScheme,
    index_factory,
    reranker: Reranker,
    rewriter: Rewriter,
    judge: EntailmentJudge,
    candidates: int,
    depth: int,
) -> tuple[Recording, ...]:
    """Run the expensive half once over one fold.

    The Chunks and the index are built once per Conversation and every Question
    of it is answered against them, which is what happens inside one request.

    Args:
        fold: Which fold to read.
        scheme: The Chunk granularities, as Settings resolved them.
        index_factory: Builds the per-request index over a Conversation's
            Chunks.
        reranker: Scores Relevance.
        rewriter: Turns a Question into the Claim a yes would agree with.
        judge: Decides Entailment.
        candidates: How many ranked Chunks are rescored per Question.
        depth: How many of them are judged. The deepest candidate any later
            sweep may ask for — a sweep cannot read deeper than was recorded.

    Raises:
        FileNotFoundError: If a Conversation of the fold is not cached.
    """
    recordings: list[Recording] = []

    for transcript_id, rows in conversations_in_fold(fold):
        segments = transcript_cache.read(transcript_id)
        edges = WordEdges.of([word for segment in segments for word in segment.words])
        index: Retriever = index_factory(chunk_conversation(segments, scheme))

        for row in rows:
            question = row["question"]
            reranked = reranker.rerank(question, index.rank(question, candidates))
            chunks = tuple(scored.chunk for scored in reranked)
            judged = judge.judge(rewriter.rewrite(question).text, chunks[:depth])

            recordings.append(
                Recording(
                    question_id=row["question_id"],
                    transcript_id=transcript_id,
                    question_type=row["question_type"],
                    truth=row["label"] == "1",
                    gold=gold_evidence(row),
                    relevance=reranked[0].relevance if reranked else 0.0,
                    chunks=chunks,
                    entailments=tuple(judgement.entailment for judgement in judged),
                    edges=edges,
                )
            )

    return tuple(recordings)


def outcome(recording: Recording, candidate: Candidate) -> QuestionOutcome:
    """What one Question would have scored under one setting of the knobs.

    This is `medapp.answerer.RerankEntailAnswerer` and
    `medapp.answerer.SpanRefiningAnswerer` replayed over the recording, and it
    has to stay that: a rule that drifts from the shipped Answerer chooses a
    knob for a system that is not the one served.
    """
    answer = False
    predicted: Span | None = None

    gated = (
        candidate.relevance_threshold is None
        or recording.relevance >= candidate.relevance_threshold
    )
    judged = recording.entailments[: candidate.entail_depth]

    if recording.chunks and gated and judged:
        best = max(range(len(judged)), key=lambda index: judged[index])

        if judged[best] >= candidate.entailment_threshold:
            answer = True
            predicted = refine_span_against(
                recording.chunks[best], recording.edges, candidate.padding
            )

    return QuestionOutcome(
        question_id=recording.question_id,
        transcript_id=recording.transcript_id,
        question_type=recording.question_type,
        truth=recording.truth,
        answer=answer,
        gold=recording.gold,
        predicted=predicted,
        tiou=(
            temporal_iou(recording.gold, predicted)
            if recording.gold is not None
            else None
        ),
    )


def outcomes(
    recordings: Sequence[Recording], candidate: Candidate
) -> tuple[QuestionOutcome, ...]:
    """Replay a whole fold under one setting of the knobs."""
    return tuple(outcome(recording, candidate) for recording in recordings)


# How many of the grid's best point estimates are then measured under
# resampling. A joint grid runs to thousands of candidates and bootstrapping
# every one of them costs hours to separate candidates that the point estimate
# has already put far apart; a shortlist this deep covers everything within
# noise of the best, which is exactly the set the lower bound has to decide
# between.
SHORTLIST = 25

# Lower bounds within this of the best are treated as tied. A bootstrap lower
# bound read off 2000 resamples of 16 Conversations carries far more Monte
# Carlo noise than its third decimal, so separating two candidates by 0.0001 of
# it is choosing on the draw. The band is wide enough to swallow that jitter
# and narrow enough not to swallow a real difference, which on this fold is
# worth a hundredth or more.
TIE_TOLERANCE = 0.005


def sweep(
    recordings: Sequence[Recording],
    candidates: Sequence[Candidate],
    shortlist: int = SHORTLIST,
) -> tuple[CandidateReport, ...]:
    """Score every candidate over one recording, best first.

    Two passes. Every candidate is scored on the fold itself, which is cheap;
    only the ``shortlist`` best are then scored on each resampled fold, which
    is not. A candidate outside the shortlist carries a degenerate interval at
    its own point estimate — it was never in contention, and reporting an
    interval it did not earn would be worse than reporting none.

    The resampled folds are drawn once and shared, so two candidates differ by
    the candidate rather than by the draw.

    Raises:
        ValueError: If there is nothing recorded to sweep over.
    """
    if not recordings:
        raise ValueError("There is nothing recorded to sweep over.")

    ranked = sorted(
        (_point_report(candidate, recordings) for candidate in candidates),
        key=lambda report: report.score,
        reverse=True,
    )
    resampled = resamples(recordings)

    return tuple(
        (
            _with_interval(report, recordings, resampled)
            if position < shortlist
            else report
        )
        for position, report in enumerate(ranked)
    )


def _point_report(
    candidate: Candidate, recordings: Sequence[Recording]
) -> CandidateReport:
    """One candidate scored on the fold itself, with no interval yet."""
    scored = outcomes(recordings, candidate)
    positives = [outcome for outcome in scored if outcome.tiou is not None]
    point = score(scored)

    return CandidateReport(
        candidate=candidate,
        score=point,
        score_interval=(point, point),
        accuracy=sum(outcome.correct for outcome in scored) / len(scored),
        accuracy_by_type=_by_type(scored),
        mean_tiou=(
            sum(outcome.tiou or 0.0 for outcome in positives) / len(positives)
            if positives
            else 0.0
        ),
        missed_positives=sum(1 for outcome in positives if not outcome.answer),
        resampled=False,
    )


def _by_type(outcomes: Sequence[QuestionOutcome]) -> dict[str, float]:
    """Accuracy within each Question type present in the outcomes."""
    types = {outcome.question_type for outcome in outcomes}

    return {
        question_type: rate(
            outcome.correct
            for outcome in outcomes
            if outcome.question_type == question_type
        )
        for question_type in sorted(types)
    }


def _with_interval(
    report: CandidateReport,
    recordings: Sequence[Recording],
    resampled: Sequence[Sequence[Recording]],
) -> CandidateReport:
    """The same candidate, with the interval its shortlisting earned it."""
    return replace(
        report,
        score_interval=interval(
            [score(outcomes(resample, report.candidate)) for resample in resampled]
        ),
        resampled=True,
    )


# How far a Question type's accuracy may fall below the reference
# configuration's before a candidate is refused, whatever it scores overall.
# Slice-based evaluation is not optional here: the three types fail for three
# different reasons and the aggregate hides which one moved, so a candidate
# that buys mean tIoU by accepting Hard Negatives can gain on the score while
# getting worse at the discrimination the case is actually about. The
# evaluation set is a different draw of Conversations, and a system leaning on
# padding gains rather than on judgement travels worse.
SLICE_FLOOR = 0.03


def chosen(
    reports: Sequence[CandidateReport],
    reference: CandidateReport | None = None,
    slice_floor: float = SLICE_FLOOR,
) -> CandidateReport:
    """The candidate with the highest lower bound on the score under resampling.

    Stability rather than argmax, as ADR-0002 chooses every threshold: the
    folds are small enough that the best point estimate is often the luckiest
    draw rather than the best configuration.

    Only the shortlisted candidates are eligible. One outside it carries no
    measured interval, so choosing it would be choosing on the point estimate
    this function exists not to choose on.

    Lower bounds within :data:`TIE_TOLERANCE` of the best count as tied, and
    the tie is broken on missed Positives, fewest first. The grid runs to
    thousands of candidates over 16 Conversations, so the best lower bound to
    four decimals is the luckiest draw rather than the best configuration, and
    a rule that reads it that finely is the overfitting the lower bound was
    adopted to avoid. A missed Positive is the right tie-break because it is
    the failure this fold measures least well: it is the only one that costs
    both halves, and the evaluation set is a different draw of Conversations,
    not of Questions. Between configurations the fold cannot tell apart, the
    one that rejects fewer Positives degrades more gently off this
    distribution.

    Args:
        reports: Every candidate scored, as :func:`sweep` returns them.
        reference: The configuration a candidate must not regress a Question
            type against — the shipped one, normally. None applies no slice
            floor, which is the ablation rather than the default.
        slice_floor: How far below the reference a type's accuracy may fall.

    Raises:
        ValueError: If no candidate was measured under resampling, or none of
            those measured clears the slice floor.
    """
    measured = [report for report in reports if report.resampled]

    if not measured:
        raise ValueError(
            "No candidate was measured under resampling, so there is no lower "
            "bound to choose on."
        )

    eligible = [
        report
        for report in measured
        if _holds_every_slice(report, reference, slice_floor)
    ]

    if not eligible:
        raise ValueError(
            "Every candidate measured under resampling regresses a Question "
            f"type by more than {slice_floor:g} against the reference. Widen "
            "the shortlist, or accept the regression deliberately by passing "
            "no reference."
        )

    measured = eligible
    best = max(report.score_interval[0] for report in measured)
    tied = [
        report
        for report in measured
        if report.score_interval[0] >= best - TIE_TOLERANCE
    ]

    return max(tied, key=lambda report: (-report.missed_positives, report.score))


def grid(
    relevance_thresholds: Sequence[float | None],
    entailment_thresholds: Sequence[float],
    depths: Sequence[int],
    paddings: Sequence[SpanPadding],
) -> tuple[Candidate, ...]:
    """Every combination of the knobs, as candidates to sweep."""
    return tuple(
        Candidate(
            relevance_threshold=relevance,
            entailment_threshold=entailment,
            entail_depth=depth,
            padding=padding,
        )
        for relevance, entailment, depth, padding in product(
            relevance_thresholds, entailment_thresholds, depths, paddings
        )
    )


def paddings(
    fixed: Sequence[float], fractions: Sequence[float]
) -> tuple[SpanPadding, ...]:
    """Every combination of fixed and proportional padding, on both ends.

    Both ends are swept independently: the Chunker undershoots the annotated
    span's start more often than its end, so a padding shared between the two
    cannot express what the data asks for.
    """
    return tuple(
        SpanPadding(
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            start_fraction=start_fraction,
            end_fraction=end_fraction,
        )
        for start_seconds, end_seconds, start_fraction, end_fraction in product(
            fixed, fixed, fractions, fractions
        )
    )


def two_pass_sweep(
    recordings: Sequence[Recording],
    decisions: Sequence[Candidate],
    paddings: Sequence[SpanPadding],
    finalists: int,
    shortlist: int = SHORTLIST,
) -> tuple[CandidateReport, ...]:
    """Sweep the decision knobs first, then the padding over the best of them.

    The full product of every knob runs past a million candidates, and most of
    them are not worth scoring. Padding is very nearly separable from the rest:
    given which Chunk a Verdict cites, moving that Chunk's boundaries changes
    no answer, so it cannot reorder the decision knobs by more than the scale
    factor it applies to the tIoU half. Nearly, not exactly — which is why the
    padding grid is swept over the ``finalists`` best decision settings rather
    than over the single best one, and why the bootstrap that actually chooses
    runs over the shortlist of the combined result.

    Args:
        recordings: The fold, recorded once.
        decisions: Decision knobs to rank, each carrying whatever padding the
            first pass should hold fixed.
        paddings: The padding grid, swept over the finalists.
        finalists: How many decision settings reach the second pass.
        shortlist: How many of the combined best are measured under resampling.

    Returns:
        Every candidate scored, best first, as :func:`sweep` returns them.

    Raises:
        ValueError: If there is nothing recorded, or no decision to rank.
    """
    if not decisions:
        raise ValueError("There are no decision knobs to rank.")

    ranked = sweep(recordings, decisions, shortlist=0)
    best = [report.candidate for report in ranked[:finalists]]

    # Deduplicated: the padding held fixed in the first pass is itself in the
    # grid, so every finalist would otherwise be scored twice and fill the
    # table with its own duplicates.
    combined = dict.fromkeys(
        tuple(decisions)
        + tuple(
            replace(decision, padding=padding)
            for decision in best
            for padding in paddings
        )
    )

    return sweep(recordings, tuple(combined), shortlist=shortlist)


def _holds_every_slice(
    report: CandidateReport,
    reference: CandidateReport | None,
    slice_floor: float,
) -> bool:
    """Whether this candidate regresses no Question type past the floor."""
    if reference is None:
        return True

    return all(
        report.accuracy_by_type.get(question_type, 0.0) >= accuracy - slice_floor
        for question_type, accuracy in reference.accuracy_by_type.items()
    )
