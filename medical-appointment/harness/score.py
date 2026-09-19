"""The competition's own score, computed on a fold.

Every other harness module measures a *component*: recall@k, Relevance,
Hard-Negative accuracy, mean tIoU over already-retrieved spans. None of them is
what the attempt is graded on, which is

    0.4 x Accuracy + 0.6 x mean tIoU

and the difference is not presentational. The tIoU average is taken over every
annotated Positive whether it was answered yes or not, so a Positive the
Answerer rejects scores zero on *both* halves: its marginal cost is
``0.4/n + 0.6 * (the tIoU it would have earned) / n``, against ``0.4/n`` for a
Hard Negative wrongly accepted. Rejecting a Positive is roughly twice as
expensive as accepting a Hard Negative, and a threshold swept on accuracy alone
cannot see that — it prices the two the same and lands too tight.

So this module is the objective the thresholds are chosen against, and the
component sweeps become diagnostics that explain a movement in it rather than
targets of their own. Resampling is at Conversation level for the reason
`harness.bootstrap` records.

Accuracy here is over every Question of the fold, and mean tIoU over its
annotated Positives, exactly as `local_evaluator.py` computes both. What is not
modelled is a failed request: the guard in `medapp.service` answers every
Question either way, and a timeout is a latency measurement rather than a
scoring one.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from harness import transcript_cache
from harness.bootstrap import interval, rate, resamples
from harness.folds import FoldName, conversations_in_fold
from local_evaluator import ACCURACY_WEIGHT, TIOU_WEIGHT
from medapp.answerer import Answerer
from medapp.types import Verdict
from utils import Span, gold_evidence, temporal_iou

QUESTION_TYPES: tuple[str, ...] = ("positive", "hard_negative", "off_topic")


@dataclass(frozen=True, slots=True)
class QuestionOutcome:
    """What the Answerer made of one Question, and what it scored.

    Attributes:
        gold: The annotated Evidence Span, or None where the true answer is no
            and there is nothing to point at.
        tiou: The temporal IoU this Question contributes. Only annotated
            Positives carry one: a Question whose true answer is no is left out
            of the tIoU average entirely, so a span volunteered beside a no can
            neither help nor hurt.
    """

    question_id: str
    transcript_id: str
    question_type: str
    truth: bool
    answer: bool
    gold: Span | None
    predicted: Span | None
    tiou: float | None

    @property
    def correct(self) -> bool:
        """Whether the answer half scored this Question."""
        return self.answer == self.truth


@dataclass(frozen=True, slots=True)
class ScoreReport:
    """One configuration's score on one fold, with the halves that make it up.

    Attributes:
        score: The competition score, ``0.4 x accuracy + 0.6 x mean_tiou``.
        score_interval: The bootstrap interval on it. This is the number a
            configuration is chosen by — its lower bound, per ADR-0002.
        mean_tiou: Over every annotated Positive, answered yes or not.
        mean_tiou_answered_yes: Over only the Positives answered yes. A
            diagnostic, not part of the score: read beside ``mean_tiou`` it
            separates "localizes badly" from "does not notice".
        missed_positives: Annotated Positives answered no. Each costs both
            halves, which is the asymmetry this module exists to price.
        accuracy_by_type: Accuracy within each of the three Question types.
    """

    fold: FoldName
    conversations: int
    questions: int
    positives: int
    score: float
    score_interval: tuple[float, float]
    accuracy: float
    accuracy_interval: tuple[float, float]
    mean_tiou: float
    mean_tiou_interval: tuple[float, float]
    mean_tiou_answered_yes: float
    missed_positives: int
    spans_not_returned: int
    accuracy_by_type: dict[str, tuple[float, int, int]]


def outcomes_for_fold(
    fold: FoldName, answerer: Answerer
) -> tuple[QuestionOutcome, ...]:
    """Answer every Question of one fold, as one request would answer ten.

    The Chunks, the index and every judge are read once per Conversation and
    all ten of its Questions are answered against them, which is what happens
    inside one request. Questions are answered in CSV order so that a lazily
    produced Verdict lines up with the row it belongs to.

    Args:
        fold: Which fold to read.
        answerer: The Answerer as it is served — including the
            :class:`~medapp.answerer.SpanRefiningAnswerer` wrapper, since the
            refined span is what the evaluator would be sent.

    Raises:
        FileNotFoundError: If a Conversation of the fold is not cached.
    """
    outcomes: list[QuestionOutcome] = []

    for transcript_id, rows in conversations_in_fold(fold):
        segments = transcript_cache.read(transcript_id)
        verdicts = answerer.answer(segments, [row["question"] for row in rows])

        for row, verdict in zip(rows, verdicts, strict=True):
            outcomes.append(_outcome(transcript_id, row, verdict))

    return tuple(outcomes)


def _outcome(
    transcript_id: str, row: dict[str, str], verdict: Verdict
) -> QuestionOutcome:
    """One row scored against the Verdict it was answered with."""
    gold = gold_evidence(row)
    answer = verdict.answer
    predicted = verdict.evidence if answer else None

    return QuestionOutcome(
        question_id=row["question_id"],
        transcript_id=transcript_id,
        question_type=row["question_type"],
        truth=row["label"] == "1",
        answer=answer,
        gold=gold,
        predicted=predicted,
        # Only an annotated Positive is averaged over. A no is left out rather
        # than scored zero, which is what keeps a volunteered span harmless.
        tiou=(temporal_iou(gold, predicted) if gold is not None else None),
    )


def score(outcomes: Sequence[QuestionOutcome]) -> float:
    """The competition score over a set of outcomes."""
    return ACCURACY_WEIGHT * accuracy(outcomes) + TIOU_WEIGHT * mean_tiou(outcomes)


def accuracy(outcomes: Sequence[QuestionOutcome]) -> float:
    """The answer half: one point per Question, no partial credit."""
    return rate(outcome.correct for outcome in outcomes)


def mean_tiou(outcomes: Sequence[QuestionOutcome]) -> float:
    """The evidence half, over every annotated Positive whether answered or not."""
    scores = [outcome.tiou for outcome in outcomes if outcome.tiou is not None]

    return sum(scores) / len(scores) if scores else 0.0


def report(fold: FoldName, answerer: Answerer) -> ScoreReport:
    """Score one fold end to end, with bootstrap intervals on every half.

    Args:
        fold: Which fold to read.
        answerer: The Answerer as it is served.

    Raises:
        FileNotFoundError: If a Conversation of the fold is not cached.
        ValueError: If the fold holds no Questions.
    """
    outcomes = outcomes_for_fold(fold, answerer)

    if not outcomes:
        raise ValueError(f"The {fold} fold holds no Questions to score.")

    return report_for(fold, outcomes)


def report_for(fold: FoldName, outcomes: Sequence[QuestionOutcome]) -> ScoreReport:
    """Assemble the report from outcomes already measured.

    Separate from :func:`report` so a sweep can answer a fold once and score
    many configurations over the recorded result, the way every other sweep in
    this harness does.

    Raises:
        ValueError: If there are no outcomes to score.
    """
    if not outcomes:
        raise ValueError("There are no outcomes to score.")

    resampled = resamples(outcomes)
    positives = [outcome for outcome in outcomes if outcome.tiou is not None]
    # Narrowed to float here rather than at the sum: every Positive carries a
    # tIoU by construction, and mypy cannot see that from the Optional field.
    answered = [
        outcome.tiou
        for outcome in positives
        if outcome.answer and outcome.tiou is not None
    ]

    return ScoreReport(
        fold=fold,
        conversations=len({outcome.transcript_id for outcome in outcomes}),
        questions=len(outcomes),
        positives=len(positives),
        score=score(outcomes),
        score_interval=interval([score(resample) for resample in resampled]),
        accuracy=accuracy(outcomes),
        accuracy_interval=interval([accuracy(resample) for resample in resampled]),
        mean_tiou=mean_tiou(outcomes),
        mean_tiou_interval=interval([mean_tiou(resample) for resample in resampled]),
        mean_tiou_answered_yes=(sum(answered) / len(answered) if answered else 0.0),
        missed_positives=sum(1 for outcome in positives if not outcome.answer),
        spans_not_returned=sum(1 for outcome in positives if outcome.predicted is None),
        accuracy_by_type={
            question_type: _accuracy_within(outcomes, question_type)
            for question_type in QUESTION_TYPES
        },
    )


def _accuracy_within(
    outcomes: Sequence[QuestionOutcome], question_type: str
) -> tuple[float, int, int]:
    """Accuracy within one Question type, as a rate over a count.

    Reported per type because the three fail differently and an aggregate
    hides which one moved: an Off-Topic Question is rejected by Relevance, a
    Hard Negative by Entailment, and a Positive is kept by neither rejecting
    it.
    """
    within = [outcome for outcome in outcomes if outcome.question_type == question_type]
    correct = sum(1 for outcome in within if outcome.correct)

    return (rate(outcome.correct for outcome in within), correct, len(within))
