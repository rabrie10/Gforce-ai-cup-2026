"""Choose every decision knob at once, on the competition score.

    python -m scripts.tune
    python -m scripts.tune --fold train --shortlist 40

The Relevance gate, the Entailment threshold and the Entailment depth are swept
jointly against ``0.4 x Accuracy + 0.6 x mean tIoU``, and the winner is the
candidate with the highest lower bound under Conversation-level resampling —
stability rather than argmax, as ADR-0002 chooses every threshold in this
project.

Swept together because they do not decompose. Judging deeper lets a Hard
Negative find a Chunk that entails it, which a higher Entailment threshold then
has to reject; and a Positive rejected by either costs the tIoU half as well as
the accuracy half, which no accuracy-only sweep can price. The values this
prints become the Settings defaults, with the table recorded beside them.

Train and dev only. The test fold is read once, at the end of tuning.
"""

import argparse
import logging

from harness.answer_sweep import (
    SHORTLIST,
    SLICE_FLOOR,
    Candidate,
    CandidateReport,
    chosen,
    grid,
    record_fold,
    sweep,
)
from harness.folds import FoldName, checked_fold_name
from medapp.chunker import SentenceScheme
from medapp.claims import ClaimRewriter
from medapp.config import settings
from medapp.entailment import NliEntailmentJudge
from medapp.reranker import CrossEncoderReranker

READABLE_FOLDS: tuple[FoldName, ...] = ("train", "dev")

# The Relevance gate. None is the ablation: no gate at all, which
# `scripts.entailment_threshold` measured Off-Topic accuracy collapsing under
# and which is kept in the grid so that finding stays visible on this metric
# too rather than being taken on trust.
RELEVANCE_THRESHOLDS: tuple[float | None, ...] = (None, 0.1, 0.2, 0.3, 0.4, 0.5)

# The NLI model's softmax saturates: most pairs score within 1e-3 of zero on
# entailment and what separates a Positive from a Hard Negative sits in that
# tail, so the candidates are spaced logarithmically across it rather than
# evenly across [0, 1], where all but the first would reject everything.
ENTAILMENT_THRESHOLDS: tuple[float, ...] = (
    0.0,
    1e-5,
    5e-5,
    1e-4,
    2e-4,
    0.000394,
    8e-4,
    2e-3,
    5e-3,
    2e-2,
    0.1,
)

# How deep the Entailment judge reads. Bounded by the candidates the Verdict
# carries, since a Chunk that was never carried cannot be judged.
DEPTHS: tuple[int, ...] = (1, 2, 3, 5, 8, 10)


def format_header() -> str:
    """The column names of the sweep table."""
    return (
        f"  {'gate':>6}{'entail':>10}{'depth':>7}"
        f"{'score':>9}{'90% interval':>18}{'acc':>8}{'pos':>7}{'hard':>7}"
        f"{'off':>7}{'tIoU':>8}{'missed':>8}"
    )


def format_report(report: CandidateReport, is_chosen: bool) -> str:
    """One candidate as a table row, with the chosen one marked."""
    candidate = report.candidate
    gate = (
        "none"
        if candidate.relevance_threshold is None
        else f"{candidate.relevance_threshold:.2f}"
    )

    return (
        f"{'*' if is_chosen else ' '} {gate:>6}"
        f"{candidate.entailment_threshold:>10.5f}"
        f"{candidate.entail_depth:>7}{report.score:>9.3f}"
        f"{_format(report.score_interval) if report.resampled else '—':>18}"
        f"{report.accuracy:>8.3f}"
        f"{report.accuracy_by_type.get('positive', 0.0):>7.3f}"
        f"{report.accuracy_by_type.get('hard_negative', 0.0):>7.3f}"
        f"{report.accuracy_by_type.get('off_topic', 0.0):>7.3f}"
        f"{report.mean_tiou:>8.3f}{report.missed_positives:>8}"
    )


def _format(bounds: tuple[float, float]) -> str:
    """One bootstrap interval, as the tables print it."""
    return f"[{bounds[0]:.3f}, {bounds[1]:.3f}]"


# What the system in service actually measured, per Question type, on the fold
# it was chosen on. The slice floor is anchored to these numbers rather than to
# whatever Settings currently hold — a floor that moved with the defaults would
# ratchet, each run measuring itself against the last one's regression and
# permitting another until a type walked down to nothing in acceptable-looking
# steps — and rather than to a Candidate replayed over the current recording,
# which stops naming the system it was taken from the moment anything upstream
# of the knobs changes. Replaying the pre-tuning knobs over a new Chunker
# measures the new Chunker.
#
# These are the overlapping-word-ladder Chunker's numbers over train and dev
# together — 103 of 122 Positives, 73 of 85 Hard Negatives, 33 of 33 Off-Topic
# — which is the system the sentence Chunker replaced, measured on the folds
# the sweep reads. Measured on the same folds deliberately: train is the harder
# of the two and a reference taken on dev alone sets a floor no candidate
# scored over both folds can clear.
REFERENCE_ACCURACY: dict[str, float] = {
    "positive": 0.844,
    "hard_negative": 0.859,
    "off_topic": 1.000,
}


def shipped_candidate() -> Candidate:
    """The knobs as Settings currently resolve them, for the table to beat."""
    return Candidate(
        relevance_threshold=(
            settings.relevance_threshold if settings.relevance_gate else None
        ),
        entailment_threshold=settings.entailment_threshold,
        entail_depth=settings.entail_depth,
    )


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Which fold to sweep and how deep the resampled shortlist runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--folds",
        default="train,dev",
        help=(
            "Which folds to sweep over, comma separated. Both readable folds "
            "by default: 24 Conversations separate candidates the 16 of dev "
            "alone cannot, and the test fold is reserved either way."
        ),
    )
    parser.add_argument(
        "--shortlist",
        type=int,
        default=SHORTLIST,
        help="How many of the best point estimates are measured under resampling.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=25,
        help="How many rows of the table to print.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Sweep the readable folds and print the table.

    Raises:
        KeyError: If a fold name does not name a fold.
    """
    logging.basicConfig(level=logging.INFO)
    arguments = parse_arguments(argv)
    folds = [checked_fold_name(name) for name in arguments.folds.split(",")]
    unreadable = [fold for fold in folds if fold not in READABLE_FOLDS]

    if unreadable:
        print(
            f"The {', '.join(unreadable)} fold is not tuned on. Score it once, "
            "at the end of tuning, with python -m scripts.score --fold test "
            "--release-the-test-fold."
        )
        return 1

    reranker = CrossEncoderReranker(settings)
    reranker.warm_up()
    judge = NliEntailmentJudge(settings)
    judge.warm_up()

    depth = min(max(DEPTHS), settings.retrieval_candidates)
    recordings = tuple(
        recording
        for fold in folds
        for recording in record_fold(
            fold=fold,
            scheme=SentenceScheme(max_words=settings.chunk_max_words),
            reranker=reranker,
            rewriter=ClaimRewriter(),
            judge=judge,
            candidates=settings.retrieval_candidates,
            depth=depth,
        )
    )

    reports = sweep(
        recordings,
        tuple(
            dict.fromkeys(
                grid(
                    relevance_thresholds=RELEVANCE_THRESHOLDS,
                    entailment_thresholds=ENTAILMENT_THRESHOLDS,
                    depths=tuple(
                        candidate for candidate in DEPTHS if candidate <= depth
                    ),
                )
                + (shipped_candidate(),)
            )
        ),
        shortlist=arguments.shortlist,
    )
    winner = chosen(reports, reference=REFERENCE_ACCURACY)

    print(
        f"{'+'.join(folds)}: "
        f"{len({r.transcript_id for r in recordings})} Conversations, "
        f"{len(recordings)} Questions, {len(reports)} candidates, "
        f"{arguments.shortlist} resampled"
    )
    print(format_header())

    for report in reports[: arguments.rows]:
        print(format_report(report, report is winner))

    print()
    print("  chosen: highest lower bound that regresses no Question type")
    print(format_header())
    print(format_report(winner, True))

    print()
    print(
        "  slice floor: no type may fall more than "
        f"{SLICE_FLOOR:g} below "
        + ", ".join(
            f"{question_type} {accuracy:.3f}"
            for question_type, accuracy in REFERENCE_ACCURACY.items()
        )
    )
    print("  shipped (Settings as they stand)")
    print(format_header())
    print(
        format_report(
            next(
                report for report in reports if report.candidate == shipped_candidate()
            ),
            False,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
