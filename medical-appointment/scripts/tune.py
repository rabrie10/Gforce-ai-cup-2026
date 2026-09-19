"""Choose every decision knob at once, on the competition score.

    python -m scripts.tune
    python -m scripts.tune --fold train --shortlist 40

The Relevance gate, the Entailment threshold, the Entailment depth and the two
span pads are swept jointly against ``0.4 x Accuracy + 0.6 x mean tIoU``, and
the winner is the candidate with the highest lower bound under Conversation-
level resampling — stability rather than argmax, as ADR-0002 chooses every
threshold in this project.

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
    Candidate,
    CandidateReport,
    chosen,
    grid,
    paddings,
    record_fold,
    two_pass_sweep,
)
from harness.folds import FoldName, checked_fold_name
from medapp.answerer import embedder_for
from medapp.chunker import ChunkScheme
from medapp.claims import ClaimRewriter
from medapp.config import settings
from medapp.entailment import NliEntailmentJudge
from medapp.reranker import CrossEncoderReranker
from medapp.retrieval import build_index_factory
from medapp.span_refiner import SpanPadding

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

# How deep the Entailment judge reads. Bounded by the candidates the retriever
# carries, since a Chunk that was never ranked cannot be judged.
DEPTHS: tuple[int, ...] = (1, 2, 3, 5, 8, 10)

# Fixed offsets in seconds from the cited Chunk's own boundary, in both
# directions, and further offsets as a fraction of the Chunk's own duration.
# The fixed candidates reach past 0.75 because the sweep that stopped there
# pinned the start pad at its own top rung, which says the optimum was outside
# the range rather than inside it.
PAD_SECONDS: tuple[float, ...] = (-0.3, -0.15, 0.0, 0.15, 0.3, 0.5, 0.75, 1.0, 1.5)
PAD_FRACTIONS: tuple[float, ...] = (-0.2, -0.1, 0.0, 0.1, 0.2, 0.4)


def format_header() -> str:
    """The column names of the sweep table."""
    return (
        f"  {'gate':>6}{'entail':>10}{'depth':>7}"
        f"{'start s':>9}{'end s':>8}{'start f':>9}{'end f':>8}"
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

    padding = candidate.padding

    return (
        f"{'*' if is_chosen else ' '} {gate:>6}"
        f"{candidate.entailment_threshold:>10.5f}"
        f"{candidate.entail_depth:>7}{padding.start_seconds:>9.2f}"
        f"{padding.end_seconds:>8.2f}{padding.start_fraction:>9.2f}"
        f"{padding.end_fraction:>8.2f}{report.score:>9.3f}"
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


# The configuration the separate component sweeps of tickets 08, 09 and 11
# arrived at, before anything was tuned against the competition score. The
# slice floor is anchored here rather than to whatever Settings currently hold:
# a floor that moved with the defaults would ratchet — each run would measure
# itself against the last one's regression and permit another, and a Question
# type could walk down to nothing in steps that each looked acceptable.
BASELINE = Candidate(
    relevance_threshold=0.3,
    entailment_threshold=0.000394,
    entail_depth=1,
    padding=SpanPadding(start_seconds=0.75, end_seconds=0.15),
)


def shipped_candidate() -> Candidate:
    """The knobs as Settings currently resolve them, for the table to beat."""
    return Candidate(
        relevance_threshold=(
            settings.relevance_threshold if settings.relevance_gate else None
        ),
        entailment_threshold=settings.entailment_threshold,
        entail_depth=settings.entail_depth,
        padding=SpanPadding(
            start_seconds=settings.span_pad_start_seconds,
            end_seconds=settings.span_pad_end_seconds,
            start_fraction=settings.span_pad_start_fraction,
            end_fraction=settings.span_pad_end_fraction,
        ),
    )


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Which fold to sweep and how deep the resampled shortlist runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold", default="dev", help="Which fold to sweep. Knobs are chosen on dev."
    )
    parser.add_argument(
        "--shortlist",
        type=int,
        default=SHORTLIST,
        help="How many of the best point estimates are measured under resampling.",
    )
    parser.add_argument(
        "--finalists",
        type=int,
        default=6,
        help="How many decision settings the full padding grid is swept over.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=25,
        help="How many rows of the table to print.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Sweep one fold and print the table.

    Raises:
        KeyError: If the fold name does not name a fold.
    """
    logging.basicConfig(level=logging.INFO)
    arguments = parse_arguments(argv)
    fold = checked_fold_name(arguments.fold)

    if fold not in READABLE_FOLDS:
        print(
            f"The {fold} fold is not tuned on. Score it once, at the end of "
            "tuning, with python -m scripts.score --fold test "
            "--release-the-test-fold."
        )
        return 1

    reranker = CrossEncoderReranker(settings)
    reranker.warm_up()
    judge = NliEntailmentJudge(settings)
    judge.warm_up()

    depth = min(max(DEPTHS), settings.retrieval_candidates)
    recordings = record_fold(
        fold=fold,
        scheme=ChunkScheme(
            word_lengths=settings.chunk_word_lengths,
            stride_fraction=settings.chunk_stride_fraction,
        ),
        index_factory=build_index_factory(settings, embedder_for(settings)),
        reranker=reranker,
        rewriter=ClaimRewriter(),
        judge=judge,
        candidates=settings.retrieval_candidates,
        depth=depth,
    )

    reports = two_pass_sweep(
        recordings,
        decisions=grid(
            relevance_thresholds=RELEVANCE_THRESHOLDS,
            entailment_thresholds=ENTAILMENT_THRESHOLDS,
            depths=tuple(candidate for candidate in DEPTHS if candidate <= depth),
            paddings=(shipped_candidate().padding,),
        )
        + (shipped_candidate(), BASELINE),
        paddings=paddings(PAD_SECONDS, PAD_FRACTIONS),
        finalists=arguments.finalists,
        shortlist=arguments.shortlist,
    )
    reference = next(report for report in reports if report.candidate == BASELINE)
    winner = chosen(reports, reference=reference)

    print(
        f"{fold}: {len({r.transcript_id for r in recordings})} Conversations, "
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
    print("  baseline (component sweeps), the slice floor is measured against it")
    print(format_header())
    print(format_report(reference, False))
    print("  shipped (Settings as they stand)")
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
