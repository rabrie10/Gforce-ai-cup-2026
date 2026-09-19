"""Choose the Relevance threshold on the dev fold, and show the work.

Every candidate is printed with Off-Topic accuracy, Positive TPR and overall
TNR, so that a threshold buying its accuracy from one Question type is visible
rather than hidden inside a blended number. The chosen value is the one with
the highest lower bound on *Off-Topic* accuracy under bootstrap resampling at
Conversation level — the component metric ADR-0002 assigns this threshold, and
stability rather than argmax — with ties going to the candidate that keeps the
most Positives.

    python -m scripts.relevance_threshold
    python -m scripts.relevance_threshold --fold train --thresholds 0.1,0.5,0.9

The value it prints becomes ``MEDAPP_RELEVANCE_THRESHOLD``'s default in
Settings, with these numbers recorded beside it.

Train and dev only. The test fold is transcribed but not read until tuning ends.
"""

import argparse

from harness.folds import FoldName
from harness.relevance_threshold import (
    JUDGING_BUDGET_SECONDS,
    QUESTIONS_PER_CONVERSATION,
    ThresholdReport,
    ThresholdSweep,
    sweep,
)
from medapp.chunker import SentenceScheme
from medapp.config import settings
from medapp.reranker import CrossEncoderReranker

READABLE_FOLDS: tuple[FoldName, ...] = ("train", "dev")


def format_header() -> str:
    """The column names of the sweep table."""
    return (
        f"  {'threshold':>11}{'off-topic':>11}{'90% interval':>18}"
        f"{'hard-neg':>10}{'TPR':>8}{'TNR':>8}{'accuracy':>10}{'90% interval':>18}"
    )


def format_report(report: ThresholdReport, chosen: bool) -> str:
    """One candidate threshold as a table row, with the chosen one marked."""
    return (
        f"{'*' if chosen else ' '} {report.threshold:>11.6f}"
        f"{report.off_topic_accuracy:>11.3f}{_format(report.off_topic_interval):>18}"
        f"{report.hard_negative_accuracy:>10.3f}"
        f"{report.true_positive_rate:>8.3f}{report.true_negative_rate:>8.3f}"
        f"{report.accuracy:>10.3f}{_format(report.accuracy_interval):>18}"
    )


def _format(interval: tuple[float, float]) -> str:
    """One bootstrap interval as a table cell."""
    low, high = interval

    return f"[{low:.3f}, {high:.3f}]"


def format_sweep(measurement: ThresholdSweep) -> str:
    """One fold's sweep: the table, the chosen threshold and the judging time."""
    ten_questions = measurement.worst_judging_seconds * QUESTIONS_PER_CONVERSATION
    lines = [
        f"{measurement.fold}: {measurement.conversations} Conversations, "
        f"{measurement.questions} Questions",
        format_header(),
    ]
    lines += [
        format_report(report, report is measurement.chosen)
        for report in measurement.reports
    ]
    lines += [
        "",
        f"  chosen threshold {measurement.chosen.threshold:.6f} "
        f"(accuracy {measurement.chosen.accuracy:.3f}, off-topic "
        f"{measurement.chosen.off_topic_accuracy:.3f}, TPR "
        f"{measurement.chosen.true_positive_rate:.3f}, TNR "
        f"{measurement.chosen.true_negative_rate:.3f})",
        f"  judging {measurement.mean_judging_seconds * 1000:.0f} ms per Question "
        f"on average, {measurement.worst_judging_seconds * 1000:.0f} ms worst; "
        f"{QUESTIONS_PER_CONVERSATION} at the worst is {ten_questions:.1f} s "
        f"against a {JUDGING_BUDGET_SECONDS:g} s budget"
        f"{'' if ten_questions <= JUDGING_BUDGET_SECONDS else ' — over'}",
    ]

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold",
        choices=READABLE_FOLDS,
        default="dev",
        help="Which fold to sweep. Thresholds are chosen on dev.",
    )
    parser.add_argument(
        "--thresholds",
        help="Comma-separated candidates. Defaults to quantiles of the "
        "observed Relevance.",
    )
    arguments = parser.parse_args()

    print(
        f"Reranking the top {settings.retrieval_candidates} BM25 Chunks with "
        f"{settings.rerank_model} on {settings.device}."
    )

    # Warmed up before the clock starts, exactly as the served process warms it
    # at import: charging the first Question for the first forward pass would
    # report a per-Question time no request ever pays.
    reranker = CrossEncoderReranker(settings)
    reranker.warm_up()

    print(
        format_sweep(
            sweep(
                fold=arguments.fold,
                scheme=SentenceScheme(max_words=settings.chunk_max_words),
                reranker=reranker,
                candidates=settings.retrieval_candidates,
                thresholds=(
                    tuple(
                        float(threshold)
                        for threshold in arguments.thresholds.split(",")
                    )
                    if arguments.thresholds
                    else None
                ),
            )
        )
    )

    print(
        "\nChosen for the highest lower bound on Off-Topic accuracy under "
        "resampling, not by argmax\nand not on blended accuracy.\n"
        "Hard-Negative accuracy is reported, not tuned on: rejecting Hard "
        "Negatives is the\nEntailment judge's job, and a threshold pulled up "
        "to chase them pays for it in Positives.\n"
        "The test fold is transcribed but not read until tuning ends."
    )


if __name__ == "__main__":
    main()
