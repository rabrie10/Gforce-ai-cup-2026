"""Choose the Entailment threshold on the dev fold, ablate the Relevance gate,
and show the work.

Every candidate is printed twice — with the Relevance gate and without it —
carrying Hard-Negative accuracy, Positive accuracy, Off-Topic accuracy and TNR,
so that a threshold buying its accuracy from one Question type is visible
rather than hidden inside a blended number. The chosen value is the one with
the highest lower bound on *Hard-Negative* accuracy under bootstrap resampling
at Conversation level — the component metric ADR-0002 assigns this threshold —
among the candidates that hold Positive accuracy at or above the bottom of its
ticket-08 interval.

    python -m scripts.entailment_threshold
    python -m scripts.entailment_threshold --fold train --thresholds 0.3,0.5,0.9

The value it prints becomes ``MEDAPP_ENTAILMENT_THRESHOLD``'s default in
Settings, and the ablation's verdict becomes ``MEDAPP_RELEVANCE_GATE``'s, with
these numbers recorded beside them.

Train and dev only. The test fold is transcribed but not read until tuning ends.
"""

import argparse

from harness.entailment_threshold import ThresholdReport, ThresholdSweep, sweep
from harness.folds import FoldName
from harness.relevance_threshold import (
    JUDGING_BUDGET_SECONDS,
    QUESTIONS_PER_CONVERSATION,
)
from medapp.chunker import ChunkScheme
from medapp.claims import ClaimRewriter
from medapp.config import settings
from medapp.entailment import NliEntailmentJudge
from medapp.reranker import CrossEncoderReranker

READABLE_FOLDS: tuple[FoldName, ...] = ("train", "dev")


def format_header() -> str:
    """The column names of the sweep table."""
    return (
        f"  {'threshold':>11}{'hard-neg':>10}{'90% interval':>18}"
        f"{'positive':>10}{'off-topic':>11}{'TNR':>8}{'accuracy':>10}"
        f"{'90% interval':>18}"
    )


def format_report(report: ThresholdReport, chosen: bool) -> str:
    """One candidate threshold as a table row, with the chosen one marked."""
    return (
        f"{'*' if chosen else ' '} {report.threshold:>11.6f}"
        f"{report.hard_negative_accuracy:>10.3f}"
        f"{_format(report.hard_negative_interval):>18}"
        f"{report.positive_accuracy:>10.3f}{report.off_topic_accuracy:>11.3f}"
        f"{report.true_negative_rate:>8.3f}{report.accuracy:>10.3f}"
        f"{_format(report.accuracy_interval):>18}"
    )


def _format(interval: tuple[float, float]) -> str:
    """One bootstrap interval as a table cell."""
    low, high = interval

    return f"[{low:.3f}, {high:.3f}]"


def format_sweep(measurement: ThresholdSweep) -> str:
    """One fold's sweep: both arms, the chosen threshold and the judging time."""
    ten_questions = measurement.worst_judging_seconds * QUESTIONS_PER_CONVERSATION
    lines = [
        f"{measurement.fold}: {measurement.conversations} Conversations, "
        f"{measurement.questions} Questions",
        "",
        f"  Positive accuracy may not fall below "
        f"{measurement.positive_floor:.3f}, the bottom of the ticket-08 "
        f"interval.",
    ]

    for gated in (True, False):
        lines += [
            "",
            f"  Relevance gate {'on' if gated else 'off'}"
            f"{f' (threshold {settings.relevance_threshold})' if gated else ''}:",
            format_header(),
        ]
        lines += [
            format_report(report, report is measurement.chosen)
            for report in measurement.reports
            if report.gated is gated
        ]

    ablation = measurement.ablation
    lines += [
        "",
        f"  chosen threshold {measurement.chosen.threshold:.6f} with the "
        f"Relevance gate {'on' if measurement.chosen.gated else 'off'} "
        f"(hard-neg {measurement.chosen.hard_negative_accuracy:.3f}, positive "
        f"{measurement.chosen.positive_accuracy:.3f}, off-topic "
        f"{measurement.chosen.off_topic_accuracy:.3f}, accuracy "
        f"{measurement.chosen.accuracy:.3f})",
        f"  ablation at that threshold: Off-Topic accuracy "
        f"{ablation.gated.off_topic_accuracy:.3f} with the gate against "
        f"{ablation.ungated.off_topic_accuracy:.3f} without it, Positive "
        f"{ablation.gated.positive_accuracy:.3f} against "
        f"{ablation.ungated.positive_accuracy:.3f} — the gate is "
        f"{'kept' if ablation.keep else 'dropped'}",
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
        "observed entailment probabilities.",
    )
    arguments = parser.parse_args()

    print(
        f"Reranking the top {settings.retrieval_candidates} BM25 Chunks with "
        f"{settings.rerank_model} and judging the best one with "
        f"{settings.nli_model} on {settings.device}."
    )

    # Both models are warmed before the clock starts, exactly as the served
    # process warms them at import: charging the first Question for the first
    # forward pass would report a per-Question time no request ever pays.
    reranker = CrossEncoderReranker(settings)
    reranker.warm_up()

    judge = NliEntailmentJudge(settings)
    judge.warm_up()

    print(
        format_sweep(
            sweep(
                fold=arguments.fold,
                scheme=ChunkScheme(
                    word_lengths=settings.chunk_word_lengths,
                    stride_fraction=settings.chunk_stride_fraction,
                ),
                reranker=reranker,
                rewriter=ClaimRewriter(),
                judge=judge,
                candidates=settings.retrieval_candidates,
                relevance_threshold=settings.relevance_threshold,
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
        "\nChosen for the highest lower bound on Hard-Negative accuracy under "
        "resampling, not by argmax\nand not on blended accuracy, among the "
        "candidates that hold Positive accuracy.\n"
        "The Relevance gate is kept only where removing it costs Off-Topic "
        "accuracy.\n"
        "The test fold is transcribed but not read until tuning ends."
    )


if __name__ == "__main__":
    main()
