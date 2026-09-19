"""Score the configured Answerer on a fold, on the competition's own metric.

    python -m scripts.score
    python -m scripts.score --fold train

This is the number the attempt is graded on — ``0.4 x Accuracy + 0.6 x mean
tIoU`` — and the one every threshold should be chosen against. The component
tables underneath it are diagnostics: they say *which* judgement moved, not
whether the move was worth making.

Train and dev only. The test fold is scored once, at the end of tuning, by
``python -m scripts.score --fold test --release-the-test-fold``, and nothing is
tuned afterwards.
"""

import argparse
import logging

from harness.folds import FoldName, checked_fold_name
from harness.score import ScoreReport, report
from medapp.answerer import SpanRefiningAnswerer, build_answerer, span_padding
from medapp.config import settings

READABLE_FOLDS: tuple[FoldName, ...] = ("train", "dev")


def format_report(measurement: ScoreReport) -> str:
    """The score, its two halves and the diagnostics behind them."""
    lines = [
        f"{measurement.fold}: {measurement.conversations} Conversations, "
        f"{measurement.questions} Questions, {measurement.positives} annotated "
        "Positives",
        f"  answer strategy      {settings.answer_strategy}",
        f"  retrieval mode       {settings.retrieval_mode}",
        "",
        f"  score                {measurement.score:.3f}  "
        f"{_format(measurement.score_interval)}",
        f"    accuracy   (x0.4)  {measurement.accuracy:.3f}  "
        f"{_format(measurement.accuracy_interval)}",
        f"    mean tIoU  (x0.6)  {measurement.mean_tiou:.3f}  "
        f"{_format(measurement.mean_tiou_interval)}",
        "",
        "  Accuracy by question type",
    ]

    for question_type, (
        accuracy,
        correct,
        total,
    ) in measurement.accuracy_by_type.items():
        lines.append(f"    {question_type:<20} {accuracy:.3f}  ({correct}/{total})")

    lines += [
        "",
        "  Evidence localization",
        f"    missed positives     {measurement.missed_positives}"
        "   (zero on both halves)",
        f"    no span returned     {measurement.spans_not_returned}",
        f"    tIoU when answered   {measurement.mean_tiou_answered_yes:.3f}"
        "   (diagnostic, not scored)",
    ]

    return "\n".join(lines)


def _format(bounds: tuple[float, float]) -> str:
    """One bootstrap interval, as the tables print it."""
    return f"[{bounds[0]:.3f}, {bounds[1]:.3f}]"


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Which fold to score, and whether the test fold has been released."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold",
        default="dev",
        help="Which fold to score. Thresholds are chosen on dev.",
    )
    parser.add_argument(
        "--release-the-test-fold",
        action="store_true",
        help=(
            "Score the test fold. It is read once, when tuning has ended, and "
            "no Setting changes afterwards."
        ),
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Score one fold and print it.

    Raises:
        KeyError: If the fold name does not name a fold.
    """
    logging.basicConfig(level=logging.INFO)
    arguments = parse_arguments(argv)
    fold = checked_fold_name(arguments.fold)

    if fold not in READABLE_FOLDS and not arguments.release_the_test_fold:
        print(
            f"The {fold} fold is scored once, when tuning has ended. Pass "
            "--release-the-test-fold to read it, and change no Setting "
            "afterwards."
        )
        return 1

    answerer = SpanRefiningAnswerer(build_answerer(settings), span_padding(settings))

    print(format_report(report(fold, answerer)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
