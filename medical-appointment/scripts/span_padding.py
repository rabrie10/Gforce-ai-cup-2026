"""Choose the Evidence Span padding on the dev fold, and show the work.

Every candidate (start, end) pad is printed with mean tIoU and its bootstrap
interval. The chosen value is the one with the highest lower bound under
resampling at Conversation level, ties going to the higher point estimate —
stability rather than argmax, as every threshold in this project is chosen.

    python -m scripts.span_padding
    python -m scripts.span_padding --fold train --pads=-0.5,0,0.5

The values it prints become ``MEDAPP_SPAN_PAD_START_SECONDS`` and
``MEDAPP_SPAN_PAD_END_SECONDS``'s defaults in Settings, with this table
recorded beside them.

Train and dev only. The test fold is transcribed but not read until tuning
ends.
"""

import argparse

from harness.folds import FoldName
from harness.span_padding import CANDIDATE_PADS, PaddingReport, PaddingSweep, sweep
from medapp.answerer import build_answerer
from medapp.config import settings

READABLE_FOLDS: tuple[FoldName, ...] = ("train", "dev")


def format_header() -> str:
    """The column names of the sweep table."""
    return f"  {'start':>8}{'end':>8}{'mean tIoU':>12}{'90% interval':>18}"


def format_report(report: PaddingReport, chosen: bool) -> str:
    """One candidate padding as a table row, with the chosen one marked."""
    return (
        f"{'*' if chosen else ' '} {report.start_pad:>8.2f}{report.end_pad:>8.2f}"
        f"{report.mean_tiou:>12.3f}{_format(report.mean_tiou_interval):>18}"
    )


def _format(bounds: tuple[float, float]) -> str:
    """One bootstrap interval as a table cell."""
    low, high = bounds

    return f"[{low:.3f}, {high:.3f}]"


def format_sweep(measurement: PaddingSweep) -> str:
    """One fold's sweep: the table and the chosen padding."""
    lines = [
        f"{measurement.fold}: {measurement.conversations} Conversations, "
        f"{measurement.positives} Positives",
        format_header(),
    ]
    lines += [
        format_report(report, report is measurement.chosen)
        for report in sorted(
            measurement.reports,
            key=lambda report: report.mean_tiou_interval[0],
            reverse=True,
        )[:20]
    ]
    lines += [
        "",
        f"  chosen padding start={measurement.chosen.start_pad:.2f}s "
        f"end={measurement.chosen.end_pad:.2f}s "
        f"(mean tIoU {measurement.chosen.mean_tiou:.3f})",
    ]

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold",
        choices=READABLE_FOLDS,
        default="dev",
        help="Which fold to sweep. The padding is chosen on dev.",
    )
    parser.add_argument(
        "--pads",
        help="Comma-separated candidate offsets in seconds, evaluated as "
        "every (start, end) pair. Defaults to the built-in grid.",
    )
    arguments = parser.parse_args()

    print(f"Answering with the {settings.answer_strategy!r} strategy.")

    answerer = build_answerer(settings)

    print(
        format_sweep(
            sweep(
                fold=arguments.fold,
                answerer=answerer,
                pads=(
                    tuple(float(pad) for pad in arguments.pads.split(","))
                    if arguments.pads
                    else CANDIDATE_PADS
                ),
            )
        )
    )

    print(
        "\nChosen for the highest lower bound on mean tIoU under resampling, "
        "not by argmax.\nTop 20 candidates shown, sorted by that bound.\n"
        "The test fold is transcribed but not read until tuning ends."
    )


if __name__ == "__main__":
    main()
