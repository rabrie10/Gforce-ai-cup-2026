"""Report the Transcriber's share of the tIoU ceiling, per fold.

Every Evidence Span the system returns is snapped to a word boundary, so the
distance from an annotated boundary to the nearest word boundary is error the
Chunker cannot remove. Knowing it before a Chunker exists is what keeps the
Chunker's gate (ADR-0002: oracle-selected tIoU >= 0.75) honest.

    python -m scripts.asr_timing_error

Train and dev only. The test fold is transcribed but not read until tuning ends.
"""

from harness.folds import FoldName
from harness.timing_error import TimingErrorReport, report

READABLE_FOLDS: tuple[FoldName, ...] = ("train", "dev")


def format_report(measurement: TimingErrorReport) -> str:
    """One report as a table row."""
    return (
        f"  {measurement.fold:<6}{measurement.spans:>7}"
        f"{measurement.median:>10.3f}{measurement.p90:>10.3f}"
        f"{measurement.worst:>10.3f}"
    )


def main() -> None:
    print("ASR boundary distance, annotated Evidence Span to nearest word edge.")
    print(f"  {'fold':<6}{'spans':>7}{'median':>10}{'p90':>10}{'worst':>10}")

    for fold in READABLE_FOLDS:
        print(format_report(report(fold)))

    print("\nSeconds. The test fold is transcribed but not read until tuning ends.")


if __name__ == "__main__":
    main()
