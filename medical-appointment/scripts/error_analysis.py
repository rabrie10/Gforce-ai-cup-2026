"""Print every train and dev Question beside the transcript, for reading.

The failure taxonomy in ``.scratch/error-taxonomy.md`` is written from this
output: each Positive Question appears with the words its annotated Evidence
Span covers and the words either side, and each Hard Negative and Off-Topic
Question appears with the passage of its Conversation it comes closest to.

    python -m scripts.error_analysis --fold train
    python -m scripts.error_analysis --fold dev --positives

Train and dev only. The test fold is refused: the taxonomy decides how the
Chunker and the judges are built, so reading the test fold here would leak it
into every design decision that follows.
"""

import argparse

from harness.error_analysis import (
    READABLE_FOLDS,
    NegativeRendering,
    PositiveRendering,
    negatives,
    positives,
    readable_fold,
)


def format_positive(rendering: PositiveRendering) -> str:
    """One annotated Question, its span and the words either side."""
    start, end = rendering.span

    return "\n".join(
        (
            f"[{rendering.question_id}] {rendering.question}",
            f"  span      {start:.2f}-{end:.2f}s "
            f"({end - start:.2f}s, {rendering.words_covered} words, "
            f"{rendering.segments_crossed} segments)",
            f"  before    ...{rendering.before}",
            f"  SPAN      {rendering.span_text}",
            f"  after     {rendering.after}...",
        )
    )


def format_negative(rendering: NegativeRendering) -> str:
    """One unannotated Question beside the passage it comes closest to."""
    start, end = rendering.nearest_span

    return "\n".join(
        (
            f"[{rendering.question_id}] ({rendering.question_type}) "
            f"{rendering.question}",
            f"  nearest   {start:.2f}-{end:.2f}s, "
            f"{rendering.overlap} content words shared",
            f"  passage   {rendering.nearest_text}",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold",
        default="train",
        help=f"Which fold to read: {' or '.join(READABLE_FOLDS)}.",
    )
    kinds = parser.add_mutually_exclusive_group()
    kinds.add_argument(
        "--positives",
        action="store_true",
        help="Only the Questions carrying an annotated Evidence Span.",
    )
    kinds.add_argument(
        "--negatives",
        action="store_true",
        help="Only the Hard Negative and Off-Topic Questions.",
    )
    arguments = parser.parse_args()

    try:
        fold = readable_fold(arguments.fold)
    except ValueError as refusal:
        parser.error(str(refusal))

    show_positives = not arguments.negatives
    show_negatives = not arguments.positives

    if show_positives:
        rendered = positives(fold)
        print(f"== {fold}: {len(rendered)} annotated Evidence Spans ==\n")
        for rendering in rendered:
            print(format_positive(rendering))
            print()

    if show_negatives:
        unannotated = negatives(fold)
        print(f"== {fold}: {len(unannotated)} Questions answered no ==\n")
        for negative in unannotated:
            print(format_negative(negative))
            print()


if __name__ == "__main__":
    main()
