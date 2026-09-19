"""Measure the Claim rewriter on the train and dev Questions, and show the work.

    python -m scripts.claim_fidelity
    python -m scripts.claim_fidelity --fold dev

Prints how many Questions were recognized as each shape, what fraction of the
Claims hold every property the rewrite guarantees, and every Question whose
Claim does not — with the Claim beside it, because a rewriter fault is read by
looking at the sentence rather than at the number.

The properties do not include *the split is the right one*, which no check here
can decide: word order is preserved by construction, so re-parsing a Claim
reports back whatever subject boundary it was handed. ADR-0002 records the
residual that leaves.

Train and dev only. The test fold is transcribed but not read until tuning ends.
"""

import argparse

from harness.claim_fidelity import FidelityReport, measure_fold
from harness.folds import FoldName
from medapp.claims import ClaimRewriter, load_parser

READABLE_FOLDS: tuple[FoldName, ...] = ("train", "dev")


def format_report(report: FidelityReport) -> str:
    """One fold's shapes, fidelity and every Claim that failed a property."""
    lines = [
        f"{report.fold}: {report.questions} Questions, fidelity {report.fidelity:.3f}",
        "",
    ]
    lines += [
        f"  {shape:<20} {count:>4}"
        for shape, count in sorted(report.shapes.items(), key=lambda pair: -pair[1])
    ]

    if not report.unfaithful:
        return "\n".join([*lines, "", "  every Claim holds every property"])

    lines += ["", f"  {len(report.unfaithful)} Claims fall short:"]

    for check in report.unfaithful:
        lines += [
            f"    {check.question_id}  ({'; '.join(check.faults)})",
            f"      Q: {check.question}",
            f"      C: {check.claim}",
        ]

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold",
        choices=READABLE_FOLDS,
        help="Which fold to measure. Both, by default.",
    )
    arguments = parser.parse_args()

    spacy_parser = load_parser()
    rewriter = ClaimRewriter(spacy_parser)

    for fold in [arguments.fold] if arguments.fold else READABLE_FOLDS:
        print(format_report(measure_fold(fold, rewriter, spacy_parser)))
        print()

    print(
        "These are the properties the rewrite guarantees, not a verdict on "
        "whether the subject\nboundary is the right one — that needs the parse "
        "this is holding to account. ADR-0002\nrecords the residual read by "
        "hand."
    )


if __name__ == "__main__":
    main()
