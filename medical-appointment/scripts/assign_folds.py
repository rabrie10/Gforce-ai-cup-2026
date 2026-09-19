"""Draw the grouped train/dev/test split once and freeze it to disk.

ADR-0002 splits 20/40/40 at Conversation level, so the draw is over the 39
transcript ids rather than the 390 Question rows. Run once; the resulting
``data/folds.json`` is committed and every later measurement reads it through
``harness.folds``. Re-running is an explicit act:

    python -m scripts.assign_folds --force

Overwriting invalidates every number measured before it, including any the test
fold has already been scored against.
"""

import argparse
import json
import random
from pathlib import Path

from harness.folds import FOLD_FILE, FOLD_SIZES, FoldAssignment, FoldName
from utils import load_sample_questions

# The seed the committed assignment was drawn with. Recorded in the fold file
# as well, so the file can be reproduced from its own contents.
DEFAULT_SEED = 20260918


def assign(seed: int = DEFAULT_SEED) -> FoldAssignment:
    """Draw the fold assignment over the supplied Conversations.

    The draw is over sorted transcript ids so that it depends on the seed
    alone, not on the order rows happen to appear in the CSV.

    Args:
        seed: Seed for the shuffle, recorded in the assignment.

    Returns:
        The assignment, with transcript ids sorted within each fold.

    Raises:
        ValueError: If the supplied data does not hold the number of
            Conversations the split assumes.
    """
    transcript_ids = sorted({row["transcript_id"] for row in load_sample_questions()})

    expected = sum(FOLD_SIZES.values())
    if len(transcript_ids) != expected:
        raise ValueError(
            f"The split assumes {expected} Conversations; the supplied data "
            f"holds {len(transcript_ids)}."
        )

    shuffled = random.Random(seed).sample(transcript_ids, k=len(transcript_ids))

    folds: dict[FoldName, tuple[str, ...]] = {}
    start = 0
    for fold, size in FOLD_SIZES.items():
        folds[fold] = tuple(sorted(shuffled[start : start + size]))
        start += size

    return FoldAssignment(seed=seed, folds=folds)


def write_assignment(
    assignment: FoldAssignment, destination: Path, force: bool
) -> None:
    """Write the assignment as JSON.

    Args:
        assignment: The assignment to freeze.
        destination: Where to write it.
        force: Whether to overwrite an existing assignment.

    Raises:
        FileExistsError: If ``destination`` exists and ``force`` is false.
    """
    if destination.exists() and not force:
        raise FileExistsError(
            f"{destination} already exists. The fold assignment is drawn once "
            "and frozen; re-drawing it invalidates every measurement taken "
            "against the old folds. Pass --force if that is what you mean."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(assignment.as_dict(), indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the frozen assignment, invalidating earlier measurements.",
    )
    arguments = parser.parse_args()

    assignment = assign(seed=arguments.seed)
    try:
        write_assignment(assignment, FOLD_FILE, force=arguments.force)
    except FileExistsError as refusal:
        raise SystemExit(str(refusal)) from refusal

    print(f"Wrote {FOLD_FILE} with seed {assignment.seed}:")
    for fold, transcript_ids in assignment.folds.items():
        print(f"  {fold:<5} {len(transcript_ids):>2}  {' '.join(transcript_ids)}")


if __name__ == "__main__":
    main()
