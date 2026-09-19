"""The frozen train/dev/test assignment, read from disk and never recomputed.

ADR-0002 splits 20/40/40 grouped at Conversation level: all ten Questions of a
Conversation live in the same fold, because they share one transcript and one
set of Chunks. ``data/folds.json`` is written once by
``scripts/assign_folds.py`` and committed; this module is the only way later
measurements reach it, so the test fold stays untouched in practice rather than
by promise.
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, cast, get_args

from utils import load_sample_questions

FoldName = Literal["train", "dev", "test"]

# 20/40/40 of the 39 supplied Conversations, rounded to whole groups.
FOLD_SIZES: dict[FoldName, int] = {"train": 8, "dev": 16, "test": 15}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FOLD_FILE = PROJECT_ROOT / "data" / "folds.json"


@dataclass(frozen=True, slots=True)
class FoldAssignment:
    """Which Conversations belong to which fold, and how that was decided.

    Attributes:
        seed: The seed the assignment was drawn with, recorded so the committed
            file can be reproduced rather than trusted.
        folds: Transcript ids per fold, sorted within each fold.
    """

    seed: int
    folds: dict[FoldName, tuple[str, ...]]

    def as_dict(self) -> dict[str, object]:
        """The on-disk form: the seed and the assignment it produced."""
        return {
            "seed": self.seed,
            "folds": {fold: list(ids) for fold, ids in self.folds.items()},
        }


def checked_fold_name(fold: str) -> FoldName:
    """The fold name, checked at the boundary where it arrives as free text.

    Raises:
        KeyError: If it names no fold.
    """
    if fold not in get_args(FoldName):
        raise KeyError(
            f"{fold!r} is not a fold: the folds are {', '.join(get_args(FoldName))}."
        )

    return cast(FoldName, fold)


@lru_cache(maxsize=1)
def load_fold_assignment(fold_file: Path | None = None) -> FoldAssignment:
    """The committed assignment.

    Args:
        fold_file: Where to read it from. Defaults to the committed file; the
            argument exists so tests can point elsewhere.

    Raises:
        FileNotFoundError: If the fold file is missing. It is committed, and a
            measurement that quietly invented its own split would be worse than
            one that stopped.
        ValueError: If the file on disk does not hold the split ADR-0002
            prescribes.
    """
    fold_file = fold_file or FOLD_FILE

    if not fold_file.exists():
        raise FileNotFoundError(
            f"No fold assignment at {fold_file}. It is committed to the "
            "repository and is generated once, by scripts/assign_folds.py."
        )

    document = json.loads(fold_file.read_text(encoding="utf-8"))
    folds = {
        checked_fold_name(fold): tuple(transcript_ids)
        for fold, transcript_ids in document["folds"].items()
    }

    _check_the_split_is_the_one_adr_0002_prescribes(folds, fold_file)

    return FoldAssignment(seed=int(document["seed"]), folds=folds)


def _check_the_split_is_the_one_adr_0002_prescribes(
    folds: dict[FoldName, tuple[str, ...]], fold_file: Path
) -> None:
    """Check the file on disk still holds a grouped 20/40/40 split.

    The loader is the last place a hand-edited fold file can be caught: a
    transcript id in two folds leaks one Conversation's Chunks across the
    split, and every number measured afterwards would be optimistic with
    nothing downstream able to see it.

    Raises:
        ValueError: If the folds are the wrong sizes, overlap, or do not cover
            the supplied Conversations exactly.
    """
    sizes = {fold: len(ids) for fold, ids in folds.items()}
    if sizes != FOLD_SIZES:
        raise ValueError(
            f"{fold_file} holds folds of {sizes}, not the {FOLD_SIZES} "
            "ADR-0002 prescribes."
        )

    assigned = [transcript_id for ids in folds.values() for transcript_id in ids]
    duplicated = {
        transcript_id for transcript_id in assigned if assigned.count(transcript_id) > 1
    }
    if duplicated:
        raise ValueError(
            f"{fold_file} puts {sorted(duplicated)} in more than one fold. All "
            "Questions of a Conversation belong to one fold."
        )

    supplied = {row["transcript_id"] for row in load_sample_questions()}
    if set(assigned) != supplied:
        raise ValueError(
            f"{fold_file} does not cover the supplied Conversations: "
            f"{sorted(supplied - set(assigned))} unassigned, "
            f"{sorted(set(assigned) - supplied)} unknown."
        )


def transcript_ids_in_fold(fold: FoldName) -> tuple[str, ...]:
    """The Conversations of one fold.

    Raises:
        KeyError: If ``fold`` is not one of train, dev or test.
    """
    return load_fold_assignment().folds[checked_fold_name(fold)]


def questions_in_fold(fold: FoldName) -> list[dict[str, str]]:
    """The supplied Question rows of one fold, in the order the evaluator sends
    them.

    Raises:
        KeyError: If ``fold`` is not one of train, dev or test.
    """
    transcript_ids = set(transcript_ids_in_fold(fold))

    return [
        row for row in load_sample_questions() if row["transcript_id"] in transcript_ids
    ]


def conversations_in_fold(fold: FoldName) -> list[tuple[str, list[dict[str, str]]]]:
    """The fold's Conversations with their Question rows, in CSV order.

    The grouping a measurement runs in: the Chunks and the index are built once
    per Conversation and every Question of it is answered against them, which
    is what happens inside one request.

    Raises:
        KeyError: If ``fold`` is not one of train, dev or test.
    """
    grouped: dict[str, list[dict[str, str]]] = {}

    for row in questions_in_fold(fold):
        grouped.setdefault(row["transcript_id"], []).append(row)

    return list(grouped.items())
