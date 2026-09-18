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
from typing import Literal, get_args

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

    def fold_of(self, transcript_id: str) -> FoldName:
        """The fold one Conversation belongs to.

        Raises:
            KeyError: If the transcript id is not one of the supplied
                Conversations.
        """
        for fold, transcript_ids in self.folds.items():
            if transcript_id in transcript_ids:
                return fold

        raise KeyError(f"{transcript_id!r} is not in the fold assignment.")

    def as_dict(self) -> dict[str, object]:
        """The on-disk form: seed, sizes and the assignment itself."""
        return {
            "seed": self.seed,
            "sizes": {fold: len(ids) for fold, ids in self.folds.items()},
            "folds": {fold: list(ids) for fold, ids in self.folds.items()},
        }


def _check_fold_name(fold: str) -> FoldName:
    if fold not in get_args(FoldName):
        raise KeyError(
            f"{fold!r} is not a fold: the folds are {', '.join(get_args(FoldName))}."
        )

    return fold  # type: ignore[return-value]


@lru_cache(maxsize=1)
def load_fold_assignment() -> FoldAssignment:
    """The committed assignment.

    Raises:
        FileNotFoundError: If the fold file is missing. It is committed, and a
            measurement that quietly invented its own split would be worse than
            one that stopped.
        ValueError: If the file on disk does not hold the split ADR-0002
            prescribes.
    """
    if not FOLD_FILE.exists():
        raise FileNotFoundError(
            f"No fold assignment at {FOLD_FILE}. It is committed to the "
            "repository and is generated once, by scripts/assign_folds.py."
        )

    document = json.loads(FOLD_FILE.read_text(encoding="utf-8"))
    folds = {
        _check_fold_name(fold): tuple(transcript_ids)
        for fold, transcript_ids in document["folds"].items()
    }

    sizes = {fold: len(ids) for fold, ids in folds.items()}
    if sizes != FOLD_SIZES:
        raise ValueError(
            f"{FOLD_FILE} holds folds of {sizes}, not the {FOLD_SIZES} "
            "ADR-0002 prescribes."
        )

    return FoldAssignment(seed=int(document["seed"]), folds=folds)


def transcript_ids_in_fold(fold: str) -> tuple[str, ...]:
    """The Conversations of one fold.

    Raises:
        KeyError: If ``fold`` is not one of train, dev or test.
    """
    return load_fold_assignment().folds[_check_fold_name(fold)]


def questions_in_fold(fold: str) -> list[dict[str, str]]:
    """The supplied Question rows of one fold, in the order the evaluator sends
    them.

    Raises:
        KeyError: If ``fold`` is not one of train, dev or test.
    """
    transcript_ids = set(transcript_ids_in_fold(fold))

    return [
        row for row in load_sample_questions() if row["transcript_id"] in transcript_ids
    ]
