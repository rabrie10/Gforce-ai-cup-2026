"""The fold assignment is grouped, frozen and exhaustive.

Grouping is the property ADR-0002 rests on: a transcript in two folds leaks a
Conversation's Chunks across the split and every number measured afterwards is
optimistic in a way nothing downstream can detect.
"""

import json

import pytest

from harness import folds as folds_module
from harness.folds import (
    FOLD_SIZES,
    load_fold_assignment,
    questions_in_fold,
    transcript_ids_in_fold,
)
from utils import load_sample_questions


def test_every_transcript_lands_in_exactly_one_fold():
    assignment = load_fold_assignment()

    seen = [
        transcript_id
        for fold in assignment.folds
        for transcript_id in assignment.folds[fold]
    ]

    assert len(seen) == len(set(seen))


def test_the_folds_cover_every_supplied_conversation():
    supplied = {row["transcript_id"] for row in load_sample_questions()}

    assigned = {
        transcript_id
        for ids in load_fold_assignment().folds.values()
        for transcript_id in ids
    }

    assert assigned == supplied


def test_fold_sizes_are_the_grouped_20_40_40_split():
    assignment = load_fold_assignment()

    sizes = {fold: len(ids) for fold, ids in assignment.folds.items()}

    assert sizes == FOLD_SIZES
    assert sum(sizes.values()) == 39


def test_the_file_records_the_seed_it_was_generated_with():
    assert isinstance(load_fold_assignment().seed, int)


def test_every_question_of_a_conversation_lands_in_the_same_fold():
    fold_of_transcript = {
        transcript_id: fold
        for fold, ids in load_fold_assignment().folds.items()
        for transcript_id in ids
    }

    for fold in FOLD_SIZES:
        for row in questions_in_fold(fold):
            assert fold_of_transcript[row["transcript_id"]] == fold


def test_the_folds_partition_the_questions():
    counts = {fold: len(questions_in_fold(fold)) for fold in FOLD_SIZES}

    assert sum(counts.values()) == len(load_sample_questions())


def test_questions_keep_the_order_the_evaluator_sends_them_in():
    supplied = [row["question_id"] for row in load_sample_questions()]
    test_ids = [row["question_id"] for row in questions_in_fold("test")]

    assert test_ids == [
        question_id for question_id in supplied if question_id in set(test_ids)
    ]


def test_an_unknown_fold_name_is_refused_rather_than_silently_empty():
    with pytest.raises(KeyError):
        transcript_ids_in_fold("validation")


def test_the_assignment_is_read_from_the_committed_file_not_regenerated(monkeypatch):
    """The file on disk is the authority; nothing recomputes a split at import."""
    monkeypatch.setattr(
        folds_module,
        "FOLD_FILE",
        folds_module.PROJECT_ROOT / "data" / "does-not-exist.json",
    )
    load_fold_assignment.cache_clear()

    with pytest.raises(FileNotFoundError):
        load_fold_assignment()

    load_fold_assignment.cache_clear()


def test_the_committed_file_is_reproducible_from_its_own_seed():
    """Regenerating with the recorded seed yields the committed assignment."""
    from scripts.assign_folds import assign

    assignment = load_fold_assignment()

    assert assign(seed=assignment.seed).folds == assignment.folds


def test_the_script_refuses_to_overwrite_the_committed_file(tmp_path):
    from scripts.assign_folds import write_assignment

    destination = tmp_path / "folds.json"
    destination.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_assignment(load_fold_assignment(), destination, force=False)


def test_an_explicit_flag_is_what_overwrites_it(tmp_path):
    from scripts.assign_folds import write_assignment

    destination = tmp_path / "folds.json"
    destination.write_text("{}", encoding="utf-8")

    write_assignment(load_fold_assignment(), destination, force=True)

    assert json.loads(destination.read_text(encoding="utf-8"))["folds"]["train"]
