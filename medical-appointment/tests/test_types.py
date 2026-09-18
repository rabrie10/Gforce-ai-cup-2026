"""The data types are frozen, and a Chunk carries its own Evidence Span."""

import dataclasses

import pytest

from medapp.types import Chunk, Segment, Verdict, Word


def _words() -> tuple[Word, ...]:
    return (
        Word(text="one", start=1.0, end=1.4, probability=0.98),
        Word(text="hundred", start=1.4, end=1.9, probability=0.95),
    )


def test_chunk_span_is_its_boundaries():
    chunk = Chunk(text="one hundred", start=1.0, end=1.9, words=_words())

    assert chunk.span == (1.0, 1.9)


def test_chunk_boundaries_cannot_be_edited_after_construction():
    chunk = Chunk(text="one hundred", start=1.0, end=1.9, words=_words())

    with pytest.raises(dataclasses.FrozenInstanceError):
        chunk.start = 0.0


@pytest.mark.parametrize(
    "instance",
    [
        Word(text="one", start=1.0, end=1.4, probability=0.98),
        Segment(start=1.0, end=1.9, text="one hundred", words=_words()),
        Verdict(answer=False, evidence=None, candidates=()),
    ],
)
def test_every_type_is_frozen(instance):
    field = dataclasses.fields(instance)[0].name

    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, field, getattr(instance, field))


def test_a_no_verdict_points_at_nothing():
    verdict = Verdict(answer=False, evidence=None, candidates=())

    assert verdict.evidence is None


@pytest.mark.parametrize(
    ("answer", "evidence"),
    [(True, None), (False, (1.0, 1.9))],
)
def test_a_verdict_answer_and_its_evidence_are_one_decision(answer, evidence):
    with pytest.raises(ValueError):
        Verdict(answer=answer, evidence=evidence, candidates=())


def test_a_yes_verdict_carries_the_span_it_was_read_from():
    chunk = Chunk(text="one hundred", start=1.0, end=1.9, words=_words())

    verdict = Verdict(answer=True, evidence=chunk.span, candidates=(chunk,))

    assert verdict.evidence == (1.0, 1.9)
