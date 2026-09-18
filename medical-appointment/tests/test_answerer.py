"""The Answerer seam: one Verdict per Question, and the Settings that name it.

The placeholder Answerer is the tracer bullet that lights the protocol seam up
before any retrieval exists. What is under test is the seam's contract — a
Verdict per Question, carrying the Chunks it was chosen from — and that a
strategy nothing implements fails loudly at startup instead of being
substituted.
"""

import pytest

from medapp.answerer import CiteFirstSegmentAnswerer, build_answerer
from medapp.config import Settings
from medapp.types import Segment, Word

SEGMENTS = (
    Segment(
        start=0.4,
        end=1.4,
        text="Take two tablets.",
        words=(Word(text="Take", start=0.4, end=1.4, probability=0.9),),
    ),
    Segment(
        start=1.6,
        end=2.6,
        text="For six weeks.",
        words=(Word(text="For", start=1.6, end=2.6, probability=0.9),),
    ),
)


def test_it_answers_every_question_citing_the_first_segment():
    questions = ["Two tablets?", "Six weeks?", "Any penicillin?"]

    verdicts = list(CiteFirstSegmentAnswerer().answer(SEGMENTS, questions))

    assert len(verdicts) == len(questions)
    assert [verdict.answer for verdict in verdicts] == [True, True, True]
    assert all(verdict.evidence == (0.4, 1.4) for verdict in verdicts)


def test_every_verdict_carries_the_chunks_it_was_chosen_from():
    verdicts = list(CiteFirstSegmentAnswerer().answer(SEGMENTS, ["Two tablets?"]))

    candidates = verdicts[0].candidates

    assert [chunk.span for chunk in candidates] == [(0.4, 1.4)]
    assert candidates[0].words == SEGMENTS[0].words


def test_a_conversation_with_no_segments_raises_rather_than_citing_nothing():
    with pytest.raises(ValueError):
        list(CiteFirstSegmentAnswerer().answer((), ["Two tablets?"]))


def test_settings_name_the_answerer_that_is_constructed():
    answerer = build_answerer(Settings(answer_strategy="cite_first_segment"))

    assert isinstance(answerer, CiteFirstSegmentAnswerer)


def test_a_strategy_with_no_implementation_fails_rather_than_falling_back():
    with pytest.raises(NotImplementedError):
        build_answerer(Settings(answer_strategy="retrieve_rerank_entail"))
