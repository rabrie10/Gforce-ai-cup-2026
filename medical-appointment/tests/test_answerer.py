"""The Answerer seam: one Verdict per Question, and the Settings that name it.

The placeholder Answerer is the tracer bullet that lights the protocol seam up
before any retrieval exists. What is under test is the seam's contract — a
Verdict per Question, carrying the Chunks it was chosen from — and that a
strategy nothing implements fails loudly at startup instead of being
substituted.
"""

from collections.abc import Iterator

import pytest

from medapp.answerer import (
    Bm25RetrievalAnswerer,
    CiteFirstSegmentAnswerer,
    build_answerer,
)
from medapp.chunker import ChunkScheme
from medapp.config import Settings
from medapp.types import Segment, Word
from tests.test_chunker import CONVERSATION

SCHEME = ChunkScheme(word_lengths=(4, 8), stride_fraction=0.5)

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


def test_the_bm25_answerer_cites_the_chunk_it_ranked_first():
    answerer = Bm25RetrievalAnswerer(scheme=SCHEME, candidates=5)

    verdicts = list(answerer.answer(CONVERSATION, ["Was the blood pressure 135/88?"]))

    assert verdicts[0].answer is True
    assert verdicts[0].evidence == verdicts[0].candidates[0].span
    assert "135/88" in verdicts[0].candidates[0].text


def test_every_bm25_verdict_carries_the_ranking_it_was_chosen_from():
    answerer = Bm25RetrievalAnswerer(scheme=SCHEME, candidates=4)
    questions = ["Was 100 mg prescribed?", "Is the course two weeks?"]

    verdicts = list(answerer.answer(CONVERSATION, questions))

    assert len(verdicts) == len(questions)
    assert all(len(verdict.candidates) == 4 for verdict in verdicts)


def test_a_question_nothing_can_be_retrieved_for_is_answered_no():
    """A no, not a raise: the Questions after it are still worth answering."""
    answerer = Bm25RetrievalAnswerer(scheme=SCHEME, candidates=5)

    verdicts = list(answerer.answer(CONVERSATION, ["And it is, or it is not?"]))

    assert verdicts[0].answer is False
    assert verdicts[0].evidence is None
    assert verdicts[0].candidates == ()


def test_the_verdicts_are_produced_lazily_so_the_deadline_is_checked_between_them():
    answerer = Bm25RetrievalAnswerer(scheme=SCHEME, candidates=1)

    verdicts = answerer.answer(CONVERSATION, ["Was 100 mg prescribed?"] * 3)

    assert next(iter(verdicts)).answer is True
    assert isinstance(verdicts, Iterator)


def test_a_conversation_that_transcribed_to_nothing_raises_rather_than_citing_nothing():
    answerer = Bm25RetrievalAnswerer(scheme=SCHEME, candidates=5)

    with pytest.raises(ValueError):
        list(answerer.answer((), ["Was 100 mg prescribed?"]))


def test_settings_resolve_the_granularities_the_bm25_answerer_is_built_with():
    answerer = build_answerer(
        Settings(
            answer_strategy="retrieve_bm25",
            chunk_word_lengths=(3, 9),
            chunk_stride_fraction=0.5,
            retrieval_candidates=2,
        )
    )

    verdicts = list(answerer.answer(CONVERSATION, ["Was the blood pressure 135/88?"]))

    assert isinstance(answerer, Bm25RetrievalAnswerer)
    assert len(verdicts[0].candidates) == 2
    assert all(len(chunk.text.split()) <= 9 for chunk in verdicts[0].candidates)
