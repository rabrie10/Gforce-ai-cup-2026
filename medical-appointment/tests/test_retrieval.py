"""BM25 over one Conversation's Chunks.

What is under test is that both sides reach the index through the same
normalizer — a Question written in symbols against speech that spells the
number out — and that stemming closes the plural gap. The ranking quality
itself is measured on the folds by ``python -m scripts.retrieval_metrics``, not
asserted here.
"""

import pytest

from medapp.chunker import ChunkScheme, chunk_conversation
from medapp.retrieval import Bm25Index
from medapp.types import Chunk, Word
from tests.test_chunker import CONVERSATION

SCHEME = ChunkScheme(word_lengths=(4, 8), stride_fraction=0.5)


def index() -> Bm25Index:
    return Bm25Index(chunk_conversation(CONVERSATION, SCHEME))


def test_a_question_written_in_symbols_finds_speech_that_spells_it_out():
    best = index().rank("Was the blood pressure 135/88?", limit=1)[0]

    assert "135/88" in best.text


def test_a_question_written_in_units_finds_the_spoken_dose():
    best = index().rank("Was 100 mg prescribed?", limit=1)[0]

    assert "100 mg" in best.text


def test_stemming_closes_the_gap_between_a_plural_and_its_singular():
    best = index().rank("Is the course one week long?", limit=1)[0]

    assert "weeks" in best.text


def test_the_ranking_is_as_deep_as_asked_for_and_no_deeper():
    ranked = index().rank("What is the blood pressure?", limit=3)

    assert len(ranked) == 3


def test_a_conversation_shorter_than_the_ranking_returns_what_it_has():
    chunks = chunk_conversation(CONVERSATION, ChunkScheme((60,), 1.0))

    assert len(Bm25Index(chunks).rank("blood pressure", limit=10)) == 1


def test_a_question_with_no_indexable_term_ranks_nothing_rather_than_guessing():
    assert index().rank("And it is, or it is not?", limit=5) == ()


def test_a_conversation_with_no_chunks_has_nothing_to_retrieve_from():
    with pytest.raises(ValueError):
        Bm25Index(())


def test_a_ranking_holds_at_least_one_chunk():
    with pytest.raises(ValueError):
        index().rank("blood pressure", limit=0)


def test_only_the_conversations_own_chunks_come_back():
    chunks = chunk_conversation(CONVERSATION, SCHEME)

    assert set(index().rank("blood pressure", limit=5)) <= set(chunks)


def test_the_index_is_built_from_the_chunks_it_was_handed():
    word = Word(text="penicillin", start=0.0, end=1.0, probability=0.9)
    chunk = Chunk(text="penicillin", start=0.0, end=1.0, words=(word,))

    assert Bm25Index([chunk]).rank("Any penicillin?", limit=1) == (chunk,)
