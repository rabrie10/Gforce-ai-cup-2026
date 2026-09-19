"""The dense half: one embedding pass per Conversation, ranked by similarity.

Loading the real bi-encoder costs seconds, so the model is stubbed: what is
under test is that the Conversation is embedded once rather than once per
Question, that both sides reach the model in the written form the Chunks carry,
and that the ranking is by similarity, best first. Whether the dense signal
beats BM25 on this corpus is measured on the folds by
``python -m scripts.retrieval_modes``, not asserted here.
"""

from collections.abc import Sequence

import numpy as np
import pytest

from medapp.config import Settings
from medapp.dense import DenseIndex
from medapp.types import Chunk


class _RecordingEmbedder:
    """An embedder that records its calls and returns fixed vectors."""

    def __init__(self, vectors: dict[str, Sequence[float]]) -> None:
        self.vectors = vectors
        self.calls: list[list[str]] = []
        self.options: dict[str, object] = {}

    def encode(self, sentences, **options):
        texts = list(sentences)
        self.calls.append(texts)
        self.options = options

        return np.array([self.vectors[text] for text in texts], dtype=np.float32)


def chunk(text: str, start: float) -> Chunk:
    return Chunk(text=text, start=start, end=start + 1.0, words=())


CHUNKS = (
    chunk("take 100 mg daily for 2 weeks", 0.0),
    chunk("your blood pressure is 135/88 today", 2.0),
    chunk("we will see you again in 6 months", 4.0),
)

# Unit vectors on three axes, so a Question placed near one of them ranks that
# Chunk first and the ordering is arithmetic rather than a model's opinion.
VECTORS: dict[str, Sequence[float]] = {
    CHUNKS[0].text: (1.0, 0.0, 0.0),
    CHUNKS[1].text: (0.0, 1.0, 0.0),
    CHUNKS[2].text: (0.0, 0.0, 1.0),
    "was the blood pressure 135/88": (0.1, 1.0, 0.2),
    "was 100 mg prescribed for 2 weeks": (1.0, 0.1, 0.0),
}


def index(chunks: Sequence[Chunk] = CHUNKS) -> tuple[DenseIndex, _RecordingEmbedder]:
    embedder = _RecordingEmbedder(VECTORS)

    return DenseIndex(chunks, embedder, Settings()), embedder


def test_the_chunks_come_back_by_similarity_best_first():
    dense, _ = index()

    ranked = dense.rank("Was the blood pressure 135/88?", 3)

    assert [candidate.text for candidate in ranked] == [
        CHUNKS[1].text,
        CHUNKS[2].text,
        CHUNKS[0].text,
    ]


def test_the_ranking_stops_at_the_limit_it_was_asked_for():
    dense, _ = index()

    assert len(dense.rank("Was the blood pressure 135/88?", 2)) == 2


def test_a_limit_deeper_than_the_conversation_returns_what_there_is():
    dense, _ = index()

    assert len(dense.rank("Was the blood pressure 135/88?", 99)) == len(CHUNKS)


def test_the_conversation_is_embedded_once_rather_than_once_per_question():
    """Ten Questions arrive about one Conversation and the Chunks do not
    change between them."""
    dense, embedder = index()

    dense.rank("Was the blood pressure 135/88?", 3)
    dense.rank("Was 100 mg prescribed for 2 weeks?", 3)

    assert embedder.calls[0] == [c.text for c in CHUNKS]
    assert [call for call in embedder.calls[1:]] == [
        ["was the blood pressure 135/88"],
        ["was 100 mg prescribed for 2 weeks"],
    ]


def test_the_question_reaches_the_model_in_the_written_form_the_chunks_carry():
    dense, embedder = index()

    dense.rank("Was 100 mg prescribed for two weeks?", 3)

    assert embedder.calls[-1] == ["was 100 mg prescribed for 2 weeks"]


def test_a_question_that_normalizes_to_nothing_ranks_nothing():
    """BM25 returns nothing for a Question it holds no term of, and fusing a
    ranking against nothing must not invent one."""
    dense, embedder = index()

    assert dense.rank("Um, uh?", 3) == ()
    assert len(embedder.calls) == 1


def test_a_conversation_with_no_chunks_raises_rather_than_ranking_nothing():
    with pytest.raises(ValueError):
        index(())


def test_a_ranking_must_hold_at_least_one_chunk():
    dense, _ = index()

    with pytest.raises(ValueError):
        dense.rank("Was the blood pressure 135/88?", 0)


def test_settings_resolve_the_batch_the_embedder_is_handed():
    embedder = _RecordingEmbedder(VECTORS)

    DenseIndex(CHUNKS, embedder, Settings(dense_batch_size=8))

    assert embedder.options["batch_size"] == 8
