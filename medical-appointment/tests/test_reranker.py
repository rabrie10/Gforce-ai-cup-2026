"""The Relevance judge: both sides normalized, scored jointly, ordered by score.

Loading the real cross-encoder costs seconds and gigabytes, so the scoring
model is stubbed here: what is under test is the pairing it is handed, the
order the scores come back in, and that a Question and the speech that answers
it reach it in one written form rather than two. Whether the scores separate
the Question types is measured on the folds by
``python -m scripts.relevance_threshold``, not asserted here.
"""

from collections.abc import Sequence

from medapp.config import Settings
from medapp.reranker import CrossEncoderReranker
from medapp.types import Chunk


class _RecordingModel:
    """A cross-encoder that records its pairs and returns fixed scores."""

    def __init__(self, scores: Sequence[float]) -> None:
        self.scores = scores
        self.pairs: Sequence[tuple[str, str]] = ()
        self.options: dict[str, object] = {}

    def predict(self, inputs, **options):
        self.pairs = list(inputs)
        self.options = options

        return list(self.scores)[: len(self.pairs)]


def chunk(text: str, start: float = 0.0) -> Chunk:
    return Chunk(text=text, start=start, end=start + 1.0, words=())


CHUNKS = (
    chunk("take 100 mg daily for two weeks", 0.0),
    chunk("your blood pressure is 135/88 today", 2.0),
    chunk("we will see you again in six months", 4.0),
)


def reranker(scores: Sequence[float]) -> tuple[CrossEncoderReranker, _RecordingModel]:
    model = _RecordingModel(scores)

    return CrossEncoderReranker(Settings(), model), model


def test_the_chunks_come_back_best_first_whatever_order_they_arrived_in():
    judge, _ = reranker([0.1, 0.9, 0.4])

    reranked = judge.rerank("Was the blood pressure 135/88?", CHUNKS)

    assert [candidate.chunk.text for candidate in reranked] == [
        CHUNKS[1].text,
        CHUNKS[2].text,
        CHUNKS[0].text,
    ]
    assert [candidate.relevance for candidate in reranked] == [0.9, 0.4, 0.1]


def test_chunks_that_score_alike_keep_the_order_the_retriever_ranked_them_in():
    judge, _ = reranker([0.5, 0.5, 0.5])

    reranked = judge.rerank("Was 100 mg prescribed?", CHUNKS)

    assert [candidate.chunk.text for candidate in reranked] == [
        chunk.text for chunk in CHUNKS
    ]


def test_the_question_reaches_the_model_in_the_written_form_the_chunks_carry():
    judge, model = reranker([0.5] * 3)

    judge.rerank("Was 100 mg prescribed for two weeks?", CHUNKS)

    assert [question for question, _ in model.pairs] == [
        "was 100 mg prescribed for 2 weeks"
    ] * 3


def test_every_retrieved_chunk_is_scored_against_the_question_jointly():
    judge, model = reranker([0.5] * 3)

    judge.rerank("Was 100 mg prescribed?", CHUNKS)

    assert [passage for _, passage in model.pairs] == [chunk.text for chunk in CHUNKS]


def test_a_question_nothing_was_retrieved_for_is_not_sent_to_the_model():
    judge, model = reranker([0.5])

    assert judge.rerank("Was 100 mg prescribed?", ()) == ()
    assert model.pairs == ()


def test_settings_resolve_the_batch_the_model_is_handed():
    model = _RecordingModel([0.5] * 3)

    CrossEncoderReranker(Settings(rerank_batch_size=4), model).rerank(
        "Was 100 mg prescribed?", CHUNKS
    )

    assert model.options["batch_size"] == 4
