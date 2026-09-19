"""The Entailment judge: both sides in one written form, labels read by name.

Loading the real NLI model costs seconds and gigabytes, so the model is stubbed
here: what is under test is the pairing it is handed, which way round premise
and hypothesis go, that the three probabilities are read off the columns the
model's own config names rather than off fixed positions, and that the order
the Chunks arrived in is kept. Whether the scores separate a Positive from a
Hard Negative is measured on the folds by
``python -m scripts.entailment_threshold``, not asserted here.
"""

from collections.abc import Sequence

import pytest

from medapp.config import Settings
from medapp.entailment import NliEntailmentJudge
from medapp.types import Chunk


class _Config:
    def __init__(self, id2label: dict[int, str]) -> None:
        self.id2label = id2label


class _RecordingModel:
    """An NLI cross-encoder that records its pairs and returns fixed rows."""

    def __init__(
        self,
        rows: Sequence[Sequence[float]],
        id2label: dict[int, str] | None = None,
    ) -> None:
        self.rows = rows
        self.config: object = _Config(
            id2label
            if id2label is not None
            else {0: "contradiction", 1: "entailment", 2: "neutral"}
        )
        self.pairs: Sequence[tuple[str, str]] = ()
        self.options: dict[str, object] = {}

    def predict(self, inputs, **options):
        self.pairs = list(inputs)
        self.options = options

        return list(self.rows)[: len(self.pairs)]


def chunk(text: str, start: float = 0.0) -> Chunk:
    return Chunk(text=text, start=start, end=start + 1.0, words=())


CHUNKS = (
    chunk("take 100 mg daily for 2 weeks", 0.0),
    chunk("your blood pressure is 135/88 today", 2.0),
)


def judge(
    rows: Sequence[Sequence[float]], id2label: dict[int, str] | None = None
) -> tuple[NliEntailmentJudge, _RecordingModel]:
    model = _RecordingModel(rows, id2label)

    return NliEntailmentJudge(Settings(), model), model


def test_the_chunk_is_the_premise_and_the_claim_is_the_hypothesis():
    """Backwards, the model would be asked whether the Claim establishes the
    Conversation."""
    entailment, model = judge([[0.1, 0.8, 0.1]] * 2)

    entailment.judge("the patient does take 100 mg", CHUNKS)

    assert [premise for premise, _ in model.pairs] == [c.text for c in CHUNKS]
    assert [hypothesis for _, hypothesis in model.pairs] == [
        "the patient does take 100 mg"
    ] * 2


def test_the_claim_reaches_the_model_in_the_written_form_the_chunks_carry():
    entailment, model = judge([[0.1, 0.8, 0.1]])

    entailment.judge("the dose was 100 milligrams for two weeks", CHUNKS[:1])

    assert model.pairs[0][1] == "the dose was 100 mg for 2 weeks"


def test_the_three_probabilities_are_read_off_the_columns_the_model_names():
    entailment, _ = judge(
        [[0.7, 0.2, 0.1]], id2label={0: "entailment", 1: "neutral", 2: "contradiction"}
    )

    judged = entailment.judge("the patient does take 100 mg", CHUNKS[:1])

    assert judged[0].entailment == pytest.approx(0.7)
    assert judged[0].neutral == pytest.approx(0.2)
    assert judged[0].contradiction == pytest.approx(0.1)


def test_a_model_that_is_not_an_nli_model_fails_rather_than_being_read_wrong():
    with pytest.raises(ValueError, match="natural-language-inference"):
        judge([[0.5]], id2label={0: "LABEL_0"})


def test_the_chunks_keep_the_order_relevance_ranked_them_in():
    """Reordering here would cite a different Chunk than the one judged."""
    entailment, _ = judge([[0.1, 0.2, 0.7], [0.1, 0.9, 0.0]])

    judged = entailment.judge("the patient does take 100 mg", CHUNKS)

    assert [judgement.chunk.text for judgement in judged] == [c.text for c in CHUNKS]
    assert judged[0].entailment == pytest.approx(0.2)


def test_a_question_nothing_was_retrieved_for_is_not_sent_to_the_model():
    entailment, model = judge([[0.1, 0.8, 0.1]])

    assert entailment.judge("the patient does take 100 mg", ()) == ()
    assert model.pairs == ()


def test_the_probabilities_are_asked_for_rather_than_the_raw_logits():
    """A threshold on a logit would not mean the same thing between models."""
    entailment, model = judge([[0.1, 0.8, 0.1]])

    entailment.judge("the patient does take 100 mg", CHUNKS[:1])

    assert model.options["apply_softmax"] is True


def test_settings_resolve_the_batch_the_model_is_handed():
    model = _RecordingModel([[0.1, 0.8, 0.1]] * 2)

    NliEntailmentJudge(Settings(nli_batch_size=4), model).judge(
        "the patient does take 100 mg", CHUNKS
    )

    assert model.options["batch_size"] == 4
