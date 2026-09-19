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
    RerankEntailAnswerer,
    RerankRelevanceAnswerer,
    build_answerer,
)
from medapp.chunker import ChunkScheme
from medapp.claims import Claim
from medapp.config import Settings
from medapp.entailment import Judgement
from medapp.normalizer import normalize_text
from medapp.types import ScoredChunk, Segment, Word
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
        build_answerer(Settings(answer_strategy="single_llm"))


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


class _RelevanceOf:
    """A Relevance judge that scores by the words a Chunk shares with a Question.

    Stands in for the cross-encoder, whose weights are not in the repository.
    What the Answerer tests need from it is a score that is high for a
    Conversation that discusses the Question and low for one that does not, and
    that is what sharing content words gives.
    """

    def rerank(self, question, chunks):
        wanted = set(normalize_text(question))

        return tuple(
            sorted(
                (
                    ScoredChunk(
                        chunk=chunk,
                        relevance=len(wanted & set(chunk.text.split()))
                        / max(len(wanted), 1),
                    )
                    for chunk in chunks
                ),
                key=lambda candidate: -candidate.relevance,
            )
        )


def rerank_answerer(threshold: float) -> RerankRelevanceAnswerer:
    return RerankRelevanceAnswerer(
        scheme=SCHEME,
        candidates=5,
        reranker=_RelevanceOf(),
        threshold=threshold,
    )


def test_the_reranked_answerer_cites_the_chunk_the_judge_ranked_first():
    verdicts = list(
        rerank_answerer(threshold=0.1).answer(
            CONVERSATION, ["Was the blood pressure 135/88?"]
        )
    )

    assert verdicts[0].answer is True
    assert verdicts[0].evidence == verdicts[0].candidates[0].span
    assert "135/88" in verdicts[0].candidates[0].text


def test_a_question_no_chunk_is_relevant_enough_for_is_answered_no_with_nulls():
    verdicts = list(
        rerank_answerer(threshold=0.9).answer(
            CONVERSATION, ["Was the blood pressure 135/88?"]
        )
    )

    assert verdicts[0].answer is False
    assert verdicts[0].evidence is None


def test_a_no_still_carries_the_candidates_it_was_judged_over():
    """The component metrics read them, and so will the Entailment judge."""
    verdicts = list(
        rerank_answerer(threshold=0.9).answer(
            CONVERSATION, ["Was the blood pressure 135/88?"]
        )
    )

    assert len(verdicts[0].candidates) == 5


def test_the_reranked_verdicts_are_produced_lazily():
    verdicts = rerank_answerer(threshold=0.1).answer(
        CONVERSATION, ["Was 100 mg prescribed?"] * 3
    )

    assert next(iter(verdicts)).answer is True
    assert isinstance(verdicts, Iterator)


def test_a_conversation_that_transcribed_to_nothing_raises_rather_than_guessing():
    with pytest.raises(ValueError):
        list(rerank_answerer(threshold=0.1).answer((), ["Was 100 mg prescribed?"]))


class _ClaimIs:
    """A rewriter that records what it was asked and returns a fixed Claim."""

    def __init__(self, text: str = "the claim") -> None:
        self.text = text
        self.questions: list[str] = []

    def rewrite(self, question):
        self.questions.append(question)

        return Claim(text=self.text, shape="declarative")


class _EntailmentOf:
    """An Entailment judge that scores by a fixed table, keyed on Chunk text."""

    def __init__(self, scores: dict[str, float], default: float = 0.0) -> None:
        self.scores = scores
        self.default = default
        self.claims: list[str] = []
        self.judged: list[tuple[str, ...]] = []

    def judge(self, claim, chunks):
        self.claims.append(claim)
        self.judged.append(tuple(chunk.text for chunk in chunks))

        return tuple(
            Judgement(
                chunk=chunk,
                entailment=self.scores.get(chunk.text, self.default),
                neutral=0.0,
                contradiction=0.0,
            )
            for chunk in chunks
        )


def entail_answerer(
    judge: _EntailmentOf,
    rewriter: _ClaimIs | None = None,
    relevance_threshold: float | None = 0.1,
    entailment_threshold: float = 0.5,
) -> RerankEntailAnswerer:
    return RerankEntailAnswerer(
        scheme=SCHEME,
        candidates=5,
        reranker=_RelevanceOf(),
        rewriter=rewriter or _ClaimIs(),
        judge=judge,
        relevance_threshold=relevance_threshold,
        entailment_threshold=entailment_threshold,
    )


BLOOD_PRESSURE = "Was the blood pressure 135/88?"


def top_chunk(question: str = BLOOD_PRESSURE) -> str:
    """What the Relevance judge ranks first for one Question."""
    answerer = rerank_answerer(threshold=0.0)

    return list(answerer.answer(CONVERSATION, [question]))[0].candidates[0].text


def test_a_chunk_that_entails_the_claim_is_answered_yes_and_cited():
    judge = _EntailmentOf({top_chunk(): 0.9})

    verdicts = list(entail_answerer(judge).answer(CONVERSATION, [BLOOD_PRESSURE]))

    assert verdicts[0].answer is True
    assert verdicts[0].evidence == verdicts[0].candidates[0].span


def test_a_relevant_chunk_that_does_not_entail_the_claim_is_answered_no():
    """The Hard Negative: the right subject, and the Conversation does not say
    it."""
    judge = _EntailmentOf({}, default=0.2)

    verdicts = list(entail_answerer(judge).answer(CONVERSATION, [BLOOD_PRESSURE]))

    assert verdicts[0].answer is False
    assert verdicts[0].evidence is None
    assert verdicts[0].candidates != ()


def test_a_question_nothing_is_relevant_enough_for_never_reaches_the_judge():
    judge = _EntailmentOf({}, default=1.0)

    verdicts = list(
        entail_answerer(judge, relevance_threshold=0.99).answer(
            CONVERSATION, [BLOOD_PRESSURE]
        )
    )

    assert verdicts[0].answer is False
    assert judge.claims == []


def test_only_the_chunk_the_evidence_span_comes_from_is_judged():
    """Judging deeper would let the yes come from a Chunk the Verdict does not
    cite."""
    judge = _EntailmentOf({top_chunk(): 0.9})

    list(entail_answerer(judge).answer(CONVERSATION, [BLOOD_PRESSURE]))

    assert judge.judged == [(top_chunk(),)]


def test_the_judge_reads_the_claim_the_question_was_rewritten_as():
    rewriter = _ClaimIs("the blood pressure was 135/88")
    judge = _EntailmentOf({top_chunk(): 0.9})

    list(entail_answerer(judge, rewriter).answer(CONVERSATION, [BLOOD_PRESSURE]))

    assert rewriter.questions == [BLOOD_PRESSURE]
    assert judge.claims == ["the blood pressure was 135/88"]


def test_the_relevance_gate_can_be_ablated_away():
    """The ablation the threshold script measures runs the same Answerer with
    the gate off."""
    judge = _EntailmentOf({top_chunk(): 0.9})

    verdicts = list(
        entail_answerer(judge, relevance_threshold=None).answer(
            CONVERSATION, [BLOOD_PRESSURE]
        )
    )

    assert verdicts[0].answer is True


def test_the_entailed_verdicts_are_produced_lazily():
    judge = _EntailmentOf({}, default=0.9)

    verdicts = entail_answerer(judge).answer(CONVERSATION, [BLOOD_PRESSURE] * 3)

    assert next(iter(verdicts)).answer is True
    assert isinstance(verdicts, Iterator)


def test_entailing_a_conversation_that_transcribed_to_nothing_raises():
    judge = _EntailmentOf({}, default=0.9)

    with pytest.raises(ValueError):
        list(entail_answerer(judge).answer((), [BLOOD_PRESSURE]))


def test_settings_resolve_the_thresholds_the_entailing_answerer_is_built_with():
    answerer = build_answerer(
        Settings(answer_strategy="retrieve_rerank_entail", device="cpu")
    )

    assert isinstance(answerer, RerankEntailAnswerer)
