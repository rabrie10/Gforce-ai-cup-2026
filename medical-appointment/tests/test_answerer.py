"""The Answerer seam: one Verdict per Question, and the Settings that name it.

What is under test is the seam's contract — a Verdict per Question, carrying
the Chunks it was chosen from and the one it read its Evidence Span off — and
that a strategy nothing implements fails loudly at startup instead of being
substituted.
"""

from collections.abc import Iterator

import pytest

from medapp.answerer import (
    CiteFirstSegmentAnswerer,
    RerankEntailAnswerer,
    RerankRelevanceAnswerer,
    build_answerer,
)
from medapp.chunker import SentenceScheme, chunk_conversation
from medapp.claims import Claim
from medapp.config import Settings
from medapp.entailment import Judgement
from medapp.normalizer import normalize_text
from medapp.types import Chunk, ScoredChunk, Segment, Word
from tests.test_chunker import conversation

SCHEME = SentenceScheme(max_words=60)

# Longer than the Chunker fixture: the depth tests need more sentences than a
# Verdict cites, so that a Chunk further down the ranking is there to be judged.
CONVERSATION = conversation(
    "so there is cardiovascular disease in your family? yes there is. "
    "your blood pressure is 135 over 88 today, which is fine. "
    "take one hundred milligrams daily for two weeks and then stop. "
    "the tablets go down after a meal, not before one. "
    "your last reading was 128 over 84 in the spring. "
    "we will see you again in six months to check on that. "
    "any swelling in the ankles since we spoke? none at all. "
    "good, then nothing else needs changing today."
)

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


def rerank_answerer(threshold: float, candidates: int = 5) -> RerankRelevanceAnswerer:
    return RerankRelevanceAnswerer(
        scheme=SCHEME,
        candidates=candidates,
        reranker=_RelevanceOf(),
        threshold=threshold,
    )


BLOOD_PRESSURE = "Was the blood pressure 135/88?"


def test_the_reranked_answerer_cites_the_chunk_the_judge_ranked_first():
    verdicts = list(
        rerank_answerer(threshold=0.1).answer(CONVERSATION, [BLOOD_PRESSURE])
    )

    assert verdicts[0].answer is True
    assert verdicts[0].evidence == verdicts[0].candidates[0].span
    assert "135/88" in verdicts[0].candidates[0].text


def test_every_sentence_is_scored_rather_than_a_retrieved_shortlist():
    """A lexical prefilter can only drop the right sentence; measured, it did."""
    scored = []

    class _Recording(_RelevanceOf):
        def rerank(self, question, chunks):
            scored.append(len(chunks))
            return super().rerank(question, chunks)

    answerer = RerankRelevanceAnswerer(
        scheme=SCHEME, candidates=5, reranker=_Recording(), threshold=0.1
    )
    list(answerer.answer(CONVERSATION, [BLOOD_PRESSURE]))

    assert scored == [len(chunk_conversation(CONVERSATION, SCHEME))]


def test_a_question_no_chunk_is_relevant_enough_for_is_answered_no_with_nulls():
    verdicts = list(
        rerank_answerer(threshold=0.9).answer(CONVERSATION, [BLOOD_PRESSURE])
    )

    assert verdicts[0].answer is False
    assert verdicts[0].evidence is None


def test_a_no_still_carries_the_candidates_it_was_judged_over():
    """The component metrics read them, and so will the Entailment judge."""
    verdicts = list(
        rerank_answerer(threshold=0.9).answer(CONVERSATION, [BLOOD_PRESSURE])
    )

    assert len(verdicts[0].candidates) == 5


def test_the_reranked_verdicts_are_produced_lazily():
    verdicts = rerank_answerer(threshold=0.1).answer(CONVERSATION, [BLOOD_PRESSURE] * 3)

    assert next(iter(verdicts)).answer is True
    assert isinstance(verdicts, Iterator)


def test_a_conversation_that_transcribed_to_nothing_is_answered_no():
    """A no, not a raise: one raise costs every Question about the Conversation."""
    [verdict] = list(rerank_answerer(threshold=0.1).answer((), [BLOOD_PRESSURE]))

    assert verdict.answer is False
    assert verdict.evidence is None
    assert verdict.candidates == ()


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
    entail_depth: int = 1,
) -> RerankEntailAnswerer:
    return RerankEntailAnswerer(
        scheme=SCHEME,
        candidates=5,
        reranker=_RelevanceOf(),
        rewriter=rewriter or _ClaimIs(),
        judge=judge,
        relevance_threshold=relevance_threshold,
        entailment_threshold=entailment_threshold,
        entail_depth=entail_depth,
    )


def ranked(question: str = BLOOD_PRESSURE) -> tuple[Chunk, ...]:
    """The candidates as Relevance ranks them, for a test to reach into."""
    return list(
        entail_answerer(_EntailmentOf({}, default=0.9)).answer(CONVERSATION, [question])
    )[0].candidates


def top_chunk(question: str = BLOOD_PRESSURE) -> str:
    """What the Relevance judge ranks first for one Question."""
    return ranked(question)[0].text


def test_a_chunk_that_entails_the_claim_is_answered_yes_and_cited():
    judge = _EntailmentOf({top_chunk(): 0.9})

    verdicts = list(entail_answerer(judge).answer(CONVERSATION, [BLOOD_PRESSURE]))

    assert verdicts[0].answer is True
    assert verdicts[0].evidence == verdicts[0].candidates[0].span


def test_a_relevant_chunk_that_does_not_entail_the_claim_is_answered_no():
    """The Hard Negative: the right subject, and the Conversation does not say it."""
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


def test_entailing_a_conversation_that_transcribed_to_nothing_is_answered_no():
    judge = _EntailmentOf({}, default=0.9)

    [verdict] = list(entail_answerer(judge).answer((), [BLOOD_PRESSURE]))

    assert verdict.answer is False


def test_settings_resolve_the_thresholds_the_entailing_answerer_is_built_with():
    answerer = build_answerer(
        Settings(answer_strategy="retrieve_rerank_entail", device="cpu")
    )

    assert isinstance(answerer, RerankEntailAnswerer)


class TestEntailDepth:
    """The cited Chunk is the one that entailed, not the one Relevance ranked
    first — the answer and the Evidence Span are one decision."""

    def test_only_the_top_relevant_chunk_is_judged_at_depth_one(self):
        judge = _EntailmentOf({}, default=0.9)

        list(
            entail_answerer(judge, entail_depth=1).answer(
                CONVERSATION, [BLOOD_PRESSURE]
            )
        )

        assert len(judge.judged[0]) == 1

    def test_the_configured_depth_of_chunks_is_judged(self):
        judge = _EntailmentOf({}, default=0.9)

        list(
            entail_answerer(judge, entail_depth=4).answer(
                CONVERSATION, [BLOOD_PRESSURE]
            )
        )

        assert len(judge.judged[0]) == 4

    def test_the_best_entailing_chunk_is_cited_rather_than_the_most_relevant(self):
        deeper = ranked()[2]
        judge = _EntailmentOf({deeper.text: 0.99}, default=0.6)

        verdict = list(
            entail_answerer(judge, entail_depth=4).answer(
                CONVERSATION, [BLOOD_PRESSURE]
            )
        )[0]

        assert verdict.answer is True
        assert verdict.cited == deeper
        assert verdict.evidence == deeper.span

    def test_the_candidates_stay_the_ranking_rather_than_being_reordered(self):
        """The span is read off the cited Chunk, so nothing has to be hoisted to
        make the two agree — and hoisting is what let them disagree."""
        deeper = ranked()[2]
        judge = _EntailmentOf({deeper.text: 0.99}, default=0.6)

        verdict = list(
            entail_answerer(judge, entail_depth=4).answer(
                CONVERSATION, [BLOOD_PRESSURE]
            )
        )[0]

        assert verdict.candidates == ranked()[:5]

    def test_a_chunk_deeper_down_can_carry_a_yes_the_top_chunk_would_not(self):
        judge = _EntailmentOf({ranked()[3].text: 0.8}, default=0.0)

        shallow = list(
            entail_answerer(judge, entail_depth=1).answer(
                CONVERSATION, [BLOOD_PRESSURE]
            )
        )[0]
        deep = list(
            entail_answerer(judge, entail_depth=4).answer(
                CONVERSATION, [BLOOD_PRESSURE]
            )
        )[0]

        assert shallow.answer is False
        assert deep.answer is True

    def test_no_chunk_entailing_is_still_a_no_however_deep_it_is_judged(self):
        judge = _EntailmentOf({}, default=0.0)

        verdict = list(
            entail_answerer(judge, entail_depth=5).answer(
                CONVERSATION, [BLOOD_PRESSURE]
            )
        )[0]

        assert verdict.answer is False
        assert verdict.evidence is None

    def test_a_depth_below_one_is_refused_rather_than_judging_nothing(self):
        with pytest.raises(ValueError, match="entail_depth"):
            entail_answerer(_EntailmentOf({}), entail_depth=0)
