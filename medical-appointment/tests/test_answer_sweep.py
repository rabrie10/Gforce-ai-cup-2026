"""The joint sweep replays the shipped Answerer, and chooses under a slice floor.

Two things are under test and the first matters most: the replay in
`harness.answer_sweep.outcome` has to be the same rule
`medapp.answerer.RerankEntailAnswerer` applies, because a rule that drifts from
it would tune a system that is not the one served. The second is the choosing —
the lower bound, the tie band and the slice floor that stops a candidate buying
mean tIoU by giving up the discrimination the case is about.
"""

from dataclasses import replace

import pytest

from harness.answer_sweep import (
    Candidate,
    CandidateReport,
    Recording,
    chosen,
    grid,
    outcome,
    paddings,
    sweep,
)
from medapp.span_refiner import SpanPadding, WordEdges
from medapp.types import Chunk, Word

WORDS = tuple(
    Word(text=text, start=start, end=start + 0.4, probability=0.9)
    for text, start in (
        ("blood", 0.0),
        ("pressure", 0.4),
        ("is", 0.8),
        ("135/88", 1.2),
        ("today", 1.6),
    )
)
EDGES = WordEdges.of(WORDS)


def _chunk(first: int, last: int) -> Chunk:
    words = WORDS[first : last + 1]

    return Chunk(
        text=" ".join(word.text for word in words),
        start=words[0].start,
        end=words[-1].end,
        words=words,
    )


CHUNKS = (_chunk(0, 1), _chunk(2, 3), _chunk(3, 4))


def _recording(
    entailments: tuple[float, ...] = (0.9, 0.1, 0.1),
    relevance: float = 0.8,
    truth: bool = True,
    gold: tuple[float, float] | None = (0.8, 1.6),
    question_type: str = "positive",
    transcript_id: str = "sample_4",
) -> Recording:
    return Recording(
        question_id="q",
        transcript_id=transcript_id,
        question_type=question_type,
        truth=truth,
        gold=gold,
        relevance=relevance,
        chunks=CHUNKS,
        entailments=entailments,
        edges=EDGES,
    )


def _candidate(**overrides) -> Candidate:
    return Candidate(
        **{
            "relevance_threshold": 0.3,
            "entailment_threshold": 0.5,
            "entail_depth": 1,
            "padding": SpanPadding(),
            **overrides,
        }
    )


class TestTheReplay:
    def test_a_chunk_that_entails_is_answered_yes_and_cited(self):
        answered = outcome(_recording(entailments=(0.9,)), _candidate())

        assert answered.answer is True
        assert answered.predicted == (CHUNKS[0].start, CHUNKS[0].end)

    def test_a_chunk_that_does_not_entail_is_answered_no_with_no_span(self):
        answered = outcome(_recording(entailments=(0.1,)), _candidate())

        assert answered.answer is False
        assert answered.predicted is None

    def test_relevance_below_the_gate_is_a_no_however_well_it_entails(self):
        answered = outcome(_recording(relevance=0.1, entailments=(0.99,)), _candidate())

        assert answered.answer is False

    def test_the_gate_can_be_ablated_away(self):
        answered = outcome(
            _recording(relevance=0.1, entailments=(0.99,)),
            _candidate(relevance_threshold=None),
        )

        assert answered.answer is True

    def test_depth_one_reads_only_the_top_chunk(self):
        answered = outcome(
            _recording(entailments=(0.1, 0.99, 0.99)), _candidate(entail_depth=1)
        )

        assert answered.answer is False, "the Chunk that entails was not judged"

    def test_a_deeper_chunk_can_carry_a_yes_and_is_the_one_cited(self):
        answered = outcome(
            _recording(entailments=(0.1, 0.99, 0.2)), _candidate(entail_depth=3)
        )

        assert answered.answer is True
        assert answered.predicted == (
            CHUNKS[1].start,
            CHUNKS[1].end,
        ), "the span comes from the Chunk that entailed, not the most relevant"

    def test_the_span_is_padded_and_snapped_to_a_word_edge(self):
        answered = outcome(
            _recording(entailments=(0.9,)),
            _candidate(padding=SpanPadding(start_seconds=0.5)),
        )

        assert answered.predicted[0] in {word.start for word in WORDS}

    def test_a_question_whose_truth_is_no_carries_no_tiou(self):
        answered = outcome(
            _recording(truth=False, gold=None, question_type="hard_negative"),
            _candidate(),
        )

        assert answered.tiou is None


class TestTheGrid:
    def test_it_is_the_product_of_every_knob(self):
        candidates = grid(
            relevance_thresholds=(0.1, 0.2),
            entailment_thresholds=(0.5,),
            depths=(1, 2, 3),
            paddings=(SpanPadding(), SpanPadding(start_seconds=1.0)),
        )

        assert len(candidates) == 2 * 1 * 3 * 2
        assert len(set(candidates)) == len(candidates)

    def test_paddings_sweep_both_ends_independently(self):
        assert len(paddings((0.0, 1.0), (0.0, 0.5))) == 2 * 2 * 2 * 2


class TestChoosing:
    def _report(
        self, lower: float, missed: int, hard: float = 0.9, resampled: bool = True
    ) -> CandidateReport:
        return CandidateReport(
            candidate=_candidate(entail_depth=missed + 1),
            score=lower + 0.04,
            score_interval=(lower, lower + 0.08),
            accuracy=0.87,
            accuracy_by_type={"positive": 0.9, "hard_negative": hard},
            mean_tiou=0.45,
            missed_positives=missed,
            resampled=resampled,
        )

    def test_it_chooses_on_the_lower_bound_rather_than_the_point_estimate(self):
        steady = self._report(lower=0.60, missed=1)
        lucky = replace(self._report(lower=0.50, missed=1), score=0.99)

        assert chosen([lucky, steady]) is steady

    def test_candidates_within_the_tie_band_are_separated_by_missed_positives(self):
        sloppy = self._report(lower=0.6004, missed=5)
        careful = self._report(lower=0.6000, missed=1)

        assert chosen([sloppy, careful]) is careful

    def test_a_difference_wider_than_the_tie_band_still_decides(self):
        better = self._report(lower=0.65, missed=5)
        careful = self._report(lower=0.60, missed=1)

        assert chosen([better, careful]) is better

    def test_a_candidate_that_regresses_a_question_type_is_refused(self):
        reference = self._report(lower=0.55, missed=2, hard=0.86)
        buys_tiou_with_hard_negatives = self._report(lower=0.70, missed=1, hard=0.70)
        holds_the_slice = self._report(lower=0.60, missed=1, hard=0.85)

        assert (
            chosen(
                [buys_tiou_with_hard_negatives, holds_the_slice],
                reference=reference,
            )
            is holds_the_slice
        )

    def test_a_regression_inside_the_floor_is_allowed(self):
        reference = self._report(lower=0.55, missed=2, hard=0.86)
        slightly_worse = self._report(lower=0.70, missed=1, hard=0.84)

        assert chosen([slightly_worse], reference=reference) is slightly_worse

    def test_no_reference_applies_no_floor(self):
        collapses_a_slice = self._report(lower=0.70, missed=1, hard=0.1)

        assert chosen([collapses_a_slice]) is collapses_a_slice

    def test_a_candidate_never_resampled_is_not_eligible(self):
        assert (
            chosen(
                [
                    self._report(lower=0.9, missed=0, resampled=False),
                    self._report(lower=0.5, missed=3),
                ]
            ).score_interval[0]
            == 0.5
        )

    def test_choosing_with_nothing_resampled_is_refused(self):
        with pytest.raises(ValueError, match="resampling"):
            chosen([self._report(lower=0.6, missed=1, resampled=False)])

    def test_every_candidate_regressing_a_slice_is_refused_rather_than_chosen(self):
        reference = self._report(lower=0.55, missed=2, hard=0.9)

        with pytest.raises(ValueError, match="regresses"):
            chosen([self._report(lower=0.7, missed=1, hard=0.1)], reference=reference)


class TestTheSweep:
    def test_only_the_shortlist_carries_a_measured_interval(self):
        recordings = [
            _recording(transcript_id="sample_4"),
            _recording(transcript_id="sample_9"),
        ]
        candidates = grid(
            relevance_thresholds=(0.1, 0.5),
            entailment_thresholds=(0.2, 0.8),
            depths=(1,),
            paddings=(SpanPadding(),),
        )

        reports = sweep(recordings, candidates, shortlist=2)

        assert [report.resampled for report in reports] == [True, True, False, False]

    def test_reports_come_back_best_first(self):
        recordings = [_recording()]
        reports = sweep(
            recordings,
            grid(
                relevance_thresholds=(0.1, 0.9),
                entailment_thresholds=(0.5,),
                depths=(1,),
                paddings=(SpanPadding(),),
            ),
            shortlist=0,
        )

        assert reports[0].score >= reports[-1].score

    def test_sweeping_nothing_is_refused(self):
        with pytest.raises(ValueError, match="nothing recorded"):
            sweep([], [_candidate()])


class TestTheReplayAgreesWithTheShippedAnswerer:
    """The one property that matters: tuning a rule that is not the rule served
    would choose knobs for a system nobody runs."""

    def test_every_verdict_matches_the_answerer_over_the_same_chunks(self):
        from medapp.answerer import RerankEntailAnswerer, SpanRefiningAnswerer
        from medapp.chunker import chunk_conversation
        from medapp.retrieval import Bm25Index
        from tests.test_answerer import (
            CONVERSATION,
            SCHEME,
            _ClaimIs,
            _EntailmentOf,
            _RelevanceOf,
        )

        index = Bm25Index(chunk_conversation(CONVERSATION, SCHEME))
        words = tuple(word for segment in CONVERSATION for word in segment.words)

        questions = [
            "Was the blood pressure 135/88?",
            "Is there any mention of a concert?",
            "Was the dose two tablets?",
        ]
        padding = SpanPadding(start_seconds=0.3, end_seconds=0.2, start_fraction=-0.1)
        reranker, rewriter = _RelevanceOf(), _ClaimIs()

        for depth, threshold, gate in ((1, 0.5, 0.1), (3, 0.5, 0.1), (3, 0.9, None)):
            judge = _EntailmentOf({}, default=0.7)
            served = SpanRefiningAnswerer(
                RerankEntailAnswerer(
                    scheme=SCHEME,
                    candidates=5,
                    index_factory=Bm25Index,
                    reranker=reranker,
                    rewriter=rewriter,
                    judge=judge,
                    relevance_threshold=gate,
                    entailment_threshold=threshold,
                    entail_depth=depth,
                ),
                padding,
            )
            verdicts = list(served.answer(CONVERSATION, questions))

            candidate = Candidate(
                relevance_threshold=gate,
                entailment_threshold=threshold,
                entail_depth=depth,
                padding=padding,
            )

            for question, verdict in zip(questions, verdicts, strict=True):
                reranked = reranker.rerank(question, index.rank(question, 5))
                chunks = tuple(scored.chunk for scored in reranked)
                judged = judge.judge(rewriter.rewrite(question).text, chunks[:depth])
                replayed = outcome(
                    Recording(
                        question_id="q",
                        transcript_id="sample_4",
                        question_type="positive",
                        truth=True,
                        gold=(0.0, 1.0),
                        relevance=reranked[0].relevance if reranked else 0.0,
                        chunks=chunks,
                        entailments=tuple(j.entailment for j in judged),
                        edges=WordEdges.of(words),
                    ),
                    candidate,
                )

                assert replayed.answer == verdict.answer
                assert replayed.predicted == verdict.evidence
