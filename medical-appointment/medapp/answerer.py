"""The Answerer seam and the Answerer Settings names.

An Answerer is pure: Segments and Questions in, a Verdict per Question out. It
does not touch the wire, the clock or the audio, so it can be measured on a
fold of the supplied data exactly as it runs under the endpoint.

Candidate Answerers are compared by running the system twice under different
configuration. There is no runtime switch between them and no degraded Answerer
to fall back to, so a strategy without an implementation is a startup failure
rather than something quietly substituted.
"""

from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Protocol

from medapp.chunker import ChunkScheme, chunk_conversation
from medapp.claims import ClaimRewriter, Rewriter
from medapp.config import AnswerStrategy, Settings
from medapp.config import settings as default_settings
from medapp.dense import Embedder
from medapp.dense import load_model as load_embedder
from medapp.dense import warm_up as warm_embedder
from medapp.entailment import EntailmentJudge, NliEntailmentJudge
from medapp.reranker import CrossEncoderReranker, Reranker
from medapp.retrieval import Bm25Index, Retriever, build_index_factory
from medapp.types import Chunk, Segment, Verdict


class Answerer(Protocol):
    """Decides every Question asked about one Conversation."""

    def answer(
        self, segments: tuple[Segment, ...], questions: Sequence[str]
    ) -> Iterable[Verdict]:
        """Answer each Question against one Conversation's Segments.

        Args:
            segments: The whole Conversation, in time order.
            questions: The Questions, in the order they arrived.

        Returns:
            One Verdict per Question, in the same order. May be produced
            lazily: the caller consumes it one Verdict at a time so that a
            per-request deadline can be enforced between Questions.
        """
        ...


class CiteFirstSegmentAnswerer:
    """The tracer bullet: answers yes and cites the first Segment.

    Scores the floor on answers and close to nothing on evidence. It exists so
    the protocol seam runs end to end — and reports real latency — before any
    retrieval exists, and it is replaced, not extended.
    """

    def answer(
        self, segments: tuple[Segment, ...], questions: Sequence[str]
    ) -> Iterable[Verdict]:
        """Answer yes to every Question, citing the first Segment.

        Raises:
            ValueError: If the Conversation produced no Segments, which leaves
                a yes with nothing to cite.
        """
        if not segments:
            raise ValueError(
                "No Segments were transcribed, so a yes has nothing to cite."
            )

        chunk = Chunk(
            text=segments[0].text,
            start=segments[0].start,
            end=segments[0].end,
            words=segments[0].words,
        )

        return [
            Verdict(answer=True, evidence=chunk.span, candidates=(chunk,))
            for _ in questions
        ]


class Bm25RetrievalAnswerer:
    """ADR-0001's measured baseline: cut the Conversation up, rank, cite the best.

    It answers yes to every Question it retrieves anything for. Saying no needs
    a Relevance judgement and an Entailment judgement, and neither exists yet;
    inventing a score threshold here would be a guess dressed as a decision. So
    what this Answerer is for is the evidence half of the score and the
    component metrics behind it — every Verdict carries the ranked Chunks it
    was chosen from, which is what recall@k and oracle-selected tIoU are
    computed from.
    """

    def __init__(self, scheme: ChunkScheme, candidates: int) -> None:
        """Fix the granularities and how deep a ranking each Verdict carries.

        Args:
            scheme: The Chunk granularities, as Settings resolved them.
            candidates: How many ranked Chunks a Verdict carries. The judges
                downstream read them, and so do the component metrics, so this
                is deeper than the one Chunk the Evidence Span comes from.
        """
        self._scheme = scheme
        self._candidates = candidates

    def answer(
        self, segments: tuple[Segment, ...], questions: Sequence[str]
    ) -> Iterable[Verdict]:
        """Answer each Question against the Chunks of one Conversation.

        The Chunks and the index are built once for the whole Conversation and
        every Question is ranked against them, which is why the Verdicts are
        produced lazily: the caller checks its deadline between them.

        Raises:
            ValueError: If the Conversation produced no Chunks, which leaves
                every Question with nothing to read an answer from.
        """
        index = Bm25Index(chunk_conversation(segments, self._scheme))

        return self._verdicts(index, questions)

    def _verdicts(
        self, index: Bm25Index, questions: Sequence[str]
    ) -> Iterator[Verdict]:
        """One Verdict per Question, ranked as the caller consumes them."""
        for question in questions:
            candidates = index.rank(question, self._candidates)

            if not candidates:
                # The Question shares no term with the Conversation at all, so
                # there is no passage a yes could be read from.
                yield Verdict(answer=False, evidence=None, candidates=())
                continue

            yield Verdict(
                answer=True, evidence=candidates[0].span, candidates=candidates
            )


class RerankRelevanceAnswerer:
    """BM25 retrieves, a cross-encoder re-orders, and Relevance decides yes.

    Two things change against the BM25 baseline. The Evidence Span comes from
    the Chunk the cross-encoder ranked first rather than the one BM25 did,
    which is the boundary discrimination ADR-0002 recorded BM25 as lacking. And
    a Question whose best Chunk does not reach the Relevance threshold is
    answered no: nothing in the Conversation is about what it asks about, which
    is what an Off-Topic Question looks like.

    It answers yes to every Question that clears the threshold, Hard Negatives
    included. Their best Chunk scores high on Relevance precisely because it is
    lexically near-identical to the truth, and the judgement that separates
    them is Entailment, which does not exist yet. Lowering this threshold until
    it catches them would reject the Positives it is their business to keep.
    """

    def __init__(
        self,
        scheme: ChunkScheme,
        candidates: int,
        index_factory: Callable[[Sequence[Chunk]], Retriever],
        reranker: Reranker,
        threshold: float,
    ) -> None:
        """Fix the granularities, the retriever, the depth reranked and the bar.

        Args:
            scheme: The Chunk granularities, as Settings resolved them.
            candidates: How many ranked Chunks are rescored and carried on the
                Verdict. The judges downstream read them, and so do the
                component metrics.
            index_factory: Builds the per-request index over one Conversation's
                Chunks, as Settings' retrieval mode names it.
            reranker: Scores how far a Chunk is about what a Question asks
                about.
            threshold: The Relevance the best Chunk must reach for a yes.
        """
        self._scheme = scheme
        self._candidates = candidates
        self._index_factory = index_factory
        self._reranker = reranker
        self._threshold = threshold

    def answer(
        self, segments: tuple[Segment, ...], questions: Sequence[str]
    ) -> Iterable[Verdict]:
        """Answer each Question against the Chunks of one Conversation.

        Raises:
            ValueError: If the Conversation produced no Chunks, which leaves
                every Question with nothing to read an answer from.
        """
        index = self._index_factory(chunk_conversation(segments, self._scheme))

        return self._verdicts(index, questions)

    def _verdicts(
        self, index: Retriever, questions: Sequence[str]
    ) -> Iterator[Verdict]:
        """One Verdict per Question, judged as the caller consumes them."""
        for question in questions:
            retrieved = index.rank(question, self._candidates)
            reranked = self._reranker.rerank(question, retrieved)
            candidates = tuple(candidate.chunk for candidate in reranked)

            if not reranked or reranked[0].relevance < self._threshold:
                # Either the Question shares no term with the Conversation at
                # all, or nothing the Conversation says comes close enough to
                # what it asks about. Both are a no with nothing to point at.
                yield Verdict(answer=False, evidence=None, candidates=candidates)
                continue

            yield Verdict(
                answer=True, evidence=candidates[0].span, candidates=candidates
            )


class RerankEntailAnswerer:
    """Relevance finds the passage; Entailment decides whether it says so.

    The Relevance judge cannot separate a Positive from a Hard Negative — the
    Hard Negative's best Chunk is lexically near-identical to the truth and
    scores just as high — so it is asked only what it can answer: whether
    *anything* in the Conversation is about what the Question asks about. That
    is what an Off-Topic Question fails. Everything that clears it goes to the
    Entailment judge, which reads the Question as the Claim a yes would be
    agreeing with and decides whether the passage establishes it.

    CONTEXT.md fixes the rule and it is not a blend of the two scores: yes
    requires entailment, and nothing less. A Chunk that contradicts the Claim
    and a Chunk that is silent on it are both a no, so the entailment
    probability alone is thresholded.

    Only the top Relevant Chunk is judged. It is the Chunk the Evidence Span
    would be read from, so it is the one the yes has to be true of; judging
    deeper would let the answer come from a Chunk the Verdict does not cite.
    """

    def __init__(
        self,
        scheme: ChunkScheme,
        candidates: int,
        index_factory: Callable[[Sequence[Chunk]], Retriever],
        reranker: Reranker,
        rewriter: Rewriter,
        judge: EntailmentJudge,
        relevance_threshold: float | None,
        entailment_threshold: float,
    ) -> None:
        """Fix the granularities, the retriever, the depth reranked and the bars.

        Args:
            scheme: The Chunk granularities, as Settings resolved them.
            candidates: How many ranked Chunks are rescored and carried on the
                Verdict.
            index_factory: Builds the per-request index over one Conversation's
                Chunks, as Settings' retrieval mode names it.
            reranker: Scores how far a Chunk is about what a Question asks
                about.
            rewriter: Turns a Question into the Claim a yes would agree with.
            judge: Decides whether a Chunk establishes that Claim.
            relevance_threshold: The Relevance the best Chunk must reach before
                Entailment is judged at all, or None to judge every Question's
                best Chunk. None is the ablation
                ``python -m scripts.entailment_threshold`` measures the gate
                against.
            entailment_threshold: The entailment probability that Chunk must
                reach for a yes.
        """
        self._scheme = scheme
        self._candidates = candidates
        self._index_factory = index_factory
        self._reranker = reranker
        self._rewriter = rewriter
        self._judge = judge
        self._relevance_threshold = relevance_threshold
        self._entailment_threshold = entailment_threshold

    def answer(
        self, segments: tuple[Segment, ...], questions: Sequence[str]
    ) -> Iterable[Verdict]:
        """Answer each Question against the Chunks of one Conversation.

        Raises:
            ValueError: If the Conversation produced no Chunks, which leaves
                every Question with nothing to read an answer from.
        """
        index = self._index_factory(chunk_conversation(segments, self._scheme))

        return self._verdicts(index, questions)

    def _verdicts(
        self, index: Retriever, questions: Sequence[str]
    ) -> Iterator[Verdict]:
        """One Verdict per Question, judged as the caller consumes them."""
        for question in questions:
            reranked = self._reranker.rerank(
                question, index.rank(question, self._candidates)
            )
            candidates = tuple(candidate.chunk for candidate in reranked)

            if not reranked or (
                self._relevance_threshold is not None
                and reranked[0].relevance < self._relevance_threshold
            ):
                # Nothing the Conversation says comes close to what the
                # Question asks about, which is what an Off-Topic Question
                # looks like. There is no Claim worth judging against it.
                yield Verdict(answer=False, evidence=None, candidates=candidates)
                continue

            claim = self._rewriter.rewrite(question)
            judged = self._judge.judge(claim.text, candidates[:1])

            if judged[0].entailment < self._entailment_threshold:
                # The passage is about the right subject and does not establish
                # the Claim: it either contradicts it or is silent on it, and
                # both are a no. This is the Hard Negative.
                yield Verdict(answer=False, evidence=None, candidates=candidates)
                continue

            yield Verdict(
                answer=True, evidence=candidates[0].span, candidates=candidates
            )


def _build_bm25_retrieval(settings: Settings) -> Answerer:
    """The BM25 baseline, with the granularities Settings resolved.

    Raises:
        ValueError: If Settings name a retrieval mode this Answerer does not
            rank with. It is ADR-0001's BM25 baseline by definition, and
            serving it under the name of another mode would report BM25's
            numbers as that mode's.
    """
    if settings.retrieval_mode != "bm25":
        raise ValueError(
            f"The BM25 baseline Answerer ranks with BM25, and Settings name "
            f"the {settings.retrieval_mode!r} retrieval mode. Answer with the "
            "'retrieve_rerank_entail' strategy to use it."
        )

    return Bm25RetrievalAnswerer(
        scheme=ChunkScheme(
            word_lengths=settings.chunk_word_lengths,
            stride_fraction=settings.chunk_stride_fraction,
        ),
        candidates=settings.retrieval_candidates,
    )


def _build_rerank_relevance(settings: Settings) -> Answerer:
    """The reranked Answerer, with the weights loaded from the local cache.

    The cross-encoder is exercised once here rather than inside the first
    request: it is built at import, before the process is serving, and the
    first forward pass costs seconds a request does not have.
    """
    reranker = CrossEncoderReranker(settings)
    reranker.warm_up()

    return RerankRelevanceAnswerer(
        scheme=ChunkScheme(
            word_lengths=settings.chunk_word_lengths,
            stride_fraction=settings.chunk_stride_fraction,
        ),
        candidates=settings.retrieval_candidates,
        index_factory=build_index_factory(settings, _embedder_for(settings)),
        reranker=reranker,
        threshold=settings.relevance_threshold,
    )


def _build_rerank_entail(settings: Settings) -> Answerer:
    """The reranked Answerer with the Entailment judgement behind it.

    Every model is exercised once here rather than inside the first request,
    for the reason the reranker alone already was: the first forward pass costs
    seconds a request does not have. The embedder is loaded only where Settings
    name a retrieval mode that ranks with one, so the mode ADR-0001's
    measurement rejected costs the request path no memory and no latency.
    """
    reranker = CrossEncoderReranker(settings)
    reranker.warm_up()

    judge = NliEntailmentJudge(settings)
    judge.warm_up()

    return RerankEntailAnswerer(
        scheme=ChunkScheme(
            word_lengths=settings.chunk_word_lengths,
            stride_fraction=settings.chunk_stride_fraction,
        ),
        candidates=settings.retrieval_candidates,
        index_factory=build_index_factory(settings, _embedder_for(settings)),
        reranker=reranker,
        rewriter=ClaimRewriter(),
        judge=judge,
        relevance_threshold=(
            settings.relevance_threshold if settings.relevance_gate else None
        ),
        entailment_threshold=settings.entailment_threshold,
    )


def _embedder_for(settings: Settings) -> Embedder | None:
    """The bi-encoder, loaded and warmed, or None where no mode ranks with one."""
    if settings.retrieval_mode == "bm25":
        return None

    embedder = load_embedder(settings)
    warm_embedder(embedder, settings)

    return embedder


_ANSWERERS: dict[AnswerStrategy, Callable[[Settings], Answerer]] = {
    "cite_first_segment": lambda _: CiteFirstSegmentAnswerer(),
    "retrieve_bm25": _build_bm25_retrieval,
    "retrieve_rerank": _build_rerank_relevance,
    "retrieve_rerank_entail": _build_rerank_entail,
}


def build_answerer(settings: Settings | None = None) -> Answerer:
    """Construct the Answerer named by Settings.

    Args:
        settings: The resolved environment. Defaults to the process-wide
            Settings.

    Raises:
        NotImplementedError: If the named strategy has no implementation yet.
    """
    settings = settings or default_settings
    strategy = settings.answer_strategy

    if strategy not in _ANSWERERS:
        raise NotImplementedError(
            f"No Answerer implements the {strategy!r} strategy yet; "
            f"MEDAPP_ANSWER_STRATEGY must name one of "
            f"{sorted(_ANSWERERS)}."
        )

    return _ANSWERERS[strategy](settings)
