from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Protocol

from medapp.chunker import SentenceScheme, chunk_conversation
from medapp.claims import ClaimRewriter, Rewriter
from medapp.config import AnswerStrategy, Settings
from medapp.config import settings as default_settings
from medapp.entailment import EntailmentJudge, NliEntailmentJudge
from medapp.reranker import CrossEncoderReranker, Reranker
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
            Verdict(answer=True, cited=chunk, candidates=(chunk,)) for _ in questions
        ]


class RerankRelevanceAnswerer:
    """The cross-encoder ranks every sentence, and Relevance decides yes.

    There is no retriever in front of the cross-encoder. One Conversation cuts
    into a few dozen sentences, which is a batch the cross-encoder scores in
    one pass, and every lexical prefilter measured over train and dev cost
    tIoU rather than saving it: at a BM25 depth of 8, 16, 24 and 32 the cited
    Chunk is worth 0.461, 0.477, 0.487 and 0.488 mean tIoU against 0.507 for
    scoring all of them. A prefilter can only remove the right sentence, and on
    a candidate set this small it buys no latency worth that.

    A Question whose best sentence does not reach the Relevance threshold is
    answered no: nothing in the Conversation is about what it asks about, which
    is what an Off-Topic Question looks like.
    """

    def __init__(
        self,
        scheme: SentenceScheme,
        candidates: int,
        reranker: Reranker,
        threshold: float,
    ) -> None:
        """Fix how the Conversation is cut, how deep a ranking is carried, and the bar.

        Args:
            scheme: How the Conversation is cut, as Settings resolved it.
            candidates: How many ranked Chunks a Verdict carries. Every
                sentence is scored; this is how many of them the Verdict
                exposes for the judges downstream and the component metrics.
            reranker: Scores how far a Chunk is about what a Question asks
                about.
            threshold: The Relevance the best Chunk must reach for a yes.
        """
        self._scheme = scheme
        self._candidates = candidates
        self._reranker = reranker
        self._threshold = threshold

    def answer(
        self, segments: tuple[Segment, ...], questions: Sequence[str]
    ) -> Iterable[Verdict]:
        """Answer each Question against the sentences of one Conversation."""
        return self._verdicts(chunk_conversation(segments, self._scheme), questions)

    def _verdicts(
        self, chunks: tuple[Chunk, ...], questions: Sequence[str]
    ) -> Iterator[Verdict]:
        """One Verdict per Question, judged as the caller consumes them."""
        for question in questions:
            reranked = self._reranker.rerank(question, chunks)
            candidates = tuple(
                candidate.chunk for candidate in reranked[: self._candidates]
            )

            if not reranked or reranked[0].relevance < self._threshold:
                # Nothing the Conversation says comes close enough to what the
                # Question asks about, which is a no with nothing to point at.
                yield Verdict(answer=False, cited=None, candidates=candidates)
                continue

            yield Verdict(answer=True, cited=candidates[0], candidates=candidates)


class RerankEntailAnswerer:
    """Relevance finds the passage; Entailment decides whether it says so.

    The Relevance judge is asked what it can answer: whether *anything* in the
    Conversation is about what the Question asks about, which is what an
    Off-Topic Question fails. Everything that clears it goes to the Entailment
    judge, which reads the Question as the Claim a yes would be agreeing with
    and decides whether the passage establishes it.

    CONTEXT.md fixes the rule and it is not a blend of the two scores: yes
    requires entailment, and nothing less. A Chunk that contradicts the Claim
    and a Chunk that is silent on it are both a no, so the entailment
    probability alone is thresholded.

    The top Relevant Chunks are judged and the Verdict cites the one that
    entailed best, which is what keeps the answer and the Evidence Span one
    decision rather than two.
    """

    def __init__(
        self,
        scheme: SentenceScheme,
        candidates: int,
        reranker: Reranker,
        rewriter: Rewriter,
        judge: EntailmentJudge,
        relevance_threshold: float | None,
        entailment_threshold: float,
        entail_depth: int,
    ) -> None:
        """Fix how the Conversation is cut, the depth carried, and the bars.

        Args:
            scheme: How the Conversation is cut, as Settings resolved it.
            candidates: How many ranked Chunks the Verdict carries.
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
            entail_depth: How many of the top Relevant Chunks are judged. The
                Verdict cites the best-entailing of them.

        Raises:
            ValueError: If fewer than one Chunk would be judged, which leaves
                every Question with no Entailment judgement to answer from.
        """
        if entail_depth < 1:
            raise ValueError(
                f"At least one Chunk has to be judged for a yes to be "
                f"entailed by anything: got entail_depth={entail_depth!r}."
            )

        self._scheme = scheme
        self._candidates = candidates
        self._reranker = reranker
        self._rewriter = rewriter
        self._judge = judge
        self._relevance_threshold = relevance_threshold
        self._entailment_threshold = entailment_threshold
        self._entail_depth = entail_depth

    def answer(
        self, segments: tuple[Segment, ...], questions: Sequence[str]
    ) -> Iterable[Verdict]:
        """Answer each Question against the sentences of one Conversation."""
        return self._verdicts(chunk_conversation(segments, self._scheme), questions)

    def _verdicts(
        self, chunks: tuple[Chunk, ...], questions: Sequence[str]
    ) -> Iterator[Verdict]:
        """One Verdict per Question, judged as the caller consumes them."""
        for question in questions:
            reranked = self._reranker.rerank(question, chunks)
            candidates = tuple(
                candidate.chunk for candidate in reranked[: self._candidates]
            )

            if not reranked or (
                self._relevance_threshold is not None
                and reranked[0].relevance < self._relevance_threshold
            ):
                # Nothing the Conversation says comes close to what the
                # Question asks about, which is what an Off-Topic Question
                # looks like. There is no Claim worth judging against it.
                yield Verdict(answer=False, cited=None, candidates=candidates)
                continue

            claim = self._rewriter.rewrite(question)
            judged = self._judge.judge(claim.text, candidates[: self._entail_depth])
            best = max(judged, key=lambda judgement: judgement.entailment)

            if best.entailment < self._entailment_threshold:
                # No passage the Conversation is about the right subject in
                # establishes the Claim: each either contradicts it or is
                # silent on it, and both are a no. This is the Hard Negative.
                yield Verdict(answer=False, cited=None, candidates=candidates)
                continue

            yield Verdict(answer=True, cited=best.chunk, candidates=candidates)


def _scheme(settings: Settings) -> SentenceScheme:
    """How the Conversation is cut, as Settings resolved it."""
    return SentenceScheme(max_words=settings.chunk_max_words)


def _build_rerank_relevance(settings: Settings) -> Answerer:
    """The reranked Answerer, with the weights loaded from the local cache.

    The cross-encoder is exercised once here rather than inside the first
    request: it is built at import, before the process is serving, and the
    first forward pass costs seconds a request does not have.
    """
    reranker = CrossEncoderReranker(settings)
    reranker.warm_up()

    return RerankRelevanceAnswerer(
        scheme=_scheme(settings),
        candidates=settings.retrieval_candidates,
        reranker=reranker,
        threshold=settings.relevance_threshold,
    )


def _build_rerank_entail(settings: Settings) -> Answerer:
    """The reranked Answerer with the Entailment judgement behind it.

    Every model is exercised once here rather than inside the first request,
    for the reason the reranker alone already was: the first forward pass costs
    seconds a request does not have.
    """
    reranker = CrossEncoderReranker(settings)
    reranker.warm_up()

    judge = NliEntailmentJudge(settings)
    judge.warm_up()

    return RerankEntailAnswerer(
        scheme=_scheme(settings),
        candidates=settings.retrieval_candidates,
        reranker=reranker,
        rewriter=ClaimRewriter(),
        judge=judge,
        relevance_threshold=(
            settings.relevance_threshold if settings.relevance_gate else None
        ),
        entailment_threshold=settings.entailment_threshold,
        entail_depth=settings.entail_depth,
    )


_ANSWERERS: dict[AnswerStrategy, Callable[[Settings], Answerer]] = {
    "cite_first_segment": lambda _: CiteFirstSegmentAnswerer(),
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
