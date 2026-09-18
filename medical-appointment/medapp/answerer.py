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
from medapp.config import AnswerStrategy, Settings
from medapp.config import settings as default_settings
from medapp.retrieval import Bm25Index
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


def _build_bm25_retrieval(settings: Settings) -> Answerer:
    """The BM25 baseline, with the granularities Settings resolved."""
    return Bm25RetrievalAnswerer(
        scheme=ChunkScheme(
            word_lengths=settings.chunk_word_lengths,
            stride_fraction=settings.chunk_stride_fraction,
        ),
        candidates=settings.retrieval_candidates,
    )


_ANSWERERS: dict[AnswerStrategy, Callable[[Settings], Answerer]] = {
    "cite_first_segment": lambda _: CiteFirstSegmentAnswerer(),
    "retrieve_bm25": _build_bm25_retrieval,
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
