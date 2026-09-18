"""The Answerer seam and the Answerer Settings names.

An Answerer is pure: Segments and Questions in, a Verdict per Question out. It
does not touch the wire, the clock or the audio, so it can be measured on a
fold of the supplied data exactly as it runs under the endpoint.

Candidate Answerers are compared by running the system twice under different
configuration. There is no runtime switch between them and no degraded Answerer
to fall back to, so a strategy without an implementation is a startup failure
rather than something quietly substituted.
"""

from collections.abc import Iterable, Sequence
from typing import Protocol

from medapp.config import AnswerStrategy, Settings
from medapp.config import settings as default_settings
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


_ANSWERERS: dict[AnswerStrategy, type[Answerer]] = {
    "cite_first_segment": CiteFirstSegmentAnswerer,
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

    return _ANSWERERS[strategy]()
