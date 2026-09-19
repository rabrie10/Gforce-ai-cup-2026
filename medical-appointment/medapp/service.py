"""The protocol seam: one Conversation in, a scorable body out, every time.

The endpoint's contract is unusual in that a raised exception is the one
outcome with no partial credit: the evaluator scores every Question about the
Conversation wrong, not just the one that failed, and a guess is worth half a
mark on average. So every stage here is wrapped, and every guard activation is
logged at error level with its reason.
"""

import logging
import threading
import time
from collections.abc import Callable
from typing import Protocol

from dtos import ASRQuestionRequestDto, ASRQuestionResponseDto
from medapp.answerer import Answerer
from medapp.config import Settings
from medapp.config import settings as default_settings
from medapp.types import Segment
from utils import Span, audio_duration_seconds, decode_audio

logger = logging.getLogger(__name__)

# One row of the response: the answer and the passage behind it, if any.
_Answered = tuple[bool, Span | None]

# What a Question is answered with when the guard fires. Yes is the guess with
# the better expectation — both splits are balanced between yes and no — and a
# guess has no passage behind it, so it points at nothing.
_GUESS: _Answered = (True, None)


class DeadlineExceeded(Exception):
    """Raised when a request has spent its per-request budget."""


class Transcriber(Protocol):
    """The part of the Transcriber this module uses."""

    def transcribe(self, audio_bytes: bytes) -> tuple[Segment, ...]: ...


class PredictionService:
    """Answers one Conversation per request, behind the protocol guard."""

    def __init__(
        self,
        transcriber: Transcriber,
        answerer: Answerer,
        settings: Settings | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Wire the seam.

        Args:
            transcriber: Turns the Conversation's audio into Segments.
            answerer: Decides the Questions against those Segments.
            settings: The resolved environment. Defaults to the process-wide
                Settings.
            clock: Monotonic source the per-request deadline is measured on.
        """
        self._transcriber = transcriber
        self._answerer = answerer
        self._settings = settings or default_settings
        self._clock = clock
        self._lock = threading.Lock()

    def predict(self, request: ASRQuestionRequestDto) -> ASRQuestionResponseDto:
        """Answer every Question about one Conversation.

        Never raises. An exception would cost every Question about this
        Conversation rather than the one that failed, so each stage is guarded
        and anything unanswered falls back to a guess.
        """
        deadline = self._clock() + self._settings.deadline_seconds
        questions = request.questions
        answered: list[_Answered] = []

        # Inference is serialized: the models are sized for one Conversation at
        # a time, and two requests decoding at once would put both over the
        # deadline rather than one.
        with self._lock:
            try:
                answered = self._answer_conversation(request, deadline)
            except Exception:
                logger.error(
                    "%s: guessing every Question — the request failed before "
                    "any Question was answered.",
                    request.audio_filename,
                    exc_info=True,
                )

        if len(answered) < len(questions):
            logger.error(
                "%s: guessing %d of %d Questions the guard left unanswered.",
                request.audio_filename,
                len(questions) - len(answered),
                len(questions),
            )
            answered += [_GUESS] * (len(questions) - len(answered))

        return ASRQuestionResponseDto(
            answers=[answer for answer, _ in answered],
            evidence_start=[span[0] if span else None for _, span in answered],
            evidence_end=[span[1] if span else None for _, span in answered],
        )

    def _answer_conversation(
        self, request: ASRQuestionRequestDto, deadline: float
    ) -> list[_Answered]:
        """Transcribe once, then answer each Question against the Segments."""
        audio_bytes = decode_audio(request.audio_base64)
        duration = audio_duration_seconds(audio_bytes)

        logger.info(
            "%s (%.1f s, %.1f MB): %d questions",
            request.audio_filename,
            duration if duration is not None else float("nan"),
            len(audio_bytes) / 1e6,
            len(request.questions),
        )

        self._check_deadline(deadline, request.audio_filename, "decoding the audio")

        segments = self._transcriber.transcribe(audio_bytes)

        self._check_deadline(deadline, request.audio_filename, "transcription")

        return self._answer_questions(segments, request, deadline)

    def _answer_questions(
        self,
        segments: tuple[Segment, ...],
        request: ASRQuestionRequestDto,
        deadline: float,
    ) -> list[_Answered]:
        """Consume the Answerer one Verdict at a time, under the deadline.

        The Verdicts are taken lazily so the deadline is checked between
        Questions. An Answerer that raises part-way through cannot be resumed,
        so the Questions after the failure are left to the guess in
        :meth:`predict` rather than retried.
        """
        answered: list[_Answered] = []
        verdicts = iter(self._answerer.answer(segments, request.questions))

        for position in range(len(request.questions)):
            try:
                self._check_deadline(
                    deadline, request.audio_filename, f"question {position}"
                )
                verdict = next(verdicts)
            except StopIteration:
                logger.error(
                    "%s: the Answerer returned %d Verdicts for %d Questions.",
                    request.audio_filename,
                    position,
                    len(request.questions),
                )
                break
            except Exception:
                # The question text is request-body content and never logged;
                # its position identifies it without it.
                logger.error(
                    "%s: guessing from question %d onwards.",
                    request.audio_filename,
                    position,
                    exc_info=True,
                )
                break

            answered.append((verdict.answer, verdict.evidence))

        return answered

    def _check_deadline(self, deadline: float, filename: str, stage: str) -> None:
        """Raise if this request has run out of its budget.

        Raises:
            DeadlineExceeded: If the deadline has passed.
        """
        remaining = deadline - self._clock()

        if remaining <= 0:
            raise DeadlineExceeded(
                f"{filename}: the {self._settings.deadline_seconds:g} s deadline "
                f"passed {-remaining:.1f} s ago, at {stage}."
            )
