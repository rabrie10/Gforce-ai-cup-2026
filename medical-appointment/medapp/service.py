import logging
import math
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

    def transcribe(
        self, audio_bytes: bytes, budget_seconds: float | None = None
    ) -> tuple[Segment, ...]: ...


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

        return scorable_response(
            ASRQuestionResponseDto(
                answers=[answer for answer, _ in answered],
                evidence_start=[span[0] if span else None for _, span in answered],
                evidence_end=[span[1] if span else None for _, span in answered],
            ),
            expected_count=len(questions),
            filename=request.audio_filename,
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

        # Transcription is the one stage long enough to carry the request past
        # the service's timeout on its own, and a deadline checked only after
        # it returns is checked too late to help. So it is handed a budget: the
        # time left, less what the ten Questions will need.
        segments = self._transcriber.transcribe(
            audio_bytes,
            budget_seconds=max(
                0.0,
                deadline - self._clock() - self._settings.answering_reserve_seconds,
            ),
        )

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


def scorable_response(
    response: ASRQuestionResponseDto, expected_count: int, filename: str
) -> ASRQuestionResponseDto:
    """The body, repaired to something the evaluator can read.

    ``utils.validate_response`` raises on a body the evaluator would refuse,
    which is the right thing in a test and the wrong thing in the request path:
    a raised exception becomes a 500, and a 500 and an unparseable body are
    scored identically — every Question about this Conversation wrong. Repairing
    is therefore strictly better than either, because the repaired body still
    carries whatever the Answerer did decide, and a guess is worth half a mark.

    Everything above this function is written so that it has nothing to do. It
    is here for the case that reasoning is wrong, and it logs at error level
    when it fires so that the bug is not hidden by the recovery.

    Args:
        response: The body as it was assembled.
        expected_count: How many Questions arrived.
        filename: The Conversation, for the log line.

    Returns:
        The body unchanged where it was already scorable, and otherwise one of
        the right shape: truncated or padded with a guess to the Question count,
        with any interval the evaluator would score zero dropped to None.
    """
    answers = _fitted(list(response.answers or []), expected_count, _GUESS[0])
    starts = _fitted(list(response.evidence_start or []), expected_count, None)
    ends = _fitted(list(response.evidence_end or []), expected_count, None)

    repaired = ASRQuestionResponseDto(
        answers=[bool(answer) for answer in answers],
        evidence_start=list(starts),
        evidence_end=list(ends),
    )

    for position, (start, end) in enumerate(zip(starts, ends, strict=True)):
        if not _scorable_interval(start, end):
            repaired.evidence_start[position] = None
            repaired.evidence_end[position] = None

    if (
        repaired.answers != list(response.answers or [])
        or repaired.evidence_start != list(response.evidence_start or [])
        or repaired.evidence_end != list(response.evidence_end or [])
    ):
        logger.error(
            "%s: the assembled body was not scorable and was repaired to %d "
            "Questions; this is a bug above the guard, not a recovery that "
            "should ever be needed.",
            filename,
            expected_count,
        )

    return repaired


def _fitted(values: list, expected_count: int, filler: object) -> list:
    """The list cut or padded to exactly ``expected_count`` entries."""
    return values[:expected_count] + [filler] * max(0, expected_count - len(values))


def _scorable_interval(start: object, end: object) -> bool:
    """Whether the evaluator would read this interval rather than score it zero.

    A half-filled interval and an end before its start both score nothing, and
    so does anything that is not a finite number of seconds.
    """
    if start is None and end is None:
        return True

    if start is None or end is None:
        return False

    if isinstance(start, bool) or isinstance(end, bool):
        return False

    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return False

    return math.isfinite(start) and math.isfinite(end) and end >= start
