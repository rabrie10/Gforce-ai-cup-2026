"""The protocol seam: a valid body every time, whatever the Answerer does.

Transcription and answering are both stubbed. What is under test is the guard —
that an exception, a missed deadline or an unusual question count still produces
a body the evaluator can score, with the answers in the order the Questions
arrived.
"""

import threading

import pytest

from dtos import ASRQuestionRequestDto
from medapp.service import PredictionService
from medapp.types import Chunk, Segment, Verdict, Word
from utils import encode_audio, validate_response


def _segment(text: str, start: float, end: float) -> Segment:
    return Segment(
        start=start,
        end=end,
        text=text,
        words=(Word(text=text, start=start, end=end, probability=0.9),),
    )


SEGMENTS = (
    _segment("Take two tablets.", 0.0, 1.4),
    _segment("For six weeks.", 1.6, 2.6),
)


class StubTranscriber:
    """Returns fixed Segments, recording the audio it was handed."""

    def __init__(self, segments: tuple[Segment, ...] = SEGMENTS) -> None:
        self.segments = segments
        self.calls: list[bytes] = []

    def transcribe(self, audio_bytes: bytes) -> tuple[Segment, ...]:
        self.calls.append(audio_bytes)

        return self.segments


class StubAnswerer:
    """Answers each Question from a scripted list of Verdicts."""

    def __init__(self, verdicts: list[Verdict]) -> None:
        self.verdicts = verdicts
        self.calls: list[tuple[tuple[Segment, ...], list[str]]] = []

    def answer(self, segments, questions):
        self.calls.append((segments, list(questions)))

        return list(self.verdicts)


def _chunk(segment: Segment) -> Chunk:
    return Chunk(
        text=segment.text,
        start=segment.start,
        end=segment.end,
        words=segment.words,
    )


def _yes(segment: Segment) -> Verdict:
    chunk = _chunk(segment)

    return Verdict(answer=True, evidence=chunk.span, candidates=(chunk,))


def _no() -> Verdict:
    return Verdict(answer=False, evidence=None, candidates=())


def _request(questions: list[str]) -> ASRQuestionRequestDto:
    return ASRQuestionRequestDto(
        audio_base64=encode_audio(b"mp3 bytes"),
        audio_filename="conversation_sample_4.mp3",
        questions=questions,
    )


def test_every_question_is_answered_in_the_order_it_arrived():
    questions = ["Two tablets?", "Six weeks?", "Any penicillin?"]
    answerer = StubAnswerer([_yes(SEGMENTS[0]), _yes(SEGMENTS[1]), _no()])
    service = PredictionService(StubTranscriber(), answerer)

    response = service.predict(_request(questions))

    validate_response(response, expected_count=len(questions))
    assert response.answers == [True, True, False]
    assert response.evidence_start == [0.0, 1.6, None]
    assert response.evidence_end == [1.4, 2.6, None]
    assert answerer.calls == [(SEGMENTS, questions)]


class ExplodingTranscriber:
    """Fails the way a decoder fails on audio it cannot read."""

    def transcribe(self, audio_bytes: bytes) -> tuple[Segment, ...]:
        raise RuntimeError("the decoder gave up")


def test_a_failure_before_transcription_guesses_every_question(caplog):
    questions = ["Two tablets?", "Six weeks?"]
    service = PredictionService(ExplodingTranscriber(), StubAnswerer([]))

    with caplog.at_level("ERROR"):
        response = service.predict(_request(questions))

    validate_response(response, expected_count=len(questions))
    assert response.answers == [True, True]
    assert response.evidence_start == [None, None]
    assert response.evidence_end == [None, None]
    assert "the decoder gave up" in caplog.text
    assert caplog.records and all(r.levelname == "ERROR" for r in caplog.records)


def test_no_log_record_carries_the_request_body(caplog):
    """audio_base64 and question text are request-body content, never logged."""
    questions = ["Two tablets?", "Six weeks?"]
    request = _request(questions)
    answerer = FailingOnOneQuestionAnswerer(0)
    service = PredictionService(StubTranscriber(), answerer)

    with caplog.at_level("INFO"):
        service.predict(request)

    for record in caplog.records:
        message = record.getMessage()
        assert request.audio_base64 not in message
        for question in questions:
            assert question not in message


@pytest.mark.parametrize("count", [0, 1, 7, 13])
def test_the_body_carries_one_entry_per_question_whatever_the_count(count):
    questions = [f"Question {i}?" for i in range(count)]
    answerer = StubAnswerer([_yes(SEGMENTS[0])] * count)
    service = PredictionService(StubTranscriber(), answerer)

    response = service.predict(_request(questions))

    validate_response(response, expected_count=count)


def test_an_answerer_that_stops_early_has_its_remaining_questions_guessed(caplog):
    questions = ["Two tablets?", "Six weeks?", "Any penicillin?"]
    answerer = StubAnswerer([_no()])
    service = PredictionService(StubTranscriber(), answerer)

    with caplog.at_level("ERROR"):
        response = service.predict(_request(questions))

    validate_response(response, expected_count=len(questions))
    assert response.answers == [False, True, True]
    assert caplog.records and all(r.levelname == "ERROR" for r in caplog.records)


class FailingOnOneQuestionAnswerer:
    """Answers until one Question raises, the way a lazy Answerer fails."""

    def __init__(self, failing_position: int) -> None:
        self.failing_position = failing_position

    def answer(self, segments, questions):
        for position, _ in enumerate(questions):
            if position == self.failing_position:
                raise RuntimeError("the reranker ran out of memory")

            yield _no()


def test_a_question_that_raises_is_guessed_and_logged(caplog):
    questions = ["Two tablets?", "Six weeks?", "Any penicillin?"]
    service = PredictionService(StubTranscriber(), FailingOnOneQuestionAnswerer(1))

    with caplog.at_level("ERROR"):
        response = service.predict(_request(questions))

    validate_response(response, expected_count=len(questions))
    assert response.answers == [False, True, True]
    assert response.evidence_start == [None, None, None]
    assert "the reranker ran out of memory" in caplog.text
    assert "Six weeks?" not in caplog.text, "question text is request-body content"
    assert caplog.records and all(r.levelname == "ERROR" for r in caplog.records)


class FakeClock:
    """A monotonic clock that advances a fixed step per reading."""

    def __init__(self, step: float) -> None:
        self.step = step
        self.now = 0.0

    def __call__(self) -> float:
        reading = self.now
        self.now += self.step

        return reading


def _settings(**overrides):
    from medapp.config import Settings

    return Settings(**overrides)


def test_a_deadline_reached_between_questions_guesses_the_rest(caplog):
    questions = [f"Question {i}?" for i in range(10)]
    answerer = StubAnswerer([_no()] * 10)
    service = PredictionService(
        StubTranscriber(),
        answerer,
        settings=_settings(deadline_seconds=4.0),
        clock=FakeClock(step=1.0),
    )

    with caplog.at_level("ERROR"):
        response = service.predict(_request(questions))

    validate_response(response, expected_count=len(questions))
    assert response.answers[-1] is True
    assert response.evidence_start[-1] is None
    assert False in response.answers, "the questions answered before the deadline stand"
    assert "deadline" in caplog.text
    assert caplog.records and all(r.levelname == "ERROR" for r in caplog.records)


def test_a_deadline_reached_before_transcription_guesses_every_question(caplog):
    questions = ["Two tablets?", "Six weeks?"]
    transcriber = StubTranscriber()
    service = PredictionService(
        transcriber,
        StubAnswerer([_no(), _no()]),
        settings=_settings(deadline_seconds=0.5),
        clock=FakeClock(step=1.0),
    )

    with caplog.at_level("ERROR"):
        response = service.predict(_request(questions))

    validate_response(response, expected_count=len(questions))
    assert response.answers == [True, True]
    assert transcriber.calls == [], "transcription is not started past the deadline"
    assert "deadline" in caplog.text


class BlockingTranscriber:
    """Records when it is inside, and stays there until released."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.inside = threading.Event()
        self.release = threading.Event()
        self._guard = threading.Lock()

    def transcribe(self, audio_bytes: bytes) -> tuple[Segment, ...]:
        with self._guard:
            self.events.append("enter")

        self.inside.set()
        self.release.wait(timeout=5)

        with self._guard:
            self.events.append("exit")

        return SEGMENTS


def test_two_concurrent_requests_are_transcribed_one_after_the_other():
    transcriber = BlockingTranscriber()
    service = PredictionService(transcriber, StubAnswerer([_no()]))
    request = _request(["Two tablets?"])

    first = threading.Thread(target=service.predict, args=(request,))
    first.start()
    assert transcriber.inside.wait(timeout=5), "the first request never started"

    second = threading.Thread(target=service.predict, args=(request,))
    second.start()
    # The second request is now either blocked on the lock or, if inference is
    # not serialized, already inside the transcriber alongside the first.
    second.join(timeout=0.2)

    transcriber.release.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert transcriber.events == ["enter", "exit", "enter", "exit"]
