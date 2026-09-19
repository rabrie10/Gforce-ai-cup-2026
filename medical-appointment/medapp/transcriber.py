import array
import io
import logging
import math
import time
import wave
from collections.abc import Callable, Iterable, Iterator
from typing import Protocol

from faster_whisper import WhisperModel

from medapp.config import Settings, resolved_cpu_threads
from medapp.config import settings as default_settings
from medapp.types import Segment, Word

logger = logging.getLogger(__name__)


class _DecodedWord(Protocol):
    """The shape faster-whisper returns a word in."""

    start: float
    end: float
    word: str
    probability: float


class _DecodedSegment(Protocol):
    """The shape faster-whisper returns a segment in."""

    start: float
    end: float
    text: str
    words: list[_DecodedWord] | None


class _DecodingModel(Protocol):
    """The part of ``WhisperModel`` this module uses."""

    def transcribe(
        self, audio: io.BytesIO, **decoding_options: object
    ) -> tuple[Iterable[_DecodedSegment], object]: ...


class Transcriber:
    """Turns a Conversation's MP3 bytes into timed Segments carrying Words."""

    def __init__(
        self,
        settings: Settings | None = None,
        model: _DecodingModel | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Load the decoding model, unless one is supplied.

        Args:
            settings: The resolved environment. Defaults to the process-wide
                Settings.
            model: An already-constructed decoding model. The argument exists so
                tests and the dev-loop scripts can hold one model across many
                Conversations rather than reloading weights per call.
            clock: Monotonic source the decoding budget is measured on.
        """
        self._settings = settings or default_settings
        self._model = model if model is not None else _load_model(self._settings)
        self._clock = clock

    @property
    def settings(self) -> Settings:
        """The configuration this Transcriber decodes under."""
        return self._settings

    def transcribe(
        self, audio_bytes: bytes, budget_seconds: float | None = None
    ) -> tuple[Segment, ...]:
        """Transcribe one Conversation, within a decoding budget.

        Segments the decoder produced no word timings for are dropped: an
        Evidence Span is returned as word boundaries, so a Segment without them
        cannot back one, and carrying it forward would put text in the index
        that no answer could ever point at.

        Args:
            audio_bytes: The MP3 as it arrived on the wire.
            budget_seconds: How long decoding may run for. None decodes the
                whole Conversation however long it takes, which is what the
                dev-loop scripts want and what a request must never do.

        Returns:
            The Segments decoded within the budget, in time order, each
            carrying its Words. A budget that runs out truncates the
            Conversation rather than raising: the Questions are still answered,
            against the opening of it.
        """
        decoded, _ = self._model.transcribe(
            io.BytesIO(audio_bytes), **self._decoding_options()
        )

        # faster-whisper yields segments lazily; decoding only runs as they are
        # consumed, which is what makes the budget below a budget on decoding
        # rather than a timer running beside it.
        return tuple(
            _as_segment(segment)
            for segment in self._within(decoded, budget_seconds)
            if segment.words
        )

    def _within(
        self, decoded: Iterable[_DecodedSegment], budget_seconds: float | None
    ) -> Iterator[_DecodedSegment]:
        """Yield decoded segments until the budget runs out, then stop.

        The budget is checked before each segment is pulled rather than after,
        so the check cannot itself be what takes the request past its deadline.
        """
        if budget_seconds is None:
            yield from decoded
            return

        deadline = self._clock() + budget_seconds
        iterator = iter(decoded)

        while self._clock() < deadline:
            try:
                yield next(iterator)
            except StopIteration:
                return

        logger.error(
            "Decoding stopped at its %.1f s budget; the Conversation is "
            "transcribed only as far as the decoder reached.",
            budget_seconds,
        )

    def warm_up(self) -> None:
        """Decode one synthetic second so no request pays the first decode.

        Weights are loaded lazily and the first decode allocates the decoder's
        state, both of which would otherwise land inside a request that has
        seconds to spare. The voice-activity filter is off here because the
        synthetic audio carries no speech and would otherwise be skipped,
        leaving the decoder untouched.
        """
        decoded, _ = self._model.transcribe(
            io.BytesIO(_warm_up_audio()), **self._decoding_options(vad_filter=False)
        )

        tuple(decoded)

    def _decoding_options(self, vad_filter: bool | None = None) -> dict[str, object]:
        """The decoding knobs, every one of them resolved from Settings."""
        return {
            "language": self._settings.whisper_language,
            "beam_size": self._settings.beam_size,
            "vad_filter": (
                self._settings.vad_filter if vad_filter is None else vad_filter
            ),
            "condition_on_previous_text": self._settings.condition_on_previous_text,
            "word_timestamps": True,
        }


def _load_model(settings: Settings) -> WhisperModel:
    """Construct the decoding model from Settings alone."""
    return WhisperModel(
        settings.whisper_model,
        device=settings.device,
        compute_type=settings.compute_type,
        cpu_threads=resolved_cpu_threads(settings.cpu_threads),
        download_root=str(settings.model_cache_dir),
    )


def _as_segment(decoded: _DecodedSegment) -> Segment:
    """Convert one decoded segment, which must carry words, into a Segment.

    Timings are coerced to plain floats: faster-whisper returns NumPy scalars,
    which compare equal but do not serialise, and the transcript cache is JSON.
    """
    assert decoded.words is not None

    return Segment(
        start=float(decoded.start),
        end=float(decoded.end),
        text=decoded.text,
        words=tuple(
            Word(
                text=word.word,
                start=float(word.start),
                end=float(word.end),
                probability=float(word.probability),
            )
            for word in decoded.words
        ),
    )


WARM_UP_SAMPLE_RATE_HZ = 16_000
WARM_UP_SECONDS = 1.0
WARM_UP_TONE_HZ = 220.0


def _warm_up_audio() -> bytes:
    """One second of tone as a WAV, for :meth:`Transcriber.warm_up`.

    Synthesised rather than shipped so the warm-up does not depend on a sample
    file surviving into the container.
    """
    sample_count = int(WARM_UP_SAMPLE_RATE_HZ * WARM_UP_SECONDS)
    radians_per_sample = 2 * math.pi * WARM_UP_TONE_HZ / WARM_UP_SAMPLE_RATE_HZ
    samples = array.array(
        "h",
        (int(8000 * math.sin(radians_per_sample * i)) for i in range(sample_count)),
    )

    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(WARM_UP_SAMPLE_RATE_HZ)
        wav.writeframes(samples.tobytes())

    return buffer.getvalue()
