"""The one Transcriber: faster-whisper, configured entirely from Settings.

Not an interface and not a set of interchangeable backends. The only realistic
change is model size, device or a decoding knob, and configuration already
covers all three — which is why this module never asks what hardware it is on.

Word timestamps come from the same decoding pass as the Segments, so the timings
the Chunker builds on cost an alignment step rather than a second traversal of
the audio. Segments are kept apart: joining them into one string throws away the
only thing that makes an Evidence Span returnable.
"""

import io
from collections.abc import Iterable
from typing import Protocol

from faster_whisper import WhisperModel

from medapp.config import Settings
from medapp.config import settings as default_settings
from medapp.types import Segment, Word


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
    ) -> None:
        """Load the decoding model, unless one is supplied.

        Args:
            settings: The resolved environment. Defaults to the process-wide
                Settings.
            model: An already-constructed decoding model. The argument exists so
                tests and the dev-loop scripts can hold one model across many
                Conversations rather than reloading weights per call.
        """
        self._settings = settings or default_settings
        self._model = model if model is not None else _load_model(self._settings)

    @property
    def settings(self) -> Settings:
        """The configuration this Transcriber decodes under."""
        return self._settings

    def transcribe(self, audio_bytes: bytes) -> tuple[Segment, ...]:
        """Transcribe one Conversation.

        Segments the decoder produced no word timings for are dropped: an
        Evidence Span is returned as word boundaries, so a Segment without them
        cannot back one, and carrying it forward would put text in the index
        that no answer could ever point at.

        Args:
            audio_bytes: The MP3 as it arrived on the wire.

        Returns:
            The Segments in time order, each carrying its Words.
        """
        decoded, _ = self._model.transcribe(
            io.BytesIO(audio_bytes),
            language=self._settings.whisper_language,
            beam_size=self._settings.beam_size,
            vad_filter=self._settings.vad_filter,
            condition_on_previous_text=self._settings.condition_on_previous_text,
            word_timestamps=True,
        )

        # faster-whisper yields segments lazily; decoding only runs as they are
        # consumed.
        return tuple(_as_segment(segment) for segment in decoded if segment.words)


def _load_model(settings: Settings) -> WhisperModel:
    """Construct the decoding model from Settings alone."""
    return WhisperModel(
        settings.whisper_model,
        device=settings.device,
        compute_type=settings.compute_type,
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
