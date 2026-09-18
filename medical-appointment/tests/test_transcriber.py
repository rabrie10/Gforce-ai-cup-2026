"""The Transcriber is configured from Settings and keeps its Segments apart.

Decoding real audio costs minutes per Conversation, so the model is stubbed:
what is under test is that every decoding knob reaches the decoder from
Settings, that Words survive with their timings, and that a Segment the decoder
gave no word timings for is dropped rather than carried forward.
"""

from dataclasses import dataclass

from medapp.config import Settings
from medapp.transcriber import Transcriber


@dataclass
class _DecodedWord:
    start: float
    end: float
    word: str
    probability: float


@dataclass
class _DecodedSegment:
    start: float
    end: float
    text: str
    words: list[_DecodedWord] | None


class _RecordingModel:
    """A decoder that records its options and returns what it was handed."""

    def __init__(self, segments: list[_DecodedSegment]) -> None:
        self.segments = segments
        self.options: dict[str, object] = {}
        self.audio: bytes | None = None

    def transcribe(self, audio, **options):
        self.audio = audio.read()
        self.options = options

        return iter(self.segments), object()


def _segments() -> list[_DecodedSegment]:
    return [
        _DecodedSegment(
            start=0.0,
            end=1.4,
            text=" Take two tablets.",
            words=[
                _DecodedWord(start=0.0, end=0.5, word=" Take", probability=0.98),
                _DecodedWord(start=0.5, end=0.9, word=" two", probability=0.91),
                _DecodedWord(start=0.9, end=1.4, word=" tablets.", probability=0.87),
            ],
        ),
        _DecodedSegment(
            start=1.6,
            end=2.2,
            text=" Twice daily.",
            words=[
                _DecodedWord(start=1.6, end=2.2, word=" Twice daily.", probability=0.8)
            ],
        ),
    ]


def test_every_decoding_knob_comes_from_settings():
    model = _RecordingModel(_segments())
    settings = Settings(
        beam_size=3,
        vad_filter=False,
        condition_on_previous_text=True,
        whisper_language="en",
    )

    Transcriber(settings=settings, model=model).transcribe(b"mp3")

    assert model.options == {
        "language": "en",
        "beam_size": 3,
        "vad_filter": False,
        "condition_on_previous_text": True,
        "word_timestamps": True,
    }


def test_the_audio_reaches_the_decoder_unchanged():
    model = _RecordingModel(_segments())

    Transcriber(settings=Settings(), model=model).transcribe(b"raw mp3 bytes")

    assert model.audio == b"raw mp3 bytes"


def test_segments_stay_apart_and_carry_their_words():
    model = _RecordingModel(_segments())

    segments = Transcriber(settings=Settings(), model=model).transcribe(b"mp3")

    assert len(segments) == 2
    assert segments[0].text == " Take two tablets."
    assert [word.text for word in segments[0].words] == [" Take", " two", " tablets."]
    assert segments[0].words[1].start == 0.5
    assert segments[0].words[1].end == 0.9
    assert segments[0].words[1].probability == 0.91
    assert segments[1].words[0].text == " Twice daily."


def test_a_segment_without_word_timings_is_dropped_rather_than_carried():
    """An Evidence Span is returned as word boundaries.

    A Segment the decoder gave no timings for could never back one, and the
    decoder emits them: keeping it would put text in the index that no answer
    can point at, and raising would cost the whole Conversation.
    """
    model = _RecordingModel(
        [
            _DecodedSegment(start=0.0, end=1.0, text=" Hello.", words=None),
            _DecodedSegment(start=1.0, end=1.5, text=" Mm.", words=[]),
            *_segments(),
        ]
    )

    segments = Transcriber(settings=Settings(), model=model).transcribe(b"mp3")

    assert [segment.text for segment in segments] == [
        " Take two tablets.",
        " Twice daily.",
    ]
