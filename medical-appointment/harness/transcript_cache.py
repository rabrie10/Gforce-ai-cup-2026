"""The development transcript cache: one JSON file per Conversation.

ADR-0002 pays the ASR cost once so that every later tuning iteration runs
offline. The cache belongs to the evaluation harness and to it alone — nothing
under ``medapp`` imports this module, because a cached transcript reaching the
request path would silently make the dev loop's shortcut into production
behaviour.

Two properties make it safe to tune against. The on-disk form round-trips: what
:func:`read` returns equals what the Transcriber produced, Word timings
included. And every file records the decoding settings that produced it, which
are checked on the way back in — a transcript decoded by a smaller model or a
different beam size is refused rather than quietly measured, because a stale
cache is invisible in every number computed from it.
"""

import json
import os
from pathlib import Path
from typing import Any

from medapp.config import Settings
from medapp.config import settings as default_settings
from medapp.types import Segment, Word

# Bumped when the on-disk shape changes, so a stale cache is refused rather
# than half-read.
CACHE_FORMAT_VERSION = 1

# The Settings fields that change what the Transcriber produces. A cache
# decoded under different values is a different measurement.
DECODING_FIELDS = (
    "whisper_model",
    "whisper_language",
    "beam_size",
    "vad_filter",
    "condition_on_previous_text",
)


def decoding_provenance(settings: Settings) -> dict[str, Any]:
    """The decoding settings a cached transcript is only valid under."""
    return {field: getattr(settings, field) for field in DECODING_FIELDS}


def cache_file(transcript_id: str, cache_dir: Path | None = None) -> Path:
    """Where one Conversation's transcript is cached."""
    return (
        cache_dir or default_settings.transcript_cache_dir
    ) / f"{transcript_id}.json"


def is_cached(
    transcript_id: str,
    cache_dir: Path | None = None,
    settings: Settings | None = None,
) -> bool:
    """Whether this Conversation is cached *under these decoding settings*.

    A file decoded under other settings counts as uncached, so changing the
    model or a decoding knob re-transcribes rather than silently reusing
    transcripts the current configuration would never produce.
    """
    source = cache_file(transcript_id, cache_dir)

    if not source.exists():
        return False

    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False

    return bool(
        document.get("format_version") == CACHE_FORMAT_VERSION
        and document.get("decoding")
        == decoding_provenance(settings or default_settings)
    )


def write(
    transcript_id: str,
    segments: tuple[Segment, ...],
    cache_dir: Path | None = None,
    settings: Settings | None = None,
) -> Path:
    """Cache one Conversation's Segments.

    Written through a temporary file in the same directory and moved into
    place, because an interrupted run resumes from what is on disk: a half
    written file would read as a cached Conversation for good.

    Args:
        transcript_id: The id the supplied questions use, e.g. ``sample_4``.
        segments: The Segments the Transcriber produced.
        cache_dir: Where to write. Defaults to the configured cache.
        settings: The settings the Transcriber decoded under, recorded with the
            transcript. Defaults to the process-wide Settings.

    Returns:
        The file written.
    """
    destination = cache_file(transcript_id, cache_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)

    document = {
        "format_version": CACHE_FORMAT_VERSION,
        "transcript_id": transcript_id,
        "decoding": decoding_provenance(settings or default_settings),
        "segments": [_segment_as_dict(segment) for segment in segments],
    }

    partial = destination.with_suffix(".json.partial")
    partial.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(partial, destination)

    return destination


def read(
    transcript_id: str,
    cache_dir: Path | None = None,
    settings: Settings | None = None,
) -> tuple[Segment, ...]:
    """The cached Segments for one Conversation.

    Args:
        transcript_id: The id the supplied questions use.
        cache_dir: Where to read from. Defaults to the configured cache.
        settings: The settings the transcript must have been decoded under.
            Defaults to the process-wide Settings.

    Raises:
        FileNotFoundError: If the Conversation has not been transcribed. The
            cache is filled by ``scripts/transcribe_cache.py``; a measurement
            that quietly skipped a Conversation would be reported over the
            wrong sample.
        ValueError: If the file was written by a different cache format or
            under different decoding settings.
    """
    source = cache_file(transcript_id, cache_dir)

    if not source.exists():
        raise FileNotFoundError(
            f"No cached transcript at {source}. Fill the cache with "
            "`python -m scripts.transcribe_cache`."
        )

    document = json.loads(source.read_text(encoding="utf-8"))

    version = document.get("format_version")
    if version != CACHE_FORMAT_VERSION:
        raise ValueError(
            f"{source} is cache format {version!r}, not {CACHE_FORMAT_VERSION}. "
            "Delete the cache and re-transcribe."
        )

    expected = decoding_provenance(settings or default_settings)
    if document.get("decoding") != expected:
        raise ValueError(
            f"{source} was decoded with {document.get('decoding')!r}, but the "
            f"current configuration is {expected!r}. Re-transcribe with "
            "`python -m scripts.transcribe_cache`; measuring one against the "
            "other would report the wrong Transcriber's numbers."
        )

    return tuple(_segment_from_dict(segment) for segment in document["segments"])


def cached_transcript_ids(cache_dir: Path | None = None) -> tuple[str, ...]:
    """Every Conversation with a file in the cache, sorted."""
    directory = cache_dir or default_settings.transcript_cache_dir

    if not directory.exists():
        return ()

    return tuple(sorted(path.stem for path in directory.glob("*.json")))


def _segment_as_dict(segment: Segment) -> dict[str, Any]:
    return {
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
        "words": [
            {
                "text": word.text,
                "start": word.start,
                "end": word.end,
                "probability": word.probability,
            }
            for word in segment.words
        ],
    }


def _segment_from_dict(document: dict[str, Any]) -> Segment:
    return Segment(
        start=document["start"],
        end=document["end"],
        text=document["text"],
        words=tuple(
            Word(
                text=word["text"],
                start=word["start"],
                end=word["end"],
                probability=word["probability"],
            )
            for word in document["words"]
        ),
    )
