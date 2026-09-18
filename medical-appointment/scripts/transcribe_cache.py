"""Transcribe every supplied Conversation once into the git-ignored cache.

ADR-0002 pays the ASR cost once. On the development machine CTranslate2 has no
Metal backend, so this is CPU-bound and runs for hours; it is meant to. Every
tuning iteration afterwards reads the cache and never touches audio.

    python -m scripts.transcribe_cache

Conversations already cached are skipped, so an interrupted run resumes where
it stopped. ``--force`` re-transcribes everything. A changed model or decoding
knob needs no flag: the cache records what it was decoded with, so transcripts
the current configuration would not produce count as uncached and are redone.
"""

import argparse
import time

from harness import transcript_cache
from medapp.config import settings
from medapp.transcriber import Transcriber
from utils import (
    audio_filename_for_transcript,
    load_sample_audio,
    load_sample_questions,
)


def supplied_transcript_ids() -> tuple[str, ...]:
    """Every Conversation in the supplied data, sorted."""
    return tuple(sorted({row["transcript_id"] for row in load_sample_questions()}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-transcribe Conversations that are already cached.",
    )
    arguments = parser.parse_args()

    transcript_ids = supplied_transcript_ids()
    outstanding = [
        transcript_id
        for transcript_id in transcript_ids
        if arguments.force or not transcript_cache.is_cached(transcript_id)
    ]

    print(
        f"{len(transcript_ids)} Conversations, {len(outstanding)} to transcribe "
        f"with {settings.whisper_model} on {settings.device}/"
        f"{settings.compute_type} (beam {settings.beam_size}, "
        f"vad_filter={settings.vad_filter})."
    )

    if not outstanding:
        return

    # One model, held across every Conversation: reloading the weights per file
    # would dominate the run.
    transcriber = Transcriber()

    for position, transcript_id in enumerate(outstanding, start=1):
        started = time.monotonic()
        audio = load_sample_audio(audio_filename_for_transcript(transcript_id))
        segments = transcriber.transcribe(audio)
        destination = transcript_cache.write(
            transcript_id, segments, settings=transcriber.settings
        )
        elapsed = time.monotonic() - started

        words = sum(len(segment.words) for segment in segments)
        print(
            f"  [{position:>2}/{len(outstanding)}] {transcript_id:<12} "
            f"{len(segments):>4} segments {words:>6} words {elapsed:>7.1f}s "
            f"-> {destination.name}"
        )


if __name__ == "__main__":
    main()
