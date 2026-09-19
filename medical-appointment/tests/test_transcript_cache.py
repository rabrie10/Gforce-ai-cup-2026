"""The cache round-trips Segments, and stays out of the request path.

Round-tripping is what makes the cache usable as a stand-in for the Transcriber
in every later measurement: if reading back changed a Word timing, tuning would
run against something the deployed system never produces.
"""

import ast
import json
from pathlib import Path

import pytest

import medapp
from harness import transcript_cache
from medapp.config import Settings
from medapp.types import Segment, Word

SEGMENTS = (
    Segment(
        start=0.0,
        end=1.4,
        text=" Take two tablets.",
        words=(
            Word(text=" Take", start=0.0, end=0.5, probability=0.98),
            Word(text=" two", start=0.5, end=0.9, probability=0.91),
            Word(text=" tablets.", start=0.9, end=1.4, probability=0.8712345),
        ),
    ),
    Segment(
        start=1.6,
        end=2.2,
        text=" Twice daily.",
        words=(Word(text=" Twice daily.", start=1.6, end=2.2, probability=0.8),),
    ),
)


def test_written_segments_read_back_identical(tmp_path):
    transcript_cache.write("sample_4", SEGMENTS, cache_dir=tmp_path)

    assert transcript_cache.read("sample_4", cache_dir=tmp_path) == SEGMENTS


def test_an_uncached_conversation_is_reported_as_missing(tmp_path):
    assert not transcript_cache.is_cached("sample_4", cache_dir=tmp_path)

    with pytest.raises(FileNotFoundError, match="scripts.transcribe_cache"):
        transcript_cache.read("sample_4", cache_dir=tmp_path)


def test_a_written_conversation_is_reported_as_cached(tmp_path):
    transcript_cache.write("sample_4", SEGMENTS, cache_dir=tmp_path)

    assert transcript_cache.is_cached("sample_4", cache_dir=tmp_path)
    assert transcript_cache.cached_transcript_ids(cache_dir=tmp_path) == ("sample_4",)


def test_a_cache_written_in_another_format_is_refused(tmp_path):
    written = transcript_cache.write("sample_4", SEGMENTS, cache_dir=tmp_path)
    document = json.loads(written.read_text(encoding="utf-8"))
    document["format_version"] = transcript_cache.CACHE_FORMAT_VERSION + 1
    written.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="cache format"):
        transcript_cache.read("sample_4", cache_dir=tmp_path)


def test_the_cache_records_the_settings_it_was_decoded_under(tmp_path):
    written = transcript_cache.write(
        "sample_4", SEGMENTS, cache_dir=tmp_path, settings=Settings(beam_size=3)
    )

    document = json.loads(written.read_text(encoding="utf-8"))

    assert document["transcript_id"] == "sample_4"
    assert document["decoding"]["beam_size"] == 3
    assert document["decoding"]["whisper_model"] == Settings().whisper_model


def test_a_transcript_decoded_under_other_settings_is_refused(tmp_path):
    """A stale cache is invisible in every number computed from it."""
    transcript_cache.write(
        "sample_4", SEGMENTS, cache_dir=tmp_path, settings=Settings(beam_size=3)
    )
    current = Settings(beam_size=5)

    assert not transcript_cache.is_cached(
        "sample_4", cache_dir=tmp_path, settings=current
    )

    with pytest.raises(ValueError, match="decoded with"):
        transcript_cache.read("sample_4", cache_dir=tmp_path, settings=current)


def test_an_interrupted_write_leaves_no_file_to_mistake_for_a_transcript(tmp_path):
    """The script is built to be resumed, so a half-written file must not count.

    ``json.dumps`` raises before anything is moved into place, which is the
    failure a cancelled or out-of-disk run takes.
    """
    unserialisable = (
        Segment(
            start=0.0,
            end=1.0,
            text=" Hm.",
            words=(Word(text=" Hm.", start=0.0, end=1.0, probability=object()),),  # type: ignore[arg-type]
        ),
    )

    with pytest.raises(TypeError):
        transcript_cache.write("sample_4", unserialisable, cache_dir=tmp_path)

    assert not transcript_cache.is_cached("sample_4", cache_dir=tmp_path)


def test_nothing_under_medapp_imports_the_harness_or_its_cache():
    """The dev loop's shortcut must not become production behaviour.

    ADR-0002 keeps the cache in the evaluation harness. An import from
    ``medapp`` would put a cached transcript one refactor away from the request
    path, where there is no cache and never will be.
    """
    package = Path(medapp.__file__).parent

    offenders = []
    for module in package.rglob("*.py"):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue

            if any(name.split(".")[0] in {"harness", "scripts"} for name in names):
                offenders.append(f"{module.name}: {names}")

    assert offenders == []
