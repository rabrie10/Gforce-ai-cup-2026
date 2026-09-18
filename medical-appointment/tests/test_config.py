"""Settings is the only module that reads the environment: defaults resolve
without it, and every value the deployment VM overrides is reachable through it.
"""

import os

import pytest
from pydantic import ValidationError

from medapp.config import Settings


def _clear_medapp_environment(monkeypatch):
    """Drops every MEDAPP_ variable the developer's shell may export."""
    for name in list(os.environ):
        if name.startswith("MEDAPP_"):
            monkeypatch.delenv(name)


def test_defaults_resolve_without_any_environment(monkeypatch):
    _clear_medapp_environment(monkeypatch)

    settings = Settings()

    assert settings.device == "cpu"
    assert settings.compute_type == "int8"
    assert settings.answer_strategy == "retrieve_rerank"
    assert settings.relevance_threshold == 0.3
    assert settings.chunk_word_lengths == (1, 2, 3, 4, 6, 8, 11, 15, 20, 27, 36, 48)
    assert settings.chunk_stride_fraction == 0.2
    assert settings.retrieval_candidates == 10
    assert settings.route_suffix == ""
    assert settings.deadline_seconds == 50.0
    assert settings.transcript_cache_dir.name == "transcripts"
    assert settings.whisper_language == "en"
    assert settings.vad_filter is True
    assert settings.condition_on_previous_text is False
    assert settings.beam_size == 5


def test_environment_overrides_every_deployment_value(monkeypatch):
    _clear_medapp_environment(monkeypatch)
    monkeypatch.setenv("MEDAPP_DEVICE", "cuda")
    monkeypatch.setenv("MEDAPP_COMPUTE_TYPE", "float16")
    monkeypatch.setenv("MEDAPP_WHISPER_MODEL", "large-v3-turbo")
    monkeypatch.setenv("MEDAPP_ANSWER_STRATEGY", "single_llm")
    monkeypatch.setenv("MEDAPP_DEADLINE_SECONDS", "42.5")
    monkeypatch.setenv("MEDAPP_MODEL_CACHE_DIR", "/weights")
    monkeypatch.setenv("MEDAPP_VAD_FILTER", "false")
    monkeypatch.setenv("MEDAPP_CONDITION_ON_PREVIOUS_TEXT", "true")
    monkeypatch.setenv("MEDAPP_BEAM_SIZE", "1")
    monkeypatch.setenv("MEDAPP_WHISPER_LANGUAGE", "de")
    monkeypatch.setenv("MEDAPP_ROUTE_SUFFIX", "-a1b2c3")

    settings = Settings()

    assert settings.device == "cuda"
    assert settings.compute_type == "float16"
    assert settings.whisper_model == "large-v3-turbo"
    assert settings.answer_strategy == "single_llm"
    assert settings.deadline_seconds == 42.5
    assert str(settings.model_cache_dir) == "/weights"
    assert settings.vad_filter is False
    assert settings.condition_on_previous_text is True
    assert settings.beam_size == 1
    assert settings.whisper_language == "de"
    assert settings.route_suffix == "-a1b2c3"


def test_deadline_must_stay_strictly_under_the_service_timeout(monkeypatch):
    _clear_medapp_environment(monkeypatch)
    monkeypatch.setenv("MEDAPP_DEADLINE_SECONDS", "60")

    with pytest.raises(ValidationError):
        Settings()


def test_unknown_device_is_rejected_rather_than_passed_to_the_transcriber(monkeypatch):
    _clear_medapp_environment(monkeypatch)
    monkeypatch.setenv("MEDAPP_DEVICE", "mps")

    with pytest.raises(ValidationError):
        Settings()


def test_a_beam_size_below_one_is_rejected(monkeypatch):
    _clear_medapp_environment(monkeypatch)
    monkeypatch.setenv("MEDAPP_BEAM_SIZE", "0")

    with pytest.raises(ValidationError):
        Settings()


def test_settings_are_frozen(monkeypatch):
    _clear_medapp_environment(monkeypatch)

    settings = Settings()

    with pytest.raises(ValidationError):
        settings.device = "cuda"
