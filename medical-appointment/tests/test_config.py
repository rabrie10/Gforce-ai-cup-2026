"""Settings is the only module that reads the environment.

These tests pin both halves of that: an unset environment gives the working
development defaults, and every value the deployment VM overrides is actually
reachable from the environment.
"""

import pytest
from pydantic import ValidationError

from medapp.config import Settings


def test_defaults_resolve_without_any_environment(monkeypatch):
    for name in ("MEDAPP_DEVICE", "MEDAPP_COMPUTE_TYPE", "MEDAPP_DEADLINE_SECONDS"):
        monkeypatch.delenv(name, raising=False)

    settings = Settings()

    assert settings.device == "cpu"
    assert settings.compute_type == "int8"
    assert settings.answer_strategy == "retrieve_rerank_entail"
    assert settings.deadline_seconds == 50.0
    assert settings.transcript_cache_dir.name == "transcripts"


def test_environment_overrides_every_deployment_value(monkeypatch):
    monkeypatch.setenv("MEDAPP_DEVICE", "cuda")
    monkeypatch.setenv("MEDAPP_COMPUTE_TYPE", "float16")
    monkeypatch.setenv("MEDAPP_WHISPER_MODEL", "large-v3-turbo")
    monkeypatch.setenv("MEDAPP_ANSWER_STRATEGY", "single_llm")
    monkeypatch.setenv("MEDAPP_DEADLINE_SECONDS", "42.5")
    monkeypatch.setenv("MEDAPP_MODEL_CACHE_DIR", "/weights")

    settings = Settings()

    assert settings.device == "cuda"
    assert settings.compute_type == "float16"
    assert settings.whisper_model == "large-v3-turbo"
    assert settings.answer_strategy == "single_llm"
    assert settings.deadline_seconds == 42.5
    assert str(settings.model_cache_dir) == "/weights"


def test_deadline_must_stay_under_the_service_timeout(monkeypatch):
    monkeypatch.setenv("MEDAPP_DEADLINE_SECONDS", "90")

    with pytest.raises(ValidationError):
        Settings()


def test_unknown_device_is_rejected_rather_than_passed_to_the_transcriber(monkeypatch):
    monkeypatch.setenv("MEDAPP_DEVICE", "mps")

    with pytest.raises(ValidationError):
        Settings()


def test_settings_are_frozen(monkeypatch):
    settings = Settings()

    with pytest.raises(ValidationError):
        settings.device = "cuda"
