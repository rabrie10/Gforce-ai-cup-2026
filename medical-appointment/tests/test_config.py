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
    assert settings.answer_strategy == "retrieve_rerank_entail"
    assert settings.relevance_threshold == 0.3
    assert settings.entail_depth == 2
    assert settings.relevance_gate is True
    assert settings.entailment_threshold == 0.000394
    assert settings.chunk_word_lengths == (1, 2, 3, 4, 6, 8, 11, 15, 20, 27, 36, 48)
    assert settings.chunk_stride_fraction == 0.2
    assert settings.retrieval_candidates == 10
    assert settings.retrieval_mode == "bm25"
    assert settings.fusion_depth == 50
    assert settings.fusion_rank_constant == 60.0
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


class TestTheTwoDeviceSettings:
    """CTranslate2 and PyTorch do not have the same device set, so they do not
    share a setting. Metal is valid for the cross-encoders and invalid for the
    Transcriber, and one Literal over both would be wrong in one direction."""

    def test_the_transcriber_device_admits_no_metal(self, monkeypatch):
        _clear_medapp_environment(monkeypatch)

        with pytest.raises(ValidationError):
            Settings(device="mps")

    def test_the_torch_device_admits_metal(self, monkeypatch):
        _clear_medapp_environment(monkeypatch)

        assert Settings(torch_device="mps").torch_device == "mps"

    def test_the_torch_device_defaults_to_auto(self, monkeypatch):
        _clear_medapp_environment(monkeypatch)

        assert Settings().torch_device == "auto"

    def test_an_explicit_torch_device_is_returned_unresolved(self):
        from medapp.config import resolved_torch_device

        assert resolved_torch_device("cpu") == "cpu"
        assert resolved_torch_device("mps") == "mps"
        assert resolved_torch_device("cuda") == "cuda"

    def test_auto_resolves_to_something_torch_accepts(self):
        import torch

        from medapp.config import resolved_torch_device

        resolved = resolved_torch_device("auto")

        assert resolved in {"cpu", "cuda", "mps"}
        # Constructing the device is what proves the string is usable; a name
        # torch rejects would only surface when a model was loaded onto it.
        assert torch.device(resolved).type == resolved

    def test_cpu_threads_defaults_to_the_decoders_own_choice(self, monkeypatch):
        _clear_medapp_environment(monkeypatch)

        assert Settings().cpu_threads == 0

    def test_cpu_threads_is_read_from_the_environment(self, monkeypatch):
        _clear_medapp_environment(monkeypatch)
        monkeypatch.setenv("MEDAPP_CPU_THREADS", "10")

        assert Settings().cpu_threads == 10

    def test_a_negative_thread_count_is_refused(self, monkeypatch):
        _clear_medapp_environment(monkeypatch)

        with pytest.raises(ValidationError):
            Settings(cpu_threads=-1)
