"""The one module that reads the environment.

Device, compute type, model names, the per-request deadline, the Answer
Strategy and the cache directories are resolved here and nowhere else, so no
other module branches on platform. Defaults are the development-machine values;
the deployment VM overrides them:

    MEDAPP_DEVICE=cuda MEDAPP_COMPUTE_TYPE=float16 python api.py
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Device = Literal["cpu", "cuda"]
ComputeType = Literal["int8", "int8_float16", "float16", "float32"]
AnswerStrategy = Literal["cite_first_segment", "retrieve_rerank_entail", "single_llm"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Every value the system resolves from its environment.

    Attributes:
        answer_strategy: Which Answerer is constructed at startup. Candidates
            are compared by running the system twice, not by a runtime switch.
            A strategy no Answerer implements yet fails at startup rather than
            being quietly substituted.
        route_suffix: Appended to the ``/predict`` path so the deployed
            endpoint sits at an unguessable route. Empty in development.
        deadline_seconds: Checked between stages and between Questions, and
            strictly under the service's 60-second request timeout.
        model_cache_dir: Where model weights live, pre-fetched into the image
            at build time rather than downloaded on first request.
        transcript_cache_dir: The development transcript cache. Written and
            read by ``scripts/`` only; nothing under ``medapp`` touches it.
        whisper_language: Fixed rather than detected per Conversation, so a
            short or noisy opening cannot send one transcript through a
            different language's decoder.
        vad_filter: Skips silence before decoding, and is the standard guard
            against the hallucination loops that multiply decode time on a
            quiet stretch.
        condition_on_previous_text: Off for the same reason; a repetition that
            enters the prompt otherwise sustains itself.
        beam_size: Worst-case latency scales with it, so it is tuned on the dev
            fold against timing error rather than taken on faith.
    """

    model_config = SettingsConfigDict(
        env_prefix="MEDAPP_",
        # ``model_`` is a pydantic-protected prefix and several fields are
        # model names.
        protected_namespaces=(),
        frozen=True,
    )

    device: Device = "cpu"
    compute_type: ComputeType = "int8"

    whisper_model: str = "large-v3"
    whisper_language: str = "en"
    vad_filter: bool = True
    condition_on_previous_text: bool = False
    beam_size: int = Field(default=5, ge=1, le=10)

    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    dense_model: str = "BAAI/bge-small-en-v1.5"

    answer_strategy: AnswerStrategy = "cite_first_segment"
    route_suffix: str = ""

    deadline_seconds: float = Field(default=50.0, gt=0, lt=60)

    model_cache_dir: Path = PROJECT_ROOT / "models"
    transcript_cache_dir: Path = PROJECT_ROOT / "transcripts"


settings = Settings()
