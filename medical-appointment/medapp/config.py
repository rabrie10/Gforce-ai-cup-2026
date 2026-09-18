"""The one module that reads the environment.

Device, compute type, model names, the per-request deadline, the Answer
Strategy and the cache directories are resolved here and nowhere else. No other
module branches on platform or checks for CUDA: the Mac development machine and
the CUDA deployment VM differ only in the values below, which is what makes the
same image runnable on both.

Defaults are the development-machine values, so an unset environment gives a
working CPU configuration. The VM overrides them, prefixed ``MEDAPP_``:

    MEDAPP_DEVICE=cuda MEDAPP_COMPUTE_TYPE=float16 python api.py
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Device = Literal["cpu", "cuda", "auto"]
ComputeType = Literal["int8", "int8_float16", "float16", "float32"]

# Which Answerer is constructed at startup. There is no runtime switch between
# them and no degraded Answerer to fall back to; candidates are compared by
# running the system twice under different configuration.
AnswerStrategy = Literal["retrieve_rerank_entail", "single_llm"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Every value the system resolves from its environment."""

    model_config = SettingsConfigDict(
        env_prefix="MEDAPP_",
        # ``model_`` is a pydantic-protected prefix and several fields here are
        # model names; the namespace is cleared rather than the fields renamed.
        protected_namespaces=(),
        frozen=True,
    )

    # --- Platform ------------------------------------------------------- #

    device: Device = "cpu"
    compute_type: ComputeType = "int8"

    # --- Models --------------------------------------------------------- #

    whisper_model: str = "large-v3"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    dense_model: str = "BAAI/bge-small-en-v1.5"

    # --- Answering ------------------------------------------------------ #

    answer_strategy: AnswerStrategy = "retrieve_rerank_entail"

    # --- Budget --------------------------------------------------------- #

    # Below the evaluation service's 60-second per-request timeout. Checked
    # between stages and between Questions by the protocol guard: being early
    # with guesses loses at most the Questions guessed, being late loses the
    # whole Conversation.
    deadline_seconds: float = Field(default=50.0, gt=0, le=60)

    # --- Caches --------------------------------------------------------- #

    # Where model weights live. Pre-fetched into the image at build time and
    # read offline at runtime, never downloaded on first request.
    model_cache_dir: Path = PROJECT_ROOT / "models"

    # The development transcript cache. Written and read by ``scripts/`` only:
    # nothing under ``medapp`` touches it, so the dev cache cannot become
    # production behaviour.
    transcript_cache_dir: Path = PROJECT_ROOT / "transcripts"


# Resolved once, at import.
settings = Settings()
