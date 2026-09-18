"""The adapter between the shipped transport and the system.

Everything the endpoint does lives in ``medapp``; this module only builds it.
The models are constructed and exercised here, at import, so the process is not
serving before it can answer: ``api.py`` imports this module, and uvicorn does
not bind the port until that import returns.
"""

import logging

from dtos import ASRQuestionRequestDto, ASRQuestionResponseDto
from medapp.answerer import build_answerer
from medapp.config import settings
from medapp.service import PredictionService
from medapp.transcriber import Transcriber

logger = logging.getLogger(__name__)

logger.info(
    "Loading the %s Transcriber on %s (%s) for the %s Answerer.",
    settings.whisper_model,
    settings.device,
    settings.compute_type,
    settings.answer_strategy,
)

_transcriber = Transcriber(settings)
_transcriber.warm_up()

_service = PredictionService(_transcriber, build_answerer(settings), settings)

logger.info("Ready.")


def predict(request: ASRQuestionRequestDto) -> ASRQuestionResponseDto:
    """Answer every question about one conversation."""
    return _service.predict(request)
