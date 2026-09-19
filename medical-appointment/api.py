"""The endpoint the evaluation service calls.

You should not need to change much in here. Put your model in ``example.py``
and leave the transport alone.

The URL you submit is used exactly as you give it, path included, so if you
keep the ``/predict`` route below then submit ``http://<your-host>:9054/predict``
rather than just the host.
"""

import datetime
import logging
import time

import uvicorn
from fastapi import FastAPI

from dtos import ASRQuestionRequestDto, ASRQuestionResponseDto
from example import predict
from medapp.config import settings
from medapp.service import scorable_response
from utils import validate_response

HOST = "0.0.0.0"
PORT = 9054

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
start_time = time.time()


@app.post(f"/predict{settings.route_suffix}", response_model=ASRQuestionResponseDto)
def predict_endpoint(request: ASRQuestionRequestDto):
    """Answer every question about one conversation.

    Nothing raises out of this route. A 500 and a body the evaluator cannot
    read are scored the same — every question about this conversation wrong —
    so the only thing an exception here would buy is a worse log line. The
    body is repaired instead, and the repair is logged at error level.
    """
    expected_count = len(request.questions)

    try:
        response = scorable_response(
            predict(request), expected_count, request.audio_filename
        )
    except Exception:
        logger.error(
            "%s: answering raised; guessing every question.",
            request.audio_filename,
            exc_info=True,
        )
        response = ASRQuestionResponseDto(
            answers=[True] * expected_count,
            evidence_start=[None] * expected_count,
            evidence_end=[None] * expected_count,
        )

    # The repair above is written to make this impossible. It stays as the
    # independent check on that claim, and it no longer decides the response.
    try:
        validate_response(response, expected_count=expected_count)
    except ValueError:
        logger.error(
            "%s: the repaired body is still not scorable.",
            request.audio_filename,
            exc_info=True,
        )

    return response


@app.get("/api")
def hello():
    return {
        "service": "medical-appointment-usecase",
        "uptime": f"{datetime.timedelta(seconds=time.time() - start_time)}",
    }


@app.get("/")
def index():
    return "Your endpoint is running!"


if __name__ == "__main__":
    uvicorn.run("api:app", host=HOST, port=PORT)
