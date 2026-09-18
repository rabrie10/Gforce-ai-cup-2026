"""The endpoint the evaluation service calls.

You should not need to change much in here. Put your model in ``example.py``
and leave the transport alone.

The URL you submit is used exactly as you give it, path included, so if you
keep the ``/predict`` route below then submit ``http://<your-host>:9053/predict``
rather than just the host.
"""

import datetime
import logging
import statistics
import threading
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from dtos import DroneFlybyPredictRequestDto, DroneFlybyPredictResponseDto
from example import predict, warmup_detector
from utils import validate_response

HOST = '0.0.0.0'
PORT = 9053

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load and warm the local model once when the service starts."""
    try:
        warmup_detector()
        logger.info('Detector loaded and warmed')
    except Exception:
        logger.exception('Detector startup failed; responses will be empty')
    yield


app = FastAPI(lifespan=lifespan)
start_time = time.time()
_endpoint_times_lock = threading.Lock()
_endpoint_times_ms = []


def reset_endpoint_metrics():
    with _endpoint_times_lock:
        _endpoint_times_ms.clear()


def endpoint_metrics():
    with _endpoint_times_lock:
        values = sorted(_endpoint_times_ms)
    if not values:
        return {'count': 0}
    position = 0.95 * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    p95 = values[lower] + (position - lower) * (values[upper] - values[lower])
    return {
        'count': len(values),
        'mean': statistics.mean(values),
        'median': statistics.median(values),
        'p95': p95,
        'max': max(values),
    }


@app.post('/predict', response_model=DroneFlybyPredictResponseDto)
def predict_endpoint(request: DroneFlybyPredictRequestDto):
    """Answer one frame."""
    started = time.perf_counter()
    response = predict(request)

    # Fail here, loudly, rather than having the evaluator silently discard the
    # frame. Every rule this checks is a rule the evaluator also enforces.
    validate_response(response)
    with _endpoint_times_lock:
        _endpoint_times_ms.append((time.perf_counter() - started) * 1000.0)

    logger.info(
        'frame %s (index %s) L%s at (%s, %s): returned %s detections',
        request.frame,
        request.frame_index,
        request.view.resolution_level,
        request.view.center_x,
        request.view.center_y,
        len(response.annotations),
    )
    return response


@app.get('/api')
def hello():
    return {
        'service': 'drone-flyby-usecase',
        'uptime': '{}'.format(datetime.timedelta(seconds=time.time() - start_time)),
    }


@app.get('/')
def index():
    return "Your endpoint is running!"


if __name__ == '__main__':
    uvicorn.run('api:app', host=HOST, port=PORT)
