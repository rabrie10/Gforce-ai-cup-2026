"""V6 multiscale active-perception endpoint. Standalone V6Pipeline; existing endpoints untouched."""
from collections import deque
from contextlib import asynccontextmanager
import logging, threading, time
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from dtos import DroneFlybyPredictRequestDto, DroneFlybyPredictResponseDto
from v5.common import timing
from v5.gpu.pipeline_v6 import V6Pipeline

pipeline = None
startup_error = None
latencies = deque(maxlen=4000)
lock = threading.Lock()


@asynccontextmanager
async def lifespan(app):
    global pipeline, startup_error
    try:
        pipeline = V6Pipeline()
    except Exception as exc:
        startup_error = f"{type(exc).__name__}: {exc}"
        logging.exception("V6 startup failed")
        pipeline = None
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
def health():
    return JSONResponse({"service": "V6", "ready": pipeline is not None, "error": startup_error},
                        status_code=200 if pipeline else 503)


@app.get("/metrics")
def metrics():
    with lock:
        s = list(latencies)
    return {"endpoint_ms": timing(s), "count": len(s), "startup_error": startup_error,
            "manifest": pipeline.manifest if pipeline else None}


@app.post("/reset")
def reset():
    if pipeline:
        with pipeline.lock:
            pipeline.states.clear()
    with lock:
        latencies.clear()
    return {"reset": True}


@app.post("/predict", response_model=DroneFlybyPredictResponseDto)
def predict(request: DroneFlybyPredictRequestDto):
    start = time.perf_counter()
    result = pipeline.predict(request) if pipeline else DroneFlybyPredictResponseDto(
        request_id=request.request_id, frame=request.frame, annotations=[])
    with lock:
        latencies.append((time.perf_counter() - start) * 1000)
    return result
