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
# Use Uvicorn's configured error logger so diagnostics are emitted into the
# server log without adding a handler or altering the existing V5.4 logging.
request_log = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app):
    global pipeline, startup_error
    try:
        pipeline = V6Pipeline()
    except Exception as exc:
        startup_error = f"{type(exc).__name__}: {exc}"
        logging.exception("V6 startup failed")
        pipeline = None
    try:
        yield
    finally:
        if pipeline:
            pipeline.close()


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
    exception_type = None
    if pipeline:
        result = pipeline.predict(request)
        diagnostic = pipeline.last_diagnostics
        exception_type = diagnostic.get("error", "").split(":", 1)[0] or None
    else:
        result = DroneFlybyPredictResponseDto(request_id=request.request_id, frame=request.frame,
                                               annotations=[])
        exception_type = startup_error or "pipeline_unavailable"
    duration_ms = (time.perf_counter() - start) * 1000
    requested = result.requested_view
    requested_view = ("-" if requested is None else
                      f"L{requested.resolution_level}:({requested.center_x},{requested.center_y})")
    with lock:
        latencies.append(duration_ms)
    request_log.info(
        "v6_request request_id=%s frame=%s frame_index=%s view_level=%s view_center=(%s,%s) "
        "requested_view=%s duration_ms=%.2f status=%s predictions=%d exception=%s",
        request.request_id, request.frame, request.frame_index, request.view.resolution_level,
        request.view.center_x, request.view.center_y, requested_view, duration_ms,
        "error" if exception_type else "ok", len(result.annotations), exception_type or "-")
    return result
