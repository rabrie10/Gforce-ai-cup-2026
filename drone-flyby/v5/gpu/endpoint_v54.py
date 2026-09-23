"""V5.4 isolated endpoint: dual pretrained aerial detectors (yolo11l-obb + yolov8l-worldv2)
merged discovery + existing DINOv2 Visual Expert (with reference-similarity background
rejection) + existing causal tracking + legal L1/L2 camera. Components are injected into
the unchanged v5.pipeline.Pipeline. Existing v5/endpoint.py is left intact as rollback."""
from collections import deque
from contextlib import asynccontextmanager
import logging, threading, time

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from dtos import DroneFlybyPredictRequestDto, DroneFlybyPredictResponseDto
from v5.pipeline import Pipeline, Config
from v5.common import timing
from v5.gpu.discovery_v54 import MergedDiscovery, RejectingExpert

pipeline = None
startup_error = None
latencies = deque(maxlen=2000)
lock = threading.Lock()


@asynccontextmanager
async def lifespan(app):
    global pipeline, startup_error
    try:
        cfg = Config()
        disc = MergedDiscovery(cfg.assets, budget=cfg.candidate_budget, device=cfg.device, threads=cfg.threads)
        expert = RejectingExpert(cfg.assets, device=cfg.device, head_name=cfg.head_name, threads=cfg.threads)
        pipeline = Pipeline(cfg, discovery=disc, expert=expert)
        import numpy as np
        img = np.zeros((540, 960, 3), dtype=np.uint8)
        pipeline.discovery.propose(img, [0, 0, 3840, 2160])
        pipeline.expert.classify(img, [[100, 100, 140, 140]])
    except Exception as exc:
        startup_error = f"{type(exc).__name__}: {exc}"
        logging.exception("V5.4 startup failed")
        pipeline = None
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
def health():
    return JSONResponse({"service": "V5.4", "ready": pipeline is not None, "error": startup_error},
                        status_code=200 if pipeline else 503)


@app.get("/metrics")
def metrics():
    with lock:
        samples = list(latencies)
    return {"endpoint_ms": timing(samples), "count": len(samples),
            "configuration": vars(pipeline.config) if pipeline else None, "startup_error": startup_error}


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
