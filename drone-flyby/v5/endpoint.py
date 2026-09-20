"""Isolated V5 endpoint. Production api.py/example.py are unchanged."""
from collections import deque
from contextlib import asynccontextmanager
import logging
import threading
import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from dtos import DroneFlybyPredictRequestDto, DroneFlybyPredictResponseDto
from v5.pipeline import Pipeline
from v5.common import timing

pipeline=None
startup_error=None
latencies=deque(maxlen=2000)
lock=threading.Lock()


@asynccontextmanager
async def lifespan(app):
    global pipeline,startup_error
    try:
        pipeline=Pipeline()
        import numpy as np
        image=np.zeros((540,960,3),dtype=np.uint8)
        pipeline.discovery.propose(image,[0,0,3840,2160])
        pipeline.expert.classify(image,[[100,100,140,140]])
    except Exception as exc:
        startup_error=f'{type(exc).__name__}: {exc}'
        logging.exception('V5 startup failed')
        pipeline=None
    yield


app=FastAPI(lifespan=lifespan)


@app.get('/')
def health():
    return JSONResponse({'service':'V5','ready':pipeline is not None,'error':startup_error},status_code=200 if pipeline else 503)


@app.get('/metrics')
def metrics():
    with lock:
        samples=list(latencies)
    return {'endpoint_ms':timing(samples),'count':len(samples),
        'configuration':vars(pipeline.config) if pipeline else None,'startup_error':startup_error}


@app.post('/reset')
def reset():
    """Explicit episode reset for isolated local evaluation (loopback binding)."""
    if pipeline:
        with pipeline.lock:
            pipeline.states.clear()
    with lock:
        latencies.clear()
    return {'reset':True}


@app.post('/predict',response_model=DroneFlybyPredictResponseDto)
def predict(request:DroneFlybyPredictRequestDto):
    start=time.perf_counter()
    result=pipeline.predict(request) if pipeline else DroneFlybyPredictResponseDto(
        request_id=request.request_id,frame=request.frame,annotations=[])
    with lock:
        latencies.append((time.perf_counter()-start)*1000)
    return result
