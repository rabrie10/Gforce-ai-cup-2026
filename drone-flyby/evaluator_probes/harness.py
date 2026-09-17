"""Temporary scenes, real HTTP transport, and read-only replay observation."""

import contextlib
import copy
import dataclasses
import inspect
import io
import json
import statistics
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

import local_evaluator as evaluator
from dtos import DroneFlybyPredictResponseDto
from utils import source_bbox_to_global


def detection(box=(100, 100, 300, 300), confidence=0.9, object_id='hangar'):
    return dict(object_id=object_id, bbox=list(box), confidence=confidence)


def make_scene(root, name, count=1, objects=None):
    """Real on-disk fixture, read by the official loaders without monkeypatches."""
    scene = Path(root) / name
    (scene / 'images').mkdir(parents=True)
    (scene / 'annotations').mkdir()
    objects = objects if objects is not None else [detection()]
    annotations = [{k: v for k, v in obj.items() if k != 'confidence'} for obj in objects]
    canvas = np.zeros((2160, 3840, 3), dtype=np.uint8)
    for obj in objects:
        x1, y1, x2, y2 = map(int, obj['bbox'])
        cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 255), -1)
    ok, png = cv2.imencode('.png', canvas)
    if not ok:
        raise RuntimeError('Fixture PNG encoding failed')
    for frame in range(count):
        (scene / 'images' / f'frame_{frame:06d}.png').write_bytes(png.tobytes())
        (scene / 'annotations' / f'frame_{frame:06d}.json').write_text(
            json.dumps(dict(frame=frame, annotations=annotations)), encoding='utf-8')
    return str(scene.resolve())


def score(scene, predictions):
    with contextlib.redirect_stdout(io.StringIO()):
        value, classes = evaluator.score(scene, predictions)
    return dict(score=value, ap_by_class=classes)


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]))
    area = lambda box: (box[2] - box[0]) * (box[3] - box[1])
    return intersection / (area(a) + area(b) - intersection)


def oracle_body(payload, predictions):
    return dict(request_id=payload['request_id'], frame=payload['frame'], annotations=[
        dict(object_id=d['object_id'], bbox=list(source_bbox_to_global(d['bbox'])),
             confidence=d['confidence']) for d in predictions[payload['frame']]])


class Endpoint:
    """Only localhost; no external services, models, or simulated clocks."""

    def __init__(self, oracle, mutate=None, delay_ms=0):
        self.oracle, self.mutate, self.delay_ms = oracle, mutate, delay_ms
        self.records = []

    def __enter__(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *args):
                pass

            def handle(self):
                try:
                    super().handle()
                except (ConnectionResetError, ConnectionAbortedError):
                    # Expected when replay abandons a timed-out connection.
                    pass

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                body = oracle_body(payload, owner.oracle)
                if owner.mutate:
                    owner.mutate(body, payload)
                # Parse precisely the bytes that will reach requests, including
                # the standard JSON exponent overflow case.
                encoded = json.dumps(body, allow_nan=False).replace('"OVERFLOW_NUMBER"', '1e999').encode()
                try:
                    DroneFlybyPredictResponseDto.model_validate(json.loads(encoded))
                    schema_valid, schema_error = True, None
                except Exception as exc:
                    schema_valid, schema_error = False, str(exc)
                # No image bodies or repeated 501-annotation payloads in results.
                record = dict(frame=payload['frame'], frame_index=payload['frame_index'],
                              view={k: v for k, v in payload['view'].items() if k != 'image'},
                              feedback=payload['camera_command_feedback'],
                              requested_view=body.get('requested_view'),
                              schema_valid=schema_valid, schema_error=schema_error,
                              annotation_count=len(body['annotations']))
                owner.records.append(record)
                started = time.monotonic()
                time.sleep(owner.delay_ms / 1000)
                record['actual_sleep_ms'] = (time.monotonic() - started) * 1000
                # allow_nan=False keeps the wire strict JSON. Infinity experiments
                # use an ordinary JSON numeric exponent that overflows on parsing.
                try:
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(encoded)))
                    self.end_headers()
                    self.wfile.write(encoded)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    record['client_disconnected'] = True

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}/predict'
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


def run_replay(scene, mutate=None, delay_ms=0, realtime=False, observe=True):
    oracle = evaluator.oracle_predictions(scene)
    transitions = []
    # Observe locals after next_index is computed. No replacement of functions,
    # clocks, HTTP calls, constants, inputs, or evaluator control flow.
    lines, first = inspect.getsourcelines(evaluator.replay)
    marker = first + next(i for i, line in enumerate(lines) if 'counted = min(next_index' in line)

    def trace(frame, event, arg):
        if frame.f_code is not evaluator.replay.__code__:
            return None
        if event == 'line' and frame.f_lineno == marker:
            local = frame.f_locals
            transitions.append(dict(frame_index=local['frame_index'], elapsed=local['elapsed'],
                                    next_index=local['next_index'],
                                    sent_elapsed=local['sent_at'] - local['started']))
        return trace

    output = io.StringIO()
    with Endpoint(oracle, mutate, delay_ms) as endpoint:
        previous_trace = sys.gettrace()
        started = time.monotonic()
        try:
            if observe:
                sys.settrace(trace)
            with contextlib.redirect_stdout(output):
                predictions, stats = evaluator.replay(endpoint.url, scene, realtime, 0, False)
        finally:
            sys.settrace(previous_trace)
        wall_seconds = time.monotonic() - started
        records = copy.deepcopy(endpoint.records)
    result = score(scene, predictions)
    result.update(statistics=dataclasses.asdict(stats), requests=records,
                  transitions=transitions, evaluator_output=output.getvalue(),
                  delay_ms=delay_ms, wall_seconds=wall_seconds, trace_enabled=observe,
                  predicted_frames=list(predictions),
                  detections_by_frame={str(k): len(v) for k, v in predictions.items()})
    times = stats.round_trip_ms
    result['rtt_ms'] = dict(mean=statistics.mean(times), median=statistics.median(times), max=max(times))
    result['recurrence_matches'] = (all(t['next_index'] == max(
        t['frame_index'] + 1, int(t['elapsed'] / evaluator.FRAME_INTERVAL_SECONDS)) for t in transitions)
        if transitions else None)
    result['frame_accounting_matches'] = (stats.frames_sent + stats.frames_skipped == stats.frames_total
                                          and stats.responses_accepted + stats.frames_unanswered == stats.frames_sent)
    return result
