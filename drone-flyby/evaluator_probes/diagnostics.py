"""Focused follow-ups justified by image overhead and interpolation results."""

import sys
import time

import local_evaluator as evaluator
from evaluator_probes.harness import detection, make_scene, run_replay, score


def timing_controls(root):
    print('P0 F controls: preprocessing and no-trace Helsinki replay', flush=True)
    preprocessing = []
    camera = evaluator.Camera()
    for frame in evaluator.frame_numbers('helsinki'):
        started = time.monotonic()
        image = evaluator.load_frame(frame, 'helsinki')
        loaded = time.monotonic()
        evaluator.render_view(image, camera)
        rendered = time.monotonic()
        preprocessing.append(dict(frame=frame, load_ms=(loaded - started) * 1000,
                                  render_ms=(rendered - loaded) * 1000,
                                  total_ms=(rendered - started) * 1000))
    no_trace = run_replay('helsinki', realtime=True, observe=False)
    # Same dimensions and exact rendering/HTTP path, much simpler PNG content.
    scene = make_scene(root, 'timing_control', 25)
    synthetic = []
    for delay in (0, 200, 250, 300, 333, 350):
        print(f'P0 F control: synthetic realtime, endpoint sleep {delay} ms', flush=True)
        synthetic.append(run_replay(scene, delay_ms=delay, realtime=True))
    return dict(preprocessing_ms=preprocessing, helsinki_no_trace=no_trace, synthetic=synthetic)


def precision_controls(root):
    """Read the actual COCO precision tensor on return from official score()."""
    scene = make_scene(root, 'precision')
    cases = {'TP_first': [detection(), detection((600, 600, 800, 800), .1)],
             'FP_first': [detection(confidence=.1), detection((600, 600, 800, 800), .9)]}
    results = {}
    for name, predictions in cases.items():
        observed = {}
        def trace(frame, event, arg):
            if frame.f_code is not evaluator.score.__code__:
                return None
            if event == 'return':
                coco = frame.f_locals['evaluator']
                observed.update(precision=coco.eval['precision'][0, :, 0, 0, -1].tolist(),
                                recall_thresholds=coco.params.recThrs.tolist(),
                                max_recall=float(coco.eval['recall'][0, 0, 0, -1]))
            return trace
        previous = sys.gettrace()
        try:
            sys.settrace(trace)
            measured = score(scene, {0: predictions})
        finally:
            sys.settrace(previous)
        results[name] = dict(**measured, **observed)
    return results
