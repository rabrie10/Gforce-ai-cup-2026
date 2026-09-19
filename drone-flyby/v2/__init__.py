"""Architecture V2 for the Drone Flyby use case.

Modules are deliberately separable: ``proposals`` (discovery), ``gallery`` and
``recognizer`` (identity), ``tracks`` (global state), ``gmc`` (shared motion),
``scheduler`` (camera policy), ``output`` (confidence and deduplication),
``telemetry`` (validation capture) and ``pipeline`` (the closed loop).

Thread-pool policy is set here, before anything imports torch, because it has
to be: OpenMP reads these once at initialization and ignores later changes.

Two inference runtimes share four vCPUs in the request path - onnxruntime for
the proposal detector, torch for the crop recognizer - and they run
back-to-back on every frame. Left at their defaults, each keeps its workers
spinning after a call and starves the other. That was measured, not guessed:
the recognizer went from 76 ms to 249 ms simply because an ORT session existed,
and the detector from 111 ms standalone to 207 ms next to torch. A spinning
idle thread costs a frame, and a late frame is scored as no detections.
"""

import os as _os

# Idle OpenMP workers sleep instead of burning a core waiting for the next
# parallel region. ORT's own spinning is disabled through its session options.
_os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')
_os.environ.setdefault('KMP_BLOCKTIME', '0')
