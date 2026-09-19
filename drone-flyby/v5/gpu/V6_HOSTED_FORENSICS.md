# V6 hosted Validation forensics (attempt 59f3a642, score 0.0178, >15 rejected camera cmds)

## Root cause (evidence-based) — NOT a deployment mismatch
The hosted process WAS the fixed V6: PID 18103 started 2026-09-19 21:32 > pipeline_v6.py
mtime 21:06 (feasible-search `_legal_step` present); `service:"V6"`. The per-request hosted log
(`endpoint_v6_public_9053.log`, preserved `.preserved_*`) shows our commands were legal vs the
`r.view` we received, e.g.:
- fi5 (frame6): r.view L1(960,540) -> emitted L2(600,1531), distance 1054 < 1102 = LEGAL vs r.view.
  Evaluator rejected citing current (2016,540) = OUR fi4 request.
- fi13 (frame14): r.view L1(2880,540) -> emitted L1(2880,1620), 1080 < 1102 = LEGAL vs r.view.
  Evaluator rejected citing current (2040,1012) = OUR fi11 request.

The hosted evaluator applies our camera commands ASYNCHRONOUSLY (pipelined) and skips frames under
the realtime clock (frame_index gaps 0,3,4,5,7,9,11,13,...). So `r.view` is STALE: it lags the
evaluator's authoritative camera by one-or-more of OUR OWN issued commands. Planning legality against
`r.view` (as the pinned code did) is therefore rejected, because the evaluator validates against its
already-advanced state. The forensic hypothesis of a deployment/source mismatch is REFUTED by the
process-start-vs-mtime and log evidence.

## Fix
`pipeline_v6.py`: plan and validate camera legality against a BELIEVED authoritative camera derived
from our own issued-command chain, reconciled with `camera_command_feedback` (a command is treated as
applied unless feedback reports it refused). Detection/tracking still use `r.view`'s region (the image
is rendered there). Added an independent final guard `command_is_legal(...)` (mirrors
`local_evaluator.Camera.apply`: allowed transition, destination bounds, movement <= current-level
limit, L0 reset exempt); an illegal proposed command becomes a no-op (camera holds).

## Tests (model-free, tests/test_camera_v6.py)
Reproduces the reported illegal pairs; verifies transitions; LAGGED/pipelined evaluator replay
(lag 1 & 2, with/without frame skips, 160 frames) -> 0 illegal commands; guard catches corrupted
internal state. Offline Helsinki mAP unchanged at 0.265 (0 refused).

## Low mAP: not camera alone
Hosted log: 148 /predict, durations 34-137 ms (no timeouts), 1-11 predictions/frame. The 0.018 hosted
score is (a) camera desync (now fixed) causing us to observe the wrong regions vs our tracking, AND
(b) the KNOWN domain gap - V6's detector is trained on Helsinki instances; hosted objects differ, so
generalization is unproven. Both must be addressed before expecting a strong hosted score.
