# V7 hosted results

| attempt | detector | score | errors |
|---|---|---|---|
| V6 initial | ms1 | 0.0178167605 | - |
| V6 camera-corrected | ms1 | 0.0127033372 | - |
| V6 diagnostic (d09922e7) | ms1 | 0.0081947755 | [] |
| V7 first try (7306c67f) | v7_e15 | 0 | >15 transport timeouts, aborted |
| **V7 retry (a23ca0fb)** | **v7_e15** | **0.0318212348** | **4** (2 timeouts, 2 illegal camera) |

Retry attempt a23ca0fbf1124a3c852680c8a39d4d53, sequence 0f1bc27f24c1475f9fb6d92d989a5bfa,
13:01:29 -> 13:02:55 UTC. Checkpoint /workspace/v7_e15_snapshot.pt
sha256 52b9fcef703d3fc5f35993f3088381f545b50d1091c862f52ff0c4df1cd091b8,
V6_CLASSIFY_MIN_PX=16, V6_TARGET_MIN=0.5, V6_DET_CONF=0.15, V6_ENSEMBLE=0, V6_DIAGNOSTIC_CAPTURE=0.

App-side: 155/249 requests reached the app, 154/155 responses delivered 200, p50 68 ms,
p95 113 ms, max 230 ms, 0 >1000 ms, 0 exceptions, peak RSS 1.83 GB, peak GPU 1948 MiB.
93 frame indices never reached the app (transport loss), longest gap 11.

## Known open issue (NOT fixed, not deployed)
Frames 196 and 202: illegal camera request, both "from L2 (2001,1609)". The planner steps from
`st.believed` (its own issued-command chain), so after the frame-184 timeout the belief diverged
from the evaluator's real camera and it repeatedly requested moves illegal from the ACTUAL
position (1041 px and 1492 px vs the 551 px L2 limit), leaving the camera stuck.
Candidate minimal fix (untested, not applied): validate/clamp the planned step against the
RECEIVED view as well as the believed state, falling back to a legal step from the received view.
Requires redeploy + another Validation to verify.
