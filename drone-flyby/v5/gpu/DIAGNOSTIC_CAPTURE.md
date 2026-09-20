# Optional V6 L1/L2 diagnostic capture

This worktree contains a default-off capture component for a future authorized
Validation. It is wired into this isolated corrected-V6 candidate only; it was
not deployed to the public Runpod endpoint.

## Verified integration point

The deployed V6 source was inspected at:

- `/workspace/v5-perception/drone-flyby/v5/gpu/pipeline_v6.py`
- `/workspace/v5-perception/drone-flyby/v5/gpu/endpoint_v6.py`

The correct hook is in `V6Pipeline._predict`, after:

1. `image = decode_view(r.view)` validates the received 960x540 image.
2. `props = self._detect(image, region)` produces raw candidates.
3. Tracking/classification constructs `annotations`.

This candidate creates one `DiagnosticCapture()` per pipeline and calls the
adapter immediately before the existing response return:

```python
record_v6_diagnostics(self.capture, r, image, props, response)
```

The call is best effort. It does not alter `response`, camera decisions,
tracking state, candidate filtering, or timing control flow.

## Activation

Capture is disabled unless `V6_DIAGNOSTIC_CAPTURE=1`. When enabled:

- only received L1/L2 views are admitted;
- each sequence is limited to 10 captures per level;
- each 50-frame window admits at most 2 captures per level, stratifying the
  approximately 0-249 frame sequence across windows 0-49, 50-99, 100-149,
  150-199, and 200-249;
- admission uses `put_nowait` on a bounded queue;
- PNG encoding and disk writes run on one daemon writer thread;
- queue overflow, encoding errors, and disk errors drop the diagnostic item;
- the default disk quota is 50 MiB;
- every PNG has a matching JSON sidecar keyed by sanitized `sequence_id` and
  `request_id`.

The sidecar records the received level, center, source crop, image dimensions,
frame identity, timestamp, pixel SHA-256, **post-merge detector candidates**,
and final emitted predictions. The candidate list is after cross-detector
greedy NMS and before classification, tracking, and final emission filtering;
it is not the unmerged output of each detector.

## Public deployment changes eventually required

No public endpoint file was changed here. A future authorized deployment would
need to deploy the updated `pipeline_v6.py`, `endpoint_v6.py`, diagnostic module,
and adapter together, set `V6_DIAGNOSTIC_CAPTURE=1`, choose a writable
`V6_DIAGNOSTIC_DIR`, and verify the disk quota before Validation. The endpoint
lifespan closes the writer on shutdown.