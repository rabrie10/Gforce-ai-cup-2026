# Deployment runbook (ticket 12)

What the `Dockerfile` builds, what has been verified locally, and what is left
to run on the real GPU VM. Update the status table as each remaining step is
done on the actual provider — this file, not the ticket checklist, is where
the evidence for each checkbox lives.

## What the image does

One `Dockerfile`, two builds:

```
# The real target: CUDA, cuDNN, large-v3, float16
docker build -t medapp .

# Mac smoke test: CPU, a small model, no GPU to decode on
docker build -t medapp:cpu-smoke \
  --build-arg BASE_IMAGE=python:3.11-slim \
  --build-arg DEVICE=cpu --build-arg COMPUTE_TYPE=int8 \
  --build-arg WHISPER_MODEL=tiny.en .
```

`uv sync --frozen` installs from `uv.lock`, not `requirements.txt`.
`scripts/fetch_models.py` runs once at build time, with the hub reachable, and
bakes every weight the request path loads — the reranker, the NLI judge and the
faster-whisper model named by `MEDAPP_WHISPER_MODEL` — into `/app/models`.
`HF_HUB_OFFLINE=1` is set only after that step, so the built image cannot reach
the hub even if the network is there. Every model loader under `medapp` already
passes `local_files_only=True` (ticket 08/09/10), so this is the last piece
that made offline-only true end to end.

Two bugs the Mac build caught, fixed in this Dockerfile:

- `HF_HUB_OFFLINE=1` set before the prefetch step blocked the prefetch itself.
  It is now set as the last `ENV`, after `scripts.fetch_models` runs.
- `CMD ["uv", "run", "python", "api.py"]` has `uv run` re-check the lockfile's
  package metadata against the network on every container start, even against
  an already-synced venv — exactly the reach-out an offline container must not
  make. `CMD` now invokes the synced venv's interpreter directly:
  `.venv/bin/python api.py`.

## Verified locally (this machine, CPU, `tiny.en`)

- `docker build` of the CPU/small-model variant succeeds, model prefetch
  included.
- `docker run --network none` serves successfully — the container reaches
  `Application startup complete` and answers on `9054` with the network
  disabled at the Docker level, proving no runtime hub reach-out survived the
  two fixes above.
- The root route (`GET /`) returns `200` from outside the container.
- `POST /predict` with the **largest supplied Conversation**
  (`conversation_sample_20.mp3`, 3.7 MB audio / 4.9 MB JSON body) returns `200`
  with a well-formed body — no body-size limit rejects it. There is no reverse
  proxy in front of uvicorn here or in the intended deployment, so no proxy
  body-size cap applies either.
- The service log for that request reads
  `conversation_sample_20.mp3 (231.9 s, 3.7 MB): 10 questions` — filename,
  duration and size, never the base64 audio or the question text.
  `medapp/service.py`'s one prior leak (question text logged on an
  Answerer exception) is fixed, with a regression test
  (`tests/test_service.py::test_no_log_record_carries_the_request_body`).
- `uv run pytest` (316 tests), `uv run ruff check .` and `uv run mypy .` all
  pass clean on this branch.

**Not verified here and cannot be from this machine:** the CUDA/cuDNN half of
the build, anything through an actual GPU (`device=cuda`), the restart
behaviour under a real supervisor, or the VM/public-internet/Verify/latency
items below. See the status table.

### The restart check specifically

`docker run --restart=always` is the correct mechanism, confirmed by
`docker inspect` showing `RestartPolicy: {"Name":"always"}` on the container.
But on this machine's Docker Desktop instance (4.91.0 / engine 29.8.0,
immediately after a full data reset), the daemon never restarted the container
after a `SIGKILL` — nor a bare `alpine` container given the same flag, which
rules out the image. `docker events` also stopped reporting *any* events
afterwards, container or otherwise, which points at this daemon's event bus
rather than the restart policy itself. This is left **unverified**, not
assumed working: confirm it on the actual VM's Docker Engine (a normal Linux
install, not Docker Desktop) before checking that box.

## Left to do on the real infrastructure

None of this is achievable from a laptop — it needs the actual cloud account,
GPU quota and a publicly reachable VM.

1. **GPU quota.** Try Azure for Students first, per the ticket. If it grants no
   GPU cores, provision on UCloud, GCP or AWS instead (the case README's
   provider list).
2. **Build and run on the VM.**
   ```
   docker build -t medapp .          # CUDA base, large-v3, float16 by default
   docker run -d --restart=always -p 9054:9054 \
     -e MEDAPP_ROUTE_SUFFIX=<random-unguessable-suffix> \
     medapp
   ```
   Confirm `WhisperModel(..., device="cuda")` actually loads before trusting
   anything measured after it — the Dockerfile's header comment has the exact
   command. A CUDA/cuDNN version mismatch between `BASE_IMAGE` and the
   `ctranslate2` wheel `faster-whisper` installs fails loudly here, at import,
   not silently later.
3. **Reachability and Verify.** Confirm the root route answers from outside the
   VM's network, then run Verify against
   `http://<vm-host>:9054/predict<MEDAPP_ROUTE_SUFFIX>`. The route suffix is a
   deploy-time secret — set by `-e`, never baked into the image or committed.
4. **Restart-under-kill**, on the VM's real Docker Engine:
   ```
   docker kill -s SIGKILL <container>
   docker ps   # expect it back, restarted, within a few seconds
   ```
5. **Latency gate.**
   ```
   python local_evaluator.py --url http://<vm-host>:9054/predict<suffix>
   ```
   Zero timeouts, worst-case per-Conversation latency ≤ 40 s against the
   longest supplied Conversation. Tune `beam_size`, `compute_type` and
   `whisper_model` (all in `medapp/config.py::Settings`) against this and
   against dev timing error (`python -m scripts.asr_timing_error`) — do not
   guess the trade-off, measure it on the VM, the same way every other
   threshold in `Settings` was.
6. **Record the result.** Once decoding parameters and model size are fixed,
   add the measured worst-case latency and dev timing error as an inline
   comment beside `whisper_model`/`beam_size`/`compute_type` in
   `medapp/config.py`, in the same style as the existing rerank/entailment/span
   measurements there. Do not fill these in from a guess — every other
   Settings comment in this file is a real, cited measurement, and this one
   should be too.

## Status against the ticket's checklist

| Item | Status |
| --- | --- |
| GPU quota confirmed | Not started — needs the real account |
| Image builds (Mac CPU) | **Verified** |
| Image builds (VM CUDA) | Not verified — needs the real VM |
| VM answers publicly, Verify passes | Not started |
| Restarts after forced kill | **Not verified** — mechanism is correct, daemon behaviour unconfirmed (see above) |
| `local_evaluator.py --url` zero timeouts, ≤ 40 s worst case | Not started |
| Decoding params/model size fixed with measured numbers | Not started |
| Request bodies absent from logs | **Verified** |
| Largest Conversation body accepted | **Verified** |
