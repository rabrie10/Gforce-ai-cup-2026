#!/bin/sh
# Run only after the full training script exits successfully, without training
# contention. No hosted service is contacted: these are saved-input/local replays.
set -eu
ROOT=/srv/gforce/v5-perception-repo/drone-flyby
ASSETS=/srv/gforce/v5-perception-assets
RESULTS=/srv/gforce/v5-perception-results
HOSTED=/srv/gforce/telemetry-v3-live/v3-live_20260919_120849_c48481a0
test -f "$ASSETS/manifest.json"
docker run --rm --name gforce-v5-hosted-review --network none --cpus 2 --memory 6g \
  -v "$ROOT:/work:ro" -v "$ASSETS:/assets:ro" -v "$RESULTS:/results" \
  -v "$HOSTED:/hosted-v3:ro" gforce-drone:v5-runtime \
  -m v5.replay --inputs /hosted-v3 > "$RESULTS/hosted_replay.log" 2>&1
sh "$ROOT/v5/deploy_local.sh"
docker run --rm --name gforce-v5-local-evaluation --network host --cpus 2 --memory 6g \
  -v "$ROOT:/work:ro" -v "$ASSETS:/assets:ro" -v "$RESULTS:/results" \
  gforce-drone:v5-runtime -m v5.evaluate --url http://127.0.0.1:9353/predict \
  > "$RESULTS/local_evaluation.log" 2>&1
