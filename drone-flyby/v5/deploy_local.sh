#!/bin/sh
# Explicit isolated endpoint launch; never changes an existing service.
set -eu
ROOT=/srv/gforce/v5-perception-repo/drone-flyby
ASSETS=/srv/gforce/v5-perception-assets
RESULTS=/srv/gforce/v5-perception-results
PORT=9353
docker ps --format '{{.Names}} {{.Image}} {{.Ports}}'
if ss -ltnH | awk '{print $4}' | grep -q ":${PORT}$"; then
  echo "Refusing occupied port ${PORT}" >&2
  exit 1
fi
test -f "$ASSETS/manifest.json"
docker run -d --name gforce-v5-endpoint --cpus 2 --memory 6g \
  --network bridge -p "127.0.0.1:${PORT}:${PORT}" \
  --read-only --tmpfs /tmp:rw,size=256m \
  -e V5_DEVICE=cpu -e V5_CANDIDATES=32 -e V5_ACTIVE_CAMERA=0 \
  -v "$ROOT:/work:ro" -v "$ASSETS:/assets:ro" \
  -v /srv/gforce/v5-perception-cache:/cache:ro \
  gforce-drone:v5-runtime
echo "Isolated V5 is bound to 127.0.0.1:${PORT}; production was not changed."
