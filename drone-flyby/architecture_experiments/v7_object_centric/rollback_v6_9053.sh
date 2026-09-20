#!/bin/bash
# ROLLBACK to the known-good V6 public configuration (ms1 detector, original gates).
set -euo pipefail
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OLD=$(lsof -ti:9053 || true); [ -n "$OLD" ] && kill $OLD && sleep 6
[ -n "$(lsof -ti:9053 || true)" ] && { lsof -ti:9053 | xargs -r kill -9; sleep 3; }
cd /workspace/v6-diagnostic-release-8ae3b97
source /workspace/env.sh >/dev/null 2>&1
export PYTHONPATH=/workspace/v6-diagnostic-release-8ae3b97
export V5_ASSETS=/workspace/assets V6_ASSETS=/workspace/assets V5_DEVICE=cuda:0 V6_DEVICE=cuda:0
export YOLO_CONFIG_DIR=/tmp/yolo
export V6_ENSEMBLE=0
export V6_DETECTOR=/workspace/training/ms1/weights/best.pt
export V6_DIAGNOSTIC_CAPTURE=1 V6_DIAGNOSTIC_DIR=/workspace/v6-diagnostics-8ae3b97
export V6_DIAGNOSTIC_WINDOW_SIZE=50 V6_DIAGNOSTIC_PER_WINDOW=2 V6_DIAGNOSTIC_PER_LEVEL=10
export V6_DIAGNOSTIC_QUEUE_SIZE=8 V6_DIAGNOSTIC_QUOTA_BYTES=52428800
setsid nohup /workspace/venv/bin/python -m uvicorn v5.gpu.endpoint_v6:app --host 0.0.0.0 --port 9053 --workers 1 \
  > /workspace/logs/endpoint_v6_rollback_9053_$STAMP.log 2>&1 < /dev/null &
for i in $(seq 1 60); do sleep 2; curl -s --max-time 3 http://127.0.0.1:9053/ | grep -q "true" && break; done
curl -s --max-time 5 http://127.0.0.1:9053/metrics; echo; echo "ROLLBACK DONE pid $(lsof -ti:9053)"
