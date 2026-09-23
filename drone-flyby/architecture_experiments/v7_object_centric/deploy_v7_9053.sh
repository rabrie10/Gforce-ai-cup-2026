#!/bin/bash
# V7 PUBLIC DEPLOYMENT to port 9053. RUN ONLY ON EXPLICIT OPERATOR APPROVAL.
# Rollback: /workspace/rollback_v6_9053.sh
set -euo pipefail
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
CKPT=/workspace/v7_e15_snapshot.pt
EXPECT=52b9fcef703d3fc5f35993f3088381f545b50d1091c862f52ff0c4df1cd091b8
test "$(sha256sum $CKPT | cut -d" " -f1)" = "$EXPECT" || { echo "CHECKPOINT SHA MISMATCH - ABORT"; exit 1; }
mkdir -p /workspace/v7-deploy-$STAMP
cp /workspace/logs/endpoint_v6_public_9053*.log /workspace/v7-deploy-$STAMP/ 2>/dev/null || true
OLD=$(lsof -ti:9053 || true); echo "old pid(s): $OLD" | tee /workspace/v7-deploy-$STAMP/replaced_pid.txt
if [ -n "$OLD" ]; then kill $OLD; sleep 6; fi
if [ -n "$(lsof -ti:9053 || true)" ]; then echo "port still busy"; lsof -ti:9053 | xargs -r kill -9; sleep 3; fi
cd /workspace/v7-release-candidate
source /workspace/env.sh >/dev/null 2>&1
export PYTHONPATH=/workspace/v7-release-candidate
export V5_ASSETS=/workspace/assets V6_ASSETS=/workspace/assets V5_DEVICE=cuda:0 V6_DEVICE=cuda:0
export YOLO_CONFIG_DIR=/tmp/yolo
export V6_ENSEMBLE=0
export V6_DIAGNOSTIC_CAPTURE=0
export V6_DETECTOR=$CKPT
export V6_CLASSIFY_MIN_PX=16
export V6_TARGET_MIN=0.5
export V6_DET_CONF=0.15
setsid nohup /workspace/venv/bin/python -m uvicorn v5.gpu.endpoint_v6:app --host 0.0.0.0 --port 9053 --workers 1 \
  > /workspace/logs/endpoint_v7_public_9053_$STAMP.log 2>&1 < /dev/null &
for i in $(seq 1 60); do sleep 2; curl -s --max-time 3 http://127.0.0.1:9053/ | grep -q "true" && break; done
echo "health:"; curl -s --max-time 5 http://127.0.0.1:9053/
echo; echo "manifest:"; curl -s --max-time 5 http://127.0.0.1:9053/metrics
echo; echo "new pid: $(lsof -ti:9053)"; echo "DEPLOY DONE $STAMP"
