#!/bin/sh
set -eu
export PYTHONUNBUFFERED=1
python -m v5.data > /results/data.log 2>&1
python -m v5.train_visual > /results/visual_training_224.log 2>&1 &
visual_pid=$!
python -m v5.train_discovery > /results/discovery_training.log 2>&1
wait "$visual_pid"
python -m v5.package > /results/manifest.log 2>&1
