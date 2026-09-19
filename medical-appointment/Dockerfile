# The deployable unit. Weights are baked in at build time and the container
# never reaches the hub afterwards — HF_HUB_OFFLINE is set below and every
# model medapp/reranker.py and medapp/entailment.py load is
# already local_files_only. See docs/deployment.md for the VM runbook this
# image is built for.
#
# Two builds share this file:
#   VM (CUDA, the real target):
#     docker build -t medapp .
#   Mac (CPU, small model, build-only smoke test — no GPU to decode on):
#     docker build -t medapp:cpu-smoke \
#       --build-arg BASE_IMAGE=python:3.11-slim \
#       --build-arg DEVICE=cpu --build-arg COMPUTE_TYPE=int8 \
#       --build-arg WHISPER_MODEL=tiny.en .
#
# BASE_IMAGE's CUDA/cuDNN version has to match what the ctranslate2 wheel
# faster-whisper installs was built against — confirm this on the VM itself
# (`docker run --gpus all medapp python -c "from faster_whisper import
# WhisperModel; WhisperModel('tiny', device='cuda')"`) before trusting any
# latency measured on it; a mismatch here fails loudly at that import, not
# silently.
ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ARG DEVICE=cuda
ARG COMPUTE_TYPE=float16
ARG WHISPER_MODEL=large-v3

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    MEDAPP_DEVICE=${DEVICE} \
    MEDAPP_COMPUTE_TYPE=${COMPUTE_TYPE} \
    MEDAPP_WHISPER_MODEL=${WHISPER_MODEL} \
    PATH="/root/.local/bin:${PATH}"

# curl is the only package this needs beyond whichever base is passed in;
# both the CUDA/Ubuntu and the python-slim/Debian base are apt-based, so one
# install step covers both builds above. uv provisions its own interpreter
# next rather than relying on the base image's python3 package, whose version
# varies by distro release and is not guaranteed to satisfy
# pyproject.toml's `requires-python`.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN uv python install 3.11

WORKDIR /app

# The lockfile alone, so this layer only rebuilds when a dependency changes,
# not on every source edit.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY dtos.py utils.py example.py api.py ./
COPY medapp/ ./medapp/
COPY scripts/__init__.py scripts/fetch_models.py ./scripts/

# Bakes every weight the request path loads into the image, at the
# MEDAPP_WHISPER_MODEL this build was configured with above. HF_HUB_OFFLINE
# is set only afterwards: this step still needs the hub, everything past it
# must not.
RUN uv run python -m scripts.fetch_models

ENV HF_HUB_OFFLINE=1

EXPOSE 9054

# The venv's own interpreter, not `uv run`: `uv run` re-checks the lockfile's
# package metadata against the network on every invocation, even against an
# already-synced venv, which is exactly the reach-out this image must not make.
CMD [".venv/bin/python", "api.py"]
