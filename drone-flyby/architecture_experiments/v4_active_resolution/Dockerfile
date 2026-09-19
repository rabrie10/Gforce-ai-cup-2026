# Reuse the immutable, already-working V3 runtime and baked recognition weights.
# Building this image performs no model inference and changes no serving image.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
WORKDIR /experiment
COPY dtos.py utils.py local_evaluator.py ./
COPY v2 ./v2
COPY architecture_experiments/v4_active_resolution ./architecture_experiments/v4_active_resolution
ENV OMP_NUM_THREADS=4
ENV OPENBLAS_NUM_THREADS=1
ENV PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["python", "architecture_experiments/v4_active_resolution/phase_a.py"]
