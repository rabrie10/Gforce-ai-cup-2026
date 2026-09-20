# Frozen installed inference stack; V5 adds build/training dependencies separately.
FROM gforce-drone:v3-82ce749
USER root
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir onnx==1.20.1 scikit-learn==1.7.2
ENV V5_AZURE_VM=1 TORCH_HOME=/cache/torch YOLO_CONFIG_DIR=/cache/yolo OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 V5_ASSETS=/assets
WORKDIR /work
ENTRYPOINT ["python"]
CMD ["-m", "uvicorn", "v5.endpoint:app", "--host", "0.0.0.0", "--port", "9353", "--workers", "1"]
