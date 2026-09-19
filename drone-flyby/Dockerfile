FROM python:3.11-slim

# opencv needs libGL and libglib even in headless use.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The frozen backbone's ImageNet weights are baked in at build time. The
# competition rules forbid cloud calls from the request path, and a startup
# download would also make a cold container depend on outbound network right
# when it is least affordable.
ENV TORCH_HOME=/app/.torch
RUN python -c "import torchvision; \
    torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)"

# The server, the pipeline and its assets. The sample data under src/ is not
# copied: it is 400 MB and only local_evaluator.py needs it.
COPY dtos.py utils.py detector.py example.py api.py ./
COPY v2 ./v2
COPY models ./models

# Match the four-vCPU deployment budget; torch otherwise oversubscribes.
ENV DRONE_TORCH_THREADS=4
ENV OMP_NUM_THREADS=4
# Telemetry is opt-in. Switch it on for a validation attempt with
#   -e DRONE_TELEMETRY=1 -v /srv/gforce/telemetry:/app/telemetry
ENV DRONE_TELEMETRY=0

EXPOSE 9053

CMD ["python", "api.py"]
