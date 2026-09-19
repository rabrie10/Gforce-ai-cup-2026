# Azure feasibility measurements — 2026-09-19

The full requested architecture is CPU-bound on the existing machine. Keep
YOLO11s, DINOv2 ViT-S/14, full 224-pixel object crops and a configurable 32-candidate
budget. Do not provision hardware or replace the production VM automatically.

## Measured host and isolation

- AMD EPYC 7763 host CPU; VM exposes 4 vCPUs, 2 cores × 2 SMT threads.
- 15.57 GiB usable RAM (16 GiB class); about 12 GiB available before experiments.
- No CUDA GPU: `torch.cuda.is_available() == False`, device count 0; no NVIDIA
  device was found in PCI inspection.
- Initial inference benchmarks: isolated container, **2-vCPU quota**, two model
  threads; no competing V5 training. This is deliberately not presented as the
  maximum possible throughput of all four vCPUs.
- Full training: isolated **3-vCPU aggregate quota**, 9 GiB memory ceiling;
  observed about 299% CPU and 2.93 GiB RAM. Other services remained running.
- YOLO11s initial epoch: roughly 5–6 minutes while full DINO feature extraction
  shares the training allocation. Thirty epochs project 2.5–3 hours if that rate
  persists; this is an estimate, not a completed training duration.

## Independent model latency

YOLO11s general pretrained PyTorch inference on recorded hosted L0 image 60:

| Detector input | p50 ms | p95 ms |
|---|---:|---:|
| 640 | 191.84 | 194.86 |
| 960 | 381.41 | 385.75 |

These are pre-adaptation detector timings, not final endpoint latency. V5 serves
the adapted detector through its ONNX export, measured separately after training.

DINOv2 ViT-S/14 ONNX, full **224×224** crops (8 measured runs, 2 warmups):

| Crops | p50 ms | p95 ms | CPU % | Peak process RSS MiB |
|---:|---:|---:|---:|---:|
| 8 | 583.56 | 584.99 | 199.74 | 1009.68 |
| 16 | 1195.87 | 1207.27 | 199.72 | 1016.00 |
| 32 | 2455.57 | 2457.12 | 199.75 | 1204.57 |
| 64 | 5273.81 | 5278.80 | 199.73 | 1601.93 |

RSS includes the export/verification process and its PyTorch copy, not just the
served ONNX encoder. The capacity test uses fixed-shape sample tensors; the
earlier PyTorch test used image crops. Real complete HTTP requests remain the
required endpoint measurement. Export maximum absolute difference from PyTorch
was approximately 2.0e-5. No quantization or replacement backbone was used.

For context, 8-crop ONNX p95 was 106.43 ms at 98 pixels and 214.05 ms at 140 pixels.
After the additional-compute update, these remain diagnostic configurations; they
do not replace the full 224-pixel primary configuration.

## Throughput implication and remaining measurement

The encoder alone at 32 crops consumes about 2.46 seconds per request. Against
333 ms frames, a sustained encoder-only upper-bound service rate is about 0.407
requests/s: roughly **86% of frames would be skipped**, before discovery, image
decode, tracking and HTTP overhead. At 64 crops the encoder alone exceeds the
3333 ms request timeout. These are workload-based estimates, not official replay
results. Actual candidate counts can be smaller than the configured ceiling.

Complete endpoint p50/p95 and the actual skipped-frame fraction must come from
`v5.evaluate` against the isolated endpoint with the trained weights. They are
not yet measured at the initial feasibility stage. Arithmetic sums of component
timings must not be reported as measured endpoint latency.

The bottleneck is model CPU arithmetic, especially batched DINOv2 inference, not
RAM capacity. Reducing memory or adding disk bandwidth would not solve it. More
modern dedicated CPU cores should help, but linear scaling is not established.
A dedicated 16-core modern CPU with at least 32 GiB RAM is a useful CPU comparison;
there is no measured basis to promise it meets 333 ms for 32 full-size crops.

## Recommended separate Azure resource

**Proposed minimum practical GPU configuration to benchmark: Standard_NC4as_T4_v3**
(4 vCPUs, 28 GiB RAM, one NVIDIA T4 with 16 GiB GPU memory). NC8as_T4_v3 gives more
CPU/RAM for decode, data loading and training orchestration if available. The same
models and batch workloads consist primarily of neural convolutions/matrix
multiplications and should benefit materially from CUDA; this is a hardware
inference, not a measured T4 speedup. Validate p95, peak GPU memory and numerical
agreement before calling it sufficient for the deadline.

The minimum GPU memory for this exact implementation has not been measured; a
16 GiB T4 is the concrete practical test recommendation, not a proven capacity
lower bound. A100/H100 hardware is not justified as the initial requirement.

Official specifications:
- https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/ncast4v3-series
- https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/compute-optimized/falsv6-series

A new NVIDIA-capable container must install GPU ONNX Runtime instead of the CPU
package and expose the driver through `--gpus all`. `V5_DEVICE=cuda:0` selects
CUDA explicitly and fails if it is unavailable. Runtime model calls make no
network requests. Provisioning, subscription quota changes and changes to the
production VM require explicit user approval; none have been performed.
