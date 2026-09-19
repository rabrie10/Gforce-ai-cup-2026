import os, time, json, subprocess
import numpy as np, cv2
from v5.discovery import Discovery
from v5.visual_expert import VisualExpert, DinoEncoder

ASSETS = os.environ.get("V5_ASSETS", "/workspace/assets")
IMG = "/workspace/hosted-v3/images/00060_80324c03bcc65ff1d04c510e_180324c03bcc65ff1d04c510e_60_0_1920_1080.png"
REGION = [0, 0, 3840, 2160]


def vram():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).decode().strip().split("\n")[0]
        return int(out)
    except Exception:
        return -1


def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    dt = (time.perf_counter() - t) / iters * 1000
    return dt


def main():
    dev = os.environ.get("V5_DEVICE", "cuda:0")
    print(f"=== device={dev}  VRAM_before={vram()}MiB ===")
    img = cv2.imread(IMG)
    assert img is not None and img.shape[:2] == (540, 960), img.shape

    # --- Discovery on GPU ---
    disc = Discovery(ASSETS, budget=32, device=dev, threads=2)
    print("discovery ONNX providers:", disc.session.get_providers()[0], "input_size", disc.size)
    raw, sel, stages = disc.propose(img, REGION)
    d_ms = bench(lambda: disc.propose(img, REGION), iters=20)
    print(f"DISCOVERY propose(): {d_ms:.1f} ms/frame  raw={len(raw)} selected={len(sel)}  VRAM={vram()}MiB")

    # --- DINOv2 encoder on GPU: batch sweep ---
    enc = DinoEncoder(ASSETS, size=224, device=dev, threads=2)
    print("dinov2 ONNX providers:", enc.session.get_providers()[0], "size", enc.size)
    results = {"discovery_ms": round(d_ms, 1), "discovery_raw": len(raw), "discovery_selected": len(sel), "encoder": {}}
    for b in [32, 64, 128, 256]:
        crops = [np.random.randint(0, 255, (60, 60, 3), np.uint8) for _ in range(b)]
        try:
            ms = bench(lambda: enc.encode(crops), iters=10, warmup=3)
            results["encoder"][b] = {"ms": round(ms, 1), "ms_per_crop": round(ms / b, 2), "vram": vram()}
            print(f"DINOv2 encode batch={b:4d}: {ms:7.1f} ms  ({ms/b:.2f} ms/crop)  VRAM={vram()}MiB")
        except Exception as e:
            results["encoder"][b] = {"error": str(e)[:120]}
            print(f"DINOv2 encode batch={b}: ERROR {e}")

    # --- Full VisualExpert.classify on real selected crops (32) ---
    expert = VisualExpert(ASSETS, device=dev, threads=2)
    boxes = [p["local_box"] for p in sel]
    res, feats, vt = expert.classify(img, boxes)
    full_ms = bench(lambda: expert.classify(img, boxes), iters=10, warmup=3)
    print(f"VisualExpert.classify({len(boxes)} real crops): {full_ms:.1f} ms  crop_ms~{vt['crop_ms']:.1f} enc+head_ms~{vt['encoder_head_ms']:.1f}")
    results["classify32_ms"] = round(full_ms, 1)

    # --- crude integrated frame estimate (discovery + classify32), no temporal/http ---
    results["integrated_disc+cls32_ms"] = round(d_ms + full_ms, 1)
    print(f"INTEGRATED discovery+classify32 (no temporal/http): {d_ms+full_ms:.1f} ms")
    results["vram_peak_MiB"] = vram()
    json.dump(results, open("/workspace/results/gpu_bench.json", "w"), indent=2)
    print("WROTE /workspace/results/gpu_bench.json")


if __name__ == "__main__":
    main()
