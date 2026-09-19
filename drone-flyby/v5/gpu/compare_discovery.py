import json, os, shutil, cv2
import torch
from ultralytics import YOLO
from v5.discovery import Discovery

OUT = "/results/epochlive"; os.makedirs(OUT, exist_ok=True)
src_best = "/assets/training/discovery/weights/best.pt"
shutil.copy2(src_best, f"{OUT}/discovery.pt")
ck = torch.load(f"{OUT}/discovery.pt", map_location="cpu", weights_only=False)
epoch = int(ck.get("epoch", -1)) + 1
torch.set_num_threads(2)
YOLO(f"{OUT}/discovery.pt").export(format="onnx", imgsz=640, batch=1, dynamic=False, simplify=False, opset=17, device="cpu")

disc = Discovery(OUT, budget=32, device="cpu", threads=2)
H = "/hosted-v3/images"
files = {60: "00060_80324c03bcc65ff1d04c510e_180324c03bcc65ff1d04c510e_60_0_1920_1080.png",
         125: "00125_80324c03bcc65ff1d04c510e_80324c03bcc65ff1d04c510e_125_0_1920_1080.png"}
region = [0, 0, 3840, 2160]
manual = json.load(open("/results/manual_targets.json"))["targets"]
by_cap = {}
for t in manual:
    by_cap.setdefault(t["capture"], []).append(t)
RES = "/results/hosted-epoch3"


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0., x2-x1), max(0., y2-y1); inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.


def match(raw, mb):
    best = max(([iou(mb, c["local_box"]), c] for c in raw), key=lambda z: z[0], default=[0, None])
    return best


print(f"=== DISCOVERY COMPARISON: epoch3 (snapshot) vs epoch{epoch} (current best) ===")
print("target | e3:bestIoU/rawRank/candRank/sel32 | eN:bestIoU/rawRank/candRank/sel32 | e3_raw_total -> eN_raw_total")
res = {"compared_epoch": epoch, "rows": []}
for cap, fn in files.items():
    img = cv2.imread(os.path.join(H, fn))
    rawN, selN, stagesN = disc.propose(img, region)
    old = json.load(open(f"{RES}/hosted_{cap:05d}_candidates.json"))["candidates"]
    for t in by_cap[cap]:
        mb = t["box"]
        io, co = match(old, mb)
        iN, cN = match(rawN, mb)
        oldr = (round(io, 3), co["raw_rank"] if co else None, co["candidate_rank"] if co else None, bool(co["selected"]) if co else False)
        newr = (round(iN, 3), cN["raw_rank"] if cN else None, cN["candidate_rank"] if cN else None, bool(cN["selected"]) if cN else False)
        print(f"{t['id']} ({t['class_guess'][:14]}) | e3 {oldr[0]}/{oldr[1]}/{oldr[2]}/{oldr[3]} | e{epoch} {newr[0]}/{newr[1]}/{newr[2]}/{newr[3]}")
        res["rows"].append({"target": t["id"], "epoch3": {"iou": oldr[0], "raw_rank": oldr[1], "cand_rank": oldr[2], "selected32": oldr[3]},
                            f"epoch{epoch}": {"iou": newr[0], "raw_rank": newr[1], "cand_rank": newr[2], "selected32": newr[3]}})
    print(f"  [{cap}] raw proposals: epoch3={len(old)} -> epoch{epoch}={len(rawN)} (selected32 both capped at 32)")
json.dump(res, open("/results/discovery_epoch_comparison.json", "w"), indent=2)
print("WROTE /results/discovery_epoch_comparison.json")
