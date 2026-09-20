"""Build failure_matrix.csv, summaries and overlays/contact sheets from raw_replay_results.json + frozen annotations."""
import json, glob, os, csv, cv2, numpy as np, collections
H = os.path.dirname(os.path.abspath(__file__)); AT = os.path.join(H, "..", "attempt_d09922e7a50f4cd88682f2d895fbef91")
R = json.load(open(f"{H}/raw_replay_results.json")); A = json.load(open(f"{H}/manual_annotations_frozen.json"))


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix*iy
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i/u if u > 0 else 0


def pad(b, s):
    cx, cy, w, h = (b[0]+b[2])/2, (b[1]+b[3])/2, (b[2]-b[0])*s, (b[3]-b[1])*s
    return [cx-w/2, cy-h/2, cx+w/2, cy+h/2]


# hand-assigned primary failure category + reason (from replay numbers; see COMPLETE_HOSTED_L1L2_AUDIT.md)
CAT = {
 "f52_heli": ("D_threshold", "raw top-1 proposal (score .112, IoU .43) removed by production det_conf=0.15; classifier would say helicopter (.59), target_p .91"),
 "f55_heli": ("B_raw_weak", "only a very low-score raw proposal (.008, rank 25, IoU .41, body only); classifier target_p .99/helicopter .84 if it were proposed"),
 "f6_armor": ("B_raw_miss", "no raw proposal within IoU .07 at any confidence; large obvious 49x26 px vehicle"),
 "f7_armor": ("B_raw_miss", "same object next frame; best IoU .05"),
 "f100_armor": ("C_loose_box_E_class", "raw score .744 top-1 covers vehicle (78x56 box vs 58x26 obj, IoU .34 tight); class head says large_launcher .27/hangar .26 (tank-like object)"),
 "f204_veh_small": ("OK_detected", "raw .861 IoU .51; final emitted small_launcher .179 with IoU .51 (class unverified)"),
 "f204_launcher_large": ("E_bg_reject", "post-merge candidate IoU .60 survives; bg head target_p .48 < .5 -> no track (class head large_launcher .67)"),
 "f4_armor": ("B_raw_miss", "no raw proposal (best IoU .01) for 24x13 px L1 vehicle"),
 "f50_armor": ("B_raw_miss", "best IoU 0.00 (top raw .324 is elsewhere)"),
 "f50_dark_b": ("B_raw_miss", "best IoU 0.00"),
 "f50_dark_a": ("B_raw_miss", "LOW-confidence tiny target; best IoU 0.00"),
 "f51_armor": ("B_raw_miss", "best IoU 0.00 (top raw .150 elsewhere)"),
 "f101_armor": ("E_minside_gate", "post-merge cand IoU .45 (score .70, box 38x21.8 px) but min side 21.8 < classify_min_px=22 -> never classified -> ev=0 -> never emitted; forced classify target_p .61 (loose box)"),
 "f105_armor": ("E_minside_gate", "post-merge cand IoU .61 (score .287) but 18 px min side < classify_min_px=22 -> never classified -> ev=0 -> never emitted (frame has 0 predictions); forced classify gives target_p .83"),
}
rows = []
for t in R["targets"]:
    g = next(x for x in A["targets"] if x["target_id"] == t["target_id"]); fr = R["frames"][f'{g["frame_index"]}_{g["level"]}']
    raws = [p["local_box"] for p in fr["raw"]]; gt = g["box_local_xyxy"]
    b13 = max([iou(pad(gt, 1.3), b) for b in raws] + [0])
    c = CAT[t["target_id"]]
    cg = t.get("cls_gt_box") or {}; cb = t.get("cls_best_postmerge") or t.get("cls_best_raw") or {}
    rows.append(dict(target=t["target_id"], frame=g["frame_index"], level=g["level"], family=g["family"], manual_conf=g["existence_confidence"],
        difficulty=g["difficulty"], px_wxh=f'{g["px_w"]}x{g["px_h"]}', short_side=g["short_side"], edge_clipped=g["edge_clipped"] or "no",
        raw_best_iou_tight=round(t["raw_best_iou"], 2), raw_best_iou_pad130=round(b13, 2),
        raw_score=round(t["raw_best_score"] or 0, 3) if t["raw_best_iou"] >= .3 else "",
        raw_rank=t["raw_best_rank"] if t["raw_best_iou"] >= .3 else "", raw_n_iou50=t["n_raw_iou50"], raw_iou_at_conf015=round(t["raw_best_iou_conf015"], 2),
        postmerge_best_iou=round(t["postmerge_stored_best_iou"], 2),
        classifier_target_p_on_candidate=round(cb["target_probability"], 2) if cb else "",
        classifier_top1=(cb["top3"][0]["class"] + f'{cb["top3"][0]["score"]:.2f}') if cb else "",
        classifier_target_p_on_GTbox=round(cg["target_probability"], 2) if cg else "",
        final_best_iou=round(t["final_best_iou_this_frame"], 2), primary_failure=c[0], reason=c[1]))
with open(f"{H}/failure_matrix.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
S = collections.defaultdict(collections.Counter)
for r in rows:
    S["difficulty"][r["difficulty"]] += 1; S["family"][r["family"]] += 1; S["level"][f'L{r["level"]}'] += 1
    S["failure_stage"][r["primary_failure"]] += 1
    S["difficulty x failure"][f'{r["difficulty"]} | {r["primary_failure"]}'] += 1
    S["level x failure"][f'L{r["level"]} | {r["primary_failure"]}'] += 1
    S["family x failure"][f'{r["family"]} | {r["primary_failure"]}'] += 1
    sz = "short<16" if r["short_side"] < 16 else "short16-30" if r["short_side"] < 30 else "short>=30"
    S["size x raw_found(IoU>=.3 tight or pad1.3)"][sz + " | " + ("yes" if max(r["raw_best_iou_tight"], r["raw_best_iou_pad130"]) >= .3 else "no")] += 1
json.dump({k: dict(v) for k, v in S.items()}, open(f"{H}/summary_counts.json", "w"), indent=1)

os.makedirs(f"{H}/overlays_audit", exist_ok=True)


def load(fr, lv): return cv2.imread(glob.glob(f"{AT}/captures/*_{fr}_{lv}_*.png")[0])


def rect(im, b, col, th=1, txt=None):
    p = [int(round(v)) for v in b]; cv2.rectangle(im, (p[0], p[1]), (p[2], p[3]), col, th)
    if txt: cv2.putText(im, txt, (p[0], max(10, p[1]-3)), cv2.FONT_HERSHEY_SIMPLEX, .38, col, 1, cv2.LINE_AA)


def label(img, txt):
    img = img.copy(); cv2.rectangle(img, (0, 0), (190, 16), (0, 0, 0), -1)
    cv2.putText(img, txt, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1, cv2.LINE_AA); return img


M, GRN, CY, OR, RED = (255, 0, 255), (0, 255, 0), (255, 255, 0), (0, 140, 255), (0, 0, 255)
gts = collections.defaultdict(list)
for g in A["targets"]: gts[(g["frame_index"], g["level"])].append(g)
keys = sorted(R["frames"], key=lambda k: int(k.split("_")[0]))
sheets = {k: [] for k in ("orig", "gt", "raw", "postmerge", "final")}
for k in keys:
    fi, lv = map(int, k.split("_")); fr = R["frames"][k]; im = load(fi, lv); reg = fr["region"]; s = (reg[2]-reg[0])/960
    sheets["orig"].append(label(im, f"f{fi} L{lv}"))
    g = im.copy()
    for t in gts[(fi, lv)]: rect(g, t["box_local_xyxy"], M, 2, t["existence_confidence"][0] + ":" + t["target_id"].split("_", 1)[1][:8])
    sheets["gt"].append(label(g, f"f{fi} L{lv} manual GT")); cv2.imwrite(f"{H}/overlays_audit/f{fi}_L{lv}_gt.png", g)
    r = g.copy()
    for p in reversed(fr["raw"][:12]): rect(r, p["local_box"], GRN if p["score"] >= .15 else CY, 1, f'{p["score"]:.2f}')
    sheets["raw"].append(label(r, f"f{fi} L{lv} RAW top12 grn>=.15")); cv2.imwrite(f"{H}/overlays_audit/f{fi}_L{lv}_raw_top12.png", r)
    pm = g.copy()
    for p in fr["post_merge_stored"]: rect(pm, p["local_box"], OR, 1, f'{p["score"]:.2f}')
    sheets["postmerge"].append(label(pm, f"f{fi} L{lv} POSTMERGE")); cv2.imwrite(f"{H}/overlays_audit/f{fi}_L{lv}_postmerge.png", pm)
    fn = g.copy()
    for p in fr["predictions"]:
        b = p["bbox"]; gb = [b[0]*3840, b[1]*2160, b[2]*3840, b[3]*2160]
        lb = [(gb[0]-reg[0])/s, (gb[1]-reg[1])/s, (gb[2]-reg[0])/s, (gb[3]-reg[1])/s]
        if lb[2] > 0 and lb[3] > 0 and lb[0] < 960 and lb[1] < 540: rect(fn, lb, RED, 2, f'{p["object_id"][:9]} {p["confidence"]:.2f}')
    sheets["final"].append(label(fn, f"f{fi} L{lv} FINAL in view")); cv2.imwrite(f"{H}/overlays_audit/f{fi}_L{lv}_final.png", fn)
names = dict(orig="contact_sheet_1_original.png", gt="contact_sheet_2_manual_boxes.png", raw="contact_sheet_3_raw_ms1.png",
             postmerge="contact_sheet_4_postmerge.png", final="contact_sheet_5_final.png")
for kname, lst in sheets.items():
    ts = [cv2.resize(x, (480, 270), interpolation=cv2.INTER_AREA) for x in lst]
    cv2.imwrite(f"{H}/{names[kname]}", np.vstack([np.hstack(ts[i:i+5]) for i in range(0, 20, 5)]))
for r in rows: print(r["target"], r["difficulty"], r["px_wxh"], "rawT", r["raw_best_iou_tight"], "rawP", r["raw_best_iou_pad130"], "pm", r["postmerge_best_iou"], r["primary_failure"])
print(json.dumps({k: dict(v) for k, v in S.items() if k in ("difficulty", "failure_stage", "difficulty x failure", "size x raw_found(IoU>=.3 tight or pad1.3)")}, indent=1))
