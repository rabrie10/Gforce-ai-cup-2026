"""Reproduce the Baseline 1 forensic failure audit from frozen artifacts.

This script never loads the model, trains, calls an endpoint, or changes an
authoritative artifact.  It consumes saved local predictions, Helsinki ground
truth, scene metadata, and training logs, then writes only inside this audit
directory.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


AUDIT_DIR = Path(__file__).resolve().parent
DRONE_ROOT = AUDIT_DIR.parents[1]
REPO_ROOT = DRONE_ROOT.parent
ARTIFACTS = DRONE_ROOT / "training" / "artifacts"
SCENE = DRONE_ROOT / "scene_audit"
EVAL_PATH = ARTIFACTS / "evaluation_results.json"
TRAIN_PATH = ARTIFACTS / "training_results.csv"
MODEL_PATH = DRONE_ROOT / "models" / "drone_yolo11n_l0.pt"
FIGURES = AUDIT_DIR / "figures"
SOURCE_W, SOURCE_H = 3840, 2160
L0_W, L0_H = 960, 540
IOU_LOCALIZED = 0.50
IOU_WEAK = 0.10
CLASSES = (
    "hangar", "helicopter", "jet_plane", "large_launcher", "large_tower",
    "medium_launcher", "medium_plane", "mine_roller", "small_launcher",
    "small_plane", "small_tower", "ta-ta", "tank", "condor", "jammer",
    "spacecraft",
)
COLORS = {
    "D_correct_iou50": "#22863a",
    "A_missed": "#d73a49",
    "B_wrong_class_iou50": "#6f42c1",
    "C_correct_class_low_iou": "#f66a0a",
    "G_wrong_class_low_iou": "#dbab09",
    "E_background_fp": "#b31d28",
    "E_low_overlap_fp": "#e36209",
    "F_duplicate_or_alternative": "#8250df",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def aggregate_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.as_posix()):
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(sha256(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + bb - inter)


def pearson(xs, ys):
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    return float(np.corrcoef(np.asarray(xs, float), np.asarray(ys, float))[0, 1])


def dist(values) -> dict:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"count": 0, "min": None, "median": None, "mean": None, "p95": None, "max": None}
    return {
        "count": len(vals), "min": min(vals), "median": statistics.median(vals),
        "mean": statistics.mean(vals), "p95": float(np.percentile(vals, 95)), "max": max(vals),
    }


def fmt(value, digits=3):
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def size_bin(short_side: float) -> str:
    if short_side < 4:
        return "<4"
    if short_side < 8:
        return "4-<8"
    if short_side < 16:
        return "8-<16"
    if short_side < 32:
        return "16-<32"
    return ">=32"


def area_bin(area: float) -> str:
    if area < 32:
        return "<32"
    if area < 128:
        return "32-<128"
    if area < 512:
        return "128-<512"
    if area < 2048:
        return "512-<2048"
    return ">=2048"


def longest_false_run(values: list[bool]) -> int:
    best = current = 0
    for value in values:
        current = 0 if value else current + 1
        best = max(best, current)
    return best


def load_inputs():
    evaluation = read_json(EVAL_PATH)
    saved_predictions = evaluation["non_realtime"]["predictions"]
    scene_rows = read_csv(SCENE / "annotation_metrics.csv")
    scene_by_key = {(int(r["frame"]), r["class"]): r for r in scene_rows}
    gts, preds = [], []
    for frame in range(25):
        payload = read_json(DRONE_ROOT / "src" / "helsinki" / "annotations" / f"frame_{frame:06d}.json")
        for index, ann in enumerate(payload["annotations"]):
            box = [float(v) for v in ann["bbox"]]
            meta = scene_by_key[(frame, ann["object_id"])]
            l0_box = [v / 4.0 for v in box]
            short = min(l0_box[2] - l0_box[0], l0_box[3] - l0_box[1])
            area = (l0_box[2] - l0_box[0]) * (l0_box[3] - l0_box[1])
            gts.append({
                "gt_id": f"f{frame:02d}-g{index:02d}", "frame": frame,
                "object_id": ann["object_id"], "bbox": box, "l0_bbox": l0_box,
                "l0_short_side": short, "l0_area": area,
                "size_bin": size_bin(short), "area_bin": area_bin(area),
                "boundary": int(meta["boundary_touch"]),
                "cx_norm": ((box[0] + box[2]) / 2) / SOURCE_W,
                "cy_norm": ((box[1] + box[3]) / 2) / SOURCE_H,
            })
        for index, pred in enumerate(saved_predictions[str(frame)]):
            box = [float(v) for v in pred["bbox"]]
            preds.append({
                "pred_id": f"f{frame:02d}-p{index:02d}", "frame": frame,
                "object_id": pred["object_id"], "bbox": box,
                "confidence": float(pred["confidence"]),
            })
    return evaluation, gts, preds


def assign_matches(gts: list[dict], preds: list[dict]):
    matches, pred_rows = [], []
    for frame in range(25):
        fg = [g for g in gts if g["frame"] == frame]
        fp = [p for p in preds if p["frame"] == frame]
        pairs = []
        for gi, gt in enumerate(fg):
            for pi, pred in enumerate(fp):
                pairs.append((gi, pi, iou(gt["bbox"], pred["bbox"])))
        used_g, used_p, assigned = set(), set(), {}

        def stage(label, selector, sort_key):
            candidates = [x for x in pairs if x[0] not in used_g and x[1] not in used_p and selector(x)]
            for gi, pi, overlap in sorted(candidates, key=sort_key, reverse=True):
                if gi in used_g or pi in used_p:
                    continue
                used_g.add(gi); used_p.add(pi); assigned[gi] = (label, pi, overlap)

        stage("D_correct_iou50", lambda x: x[2] >= IOU_LOCALIZED and fg[x[0]]["object_id"] == fp[x[1]]["object_id"],
              lambda x: (fp[x[1]]["confidence"], x[2], -x[0], -x[1]))
        stage("B_wrong_class_iou50", lambda x: x[2] >= IOU_LOCALIZED and fg[x[0]]["object_id"] != fp[x[1]]["object_id"],
              lambda x: (x[2], fp[x[1]]["confidence"], -x[0], -x[1]))
        stage("C_correct_class_low_iou", lambda x: IOU_WEAK <= x[2] < IOU_LOCALIZED and fg[x[0]]["object_id"] == fp[x[1]]["object_id"],
              lambda x: (x[2], fp[x[1]]["confidence"], -x[0], -x[1]))
        stage("G_wrong_class_low_iou", lambda x: IOU_WEAK <= x[2] < IOU_LOCALIZED and fg[x[0]]["object_id"] != fp[x[1]]["object_id"],
              lambda x: (x[2], fp[x[1]]["confidence"], -x[0], -x[1]))

        for gi, gt in enumerate(fg):
            overlaps = [(iou(gt["bbox"], p["bbox"]), p) for p in fp]
            correct = [(o, p) for o, p in overlaps if p["object_id"] == gt["object_id"]]
            wrong = [(o, p) for o, p in overlaps if p["object_id"] != gt["object_id"]]
            best_any = max(overlaps, key=lambda x: (x[0], x[1]["confidence"]), default=(0.0, None))
            best_correct = max(correct, key=lambda x: (x[0], x[1]["confidence"]), default=(0.0, None))
            best_wrong = max(wrong, key=lambda x: (x[0], x[1]["confidence"]), default=(0.0, None))
            label, pi, overlap = assigned.get(gi, ("A_missed", None, 0.0))
            pred = fp[pi] if pi is not None else None
            row = dict(gt)
            row.update({
                "failure_type": label,
                "matched_pred_id": pred["pred_id"] if pred else "",
                "predicted_class": pred["object_id"] if pred else "no_detection",
                "confidence": pred["confidence"] if pred else None,
                "iou": overlap if pred else 0.0,
                "correct_class": int(bool(pred and pred["object_id"] == gt["object_id"])),
                "iou_ge_050": int(overlap >= IOU_LOCALIZED),
                "any_class_iou50": int(any(o >= IOU_LOCALIZED for o, _ in overlaps)),
                "correct_class_iou50": int(any(o >= IOU_LOCALIZED for o, p in correct)),
                "best_any_iou": best_any[0],
                "best_any_class": best_any[1]["object_id"] if best_any[1] else "no_detection",
                "best_any_confidence": best_any[1]["confidence"] if best_any[1] else None,
                "best_correct_iou": best_correct[0],
                "best_correct_confidence": best_correct[1]["confidence"] if best_correct[1] else None,
                "best_wrong_iou": best_wrong[0],
                "best_wrong_class": best_wrong[1]["object_id"] if best_wrong[1] else "",
                "best_wrong_confidence": best_wrong[1]["confidence"] if best_wrong[1] else None,
            })
            matches.append(row)

        for pi, pred in enumerate(fp):
            overlaps = [(iou(pred["bbox"], gt["bbox"]), gt) for gt in fg]
            best = max(overlaps, key=lambda x: x[0], default=(0.0, None))
            assigned_gt = next((fg[gi] for gi, value in assigned.items() if value[1] == pi), None)
            if assigned_gt:
                status = assigned[next(gi for gi, value in assigned.items() if value[1] == pi)][0]
            elif best[0] >= IOU_LOCALIZED:
                status = "F_duplicate_or_alternative"
            elif best[0] >= IOU_WEAK:
                status = "E_low_overlap_fp"
            else:
                status = "E_background_fp"
            pred_rows.append({
                **pred, "prediction_status": status,
                "matched_gt_id": assigned_gt["gt_id"] if assigned_gt else "",
                "best_gt_iou": best[0],
                "best_gt_class": best[1]["object_id"] if best[1] else "",
                "is_true_positive": int(status == "D_correct_iou50"),
            })
    return matches, pred_rows


def summarize_groups(matches, field, ordered_values):
    rows = []
    for value in ordered_values:
        group = [r for r in matches if r[field] == value]
        localized = sum(r["any_class_iou50"] for r in group)
        correct = sum(r["correct_class_iou50"] for r in group)
        class_signal = sum(r["best_correct_iou"] >= IOU_WEAK for r in group)
        conf = [r["confidence"] for r in group if r["confidence"] is not None]
        ious = [r["iou"] for r in group if r["matched_pred_id"]]
        rows.append({
            field: value, "gt_count": len(group),
            "any_class_iou50_count": localized,
            "any_class_iou50_recall": localized / len(group) if group else None,
            "correct_class_overlap_count": class_signal,
            "correct_class_overlap_recall": class_signal / len(group) if group else None,
            "correct_class_iou50_count": correct,
            "correct_class_iou50_recall": correct / len(group) if group else None,
            "matched_confidence_median": statistics.median(conf) if conf else None,
            "matched_confidence_mean": statistics.mean(conf) if conf else None,
            "matched_iou_median": statistics.median(ious) if ious else None,
            "failure_A_missed": sum(r["failure_type"] == "A_missed" for r in group),
            "failure_B_wrong_class_iou50": sum(r["failure_type"] == "B_wrong_class_iou50" for r in group),
            "failure_C_correct_class_low_iou": sum(r["failure_type"] == "C_correct_class_low_iou" for r in group),
            "failure_D_correct_iou50": sum(r["failure_type"] == "D_correct_iou50" for r in group),
            "failure_G_wrong_class_low_iou": sum(r["failure_type"] == "G_wrong_class_low_iou" for r in group),
        })
    return rows


def temporal_summary(matches):
    result = []
    for name in CLASSES:
        rows = sorted((r for r in matches if r["object_id"] == name), key=lambda r: r["frame"])
        detected = [r["failure_type"] == "D_correct_iou50" for r in rows]
        labels = [r["predicted_class"] for r in rows if r["predicted_class"] != "no_detection"]
        correct_conf = [r["confidence"] for r in rows if r["failure_type"] == "D_correct_iou50"]
        best_labels = [r["best_any_class"] for r in rows if r["best_any_iou"] >= IOU_WEAK]
        dominant = Counter(best_labels).most_common(1)[0] if best_labels else ("no_detection", 0)
        result.append({
            "object_id": name, "first_frame": rows[0]["frame"], "last_frame": rows[-1]["frame"],
            "visible_frames": len(rows), "correct_iou50_frames": sum(detected),
            "correct_iou50_rate": sum(detected) / len(rows),
            "any_class_iou50_frames": sum(r["any_class_iou50"] for r in rows),
            "any_class_iou50_rate": sum(r["any_class_iou50"] for r in rows) / len(rows),
            "longest_correct_detection_gap": longest_false_run(detected),
            "correct_confidence_mean": statistics.mean(correct_conf) if correct_conf else None,
            "correct_confidence_std": statistics.pstdev(correct_conf) if len(correct_conf) > 1 else (0.0 if correct_conf else None),
            "matched_iou_median": statistics.median([r["iou"] for r in rows if r["matched_pred_id"]]) if any(r["matched_pred_id"] for r in rows) else None,
            "dominant_best_overlap_label": dominant[0],
            "dominant_label_fraction": dominant[1] / len(best_labels) if best_labels else None,
            "class_stability_fraction": max(Counter(labels).values()) / len(labels) if labels else None,
            "confidence_vs_short_side_r": pearson([r["l0_short_side"] for r in rows if r["best_correct_confidence"] is not None], [r["best_correct_confidence"] for r in rows if r["best_correct_confidence"] is not None]),
            "confidence_vs_frame_r": pearson([r["frame"] for r in rows if r["best_correct_confidence"] is not None], [r["best_correct_confidence"] for r in rows if r["best_correct_confidence"] is not None]),
            "confidence_vs_x_r": pearson([r["cx_norm"] for r in rows if r["best_correct_confidence"] is not None], [r["best_correct_confidence"] for r in rows if r["best_correct_confidence"] is not None]),
        })
    return result


def confidence_summary(pred_rows):
    rows = []
    groups = [("ALL", pred_rows), ("TRUE_POSITIVE", [r for r in pred_rows if r["is_true_positive"]]),
              ("INCORRECT", [r for r in pred_rows if not r["is_true_positive"]])]
    groups += [(name, [r for r in pred_rows if r["object_id"] == name]) for name in CLASSES]
    for name, group in groups:
        d = dist([r["confidence"] for r in group])
        rows.append({"group": name, **d, "tp_count": sum(r["is_true_positive"] for r in group),
                     "incorrect_count": sum(not r["is_true_positive"] for r in group)})
    return rows


def training_summary():
    rows = read_csv(TRAIN_PATH)
    numeric = [{k: float(v) for k, v in row.items()} for row in rows]
    best = max(numeric, key=lambda r: r["metrics/mAP50(B)"])
    first10, last10 = numeric[:10], numeric[-10:]
    mean = lambda rs, key: statistics.mean(r[key] for r in rs)
    return {
        "epochs": len(numeric), "best_map50_epoch": int(best["epoch"]),
        "best_same_sequence_map50": best["metrics/mAP50(B)"],
        "final_same_sequence_map50": numeric[-1]["metrics/mAP50(B)"],
        "final_precision": numeric[-1]["metrics/precision(B)"],
        "final_recall": numeric[-1]["metrics/recall(B)"],
        "train_box_loss_first10_mean": mean(first10, "train/box_loss"),
        "train_box_loss_last10_mean": mean(last10, "train/box_loss"),
        "train_cls_loss_first10_mean": mean(first10, "train/cls_loss"),
        "train_cls_loss_last10_mean": mean(last10, "train/cls_loss"),
        "val_box_loss_first10_mean": mean(first10, "val/box_loss"),
        "val_box_loss_last10_mean": mean(last10, "val/box_loss"),
        "val_cls_loss_first10_mean": mean(first10, "val/cls_loss"),
        "val_cls_loss_last10_mean": mean(last10, "val/cls_loss"),
        "map50_epoch50": numeric[49]["metrics/mAP50(B)"],
        "map50_gain_epoch50_to60": numeric[-1]["metrics/mAP50(B)"] - numeric[49]["metrics/mAP50(B)"],
        "interpretation": "Same-sequence fit continues improving late; these curves cannot measure external generalization because train and validation contain the same 25 frames.",
    }


def font(size=14):
    candidates = [Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/segoeui.ttf")]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def save_bar(path, title, labels, values, ylabel="count", colors=None, width=1100, height=600):
    image = Image.new("RGB", (width, height), "white"); draw = ImageDraw.Draw(image)
    title_font, label_font, small = font(22), font(14), font(11)
    draw.text((30, 20), title, fill="#111", font=title_font)
    left, top, right, bottom = 75, 75, width - 25, height - 125
    draw.line((left, top, left, bottom, right, bottom), fill="#333", width=2)
    maxv = max(values) if values else 1
    maxv = max(maxv, 1)
    gap = (right - left) / max(len(values), 1)
    bw = gap * 0.68
    for i, (lab, val) in enumerate(zip(labels, values)):
        x1 = left + i * gap + (gap - bw) / 2; x2 = x1 + bw
        y1 = bottom - (bottom - top) * val / maxv
        color = colors[i] if colors else "#2f81f7"
        draw.rectangle((x1, y1, x2, bottom), fill=color, outline="#222")
        draw.text((x1, max(top, y1 - 18)), f"{val:.2f}" if isinstance(val, float) and not float(val).is_integer() else str(int(val)), fill="#111", font=small)
        text = lab if len(lab) <= 16 else lab[:15] + "…"
        draw.text((x1, bottom + 8), text, fill="#111", font=small)
    draw.text((8, top), ylabel, fill="#333", font=label_font)
    image.save(path)


def save_grouped_bar(path, title, labels, series, ylabel="rate"):
    width, height = 1150, 620
    image = Image.new("RGB", (width, height), "white"); draw = ImageDraw.Draw(image)
    draw.text((30, 20), title, fill="#111", font=font(22))
    left, top, right, bottom = 75, 80, width - 30, height - 130
    draw.line((left, top, left, bottom, right, bottom), fill="#333", width=2)
    maxv = max((v for _, vals, _ in series for v in vals), default=1) or 1
    gap = (right-left)/max(len(labels), 1); sub = gap/(len(series)+1)
    for sidx, (name, vals, color) in enumerate(series):
        for i, val in enumerate(vals):
            x1 = left+i*gap+sub*(sidx+0.35); x2=x1+sub*0.75
            y1=bottom-(bottom-top)*val/maxv
            draw.rectangle((x1,y1,x2,bottom),fill=color,outline="#333")
        draw.rectangle((left+sidx*190, height-42, left+sidx*190+18, height-24), fill=color)
        draw.text((left+sidx*190+25,height-44),name,fill="#111",font=font(12))
    for i, lab in enumerate(labels):
        draw.text((left+i*gap+5,bottom+8),lab,fill="#111",font=font(11))
    draw.text((8, top), ylabel, fill="#333", font=font(14)); image.save(path)


def save_scatter(path, title, xs, ys, groups, xlabel, ylabel):
    width,height=950,650; image=Image.new("RGB",(width,height),"white"); draw=ImageDraw.Draw(image)
    draw.text((30,20),title,fill="#111",font=font(22)); left,top,right,bottom=85,75,width-35,height-70
    draw.line((left,top,left,bottom,right,bottom),fill="#333",width=2)
    xmin,xmax=min(xs,default=0),max(xs,default=1); ymin,ymax=min(ys,default=0),max(ys,default=1)
    if xmax==xmin:xmax=xmin+1
    if ymax==ymin:ymax=ymin+1
    for x,y,g in zip(xs,ys,groups):
        px=left+(x-xmin)/(xmax-xmin)*(right-left); py=bottom-(y-ymin)/(ymax-ymin)*(bottom-top)
        color="#22863a" if g else "#d73a49"; draw.ellipse((px-4,py-4,px+4,py+4),fill=color,outline="#222")
    draw.text((left,bottom+25),f"{xlabel} ({xmin:.1f} to {xmax:.1f})",fill="#333",font=font(13))
    draw.text((8,top),f"{ylabel} ({ymin:.2f} to {ymax:.2f})",fill="#333",font=font(13)); image.save(path)


def save_heatmap(path, matrix, rows, cols, title):
    cell=42; left=180; top=80; width=left+cell*len(cols)+30; height=top+cell*len(rows)+160
    image=Image.new("RGB",(width,height),"white"); draw=ImageDraw.Draw(image)
    draw.text((20,20),title,fill="#111",font=font(20)); vmax=max((max(r) for r in matrix),default=1) or 1
    for i,row in enumerate(matrix):
        draw.text((5,top+i*cell+12),rows[i],fill="#111",font=font(11))
        for j,val in enumerate(row):
            intensity=int(245-190*val/vmax); color=(255,intensity,intensity)
            x=left+j*cell;y=top+i*cell
            draw.rectangle((x,y,x+cell,y+cell),fill=color,outline="#ddd")
            if val: draw.text((x+14,y+12),str(val),fill="#111",font=font(11))
    for j,col in enumerate(cols):
        draw.text((left+j*cell+6,top+cell*len(rows)+8),col[:5],fill="#111",font=font(9))
    draw.text((left, height-55),"Columns abbreviated; full matrix is in confusion_matrix.csv",fill="#444",font=font(12)); image.save(path)


def save_timeline(path, matches):
    width,height=1150,670; image=Image.new("RGB",(width,height),"white"); draw=ImageDraw.Draw(image)
    draw.text((25,15),"Per-object temporal perception timeline",fill="#111",font=font(21))
    left,top=190,65; cw=35; rh=34
    for f in range(25): draw.text((left+f*cw+9,top-20),str(f),fill="#333",font=font(9))
    lookup={(r["object_id"],r["frame"]):r for r in matches}
    for i,name in enumerate(CLASSES):
        y=top+i*rh; draw.text((5,y+8),name,fill="#111",font=font(12))
        for f in range(25):
            x=left+f*cw; row=lookup.get((name,f))
            color="#eeeeee" if row is None else COLORS[row["failure_type"]]
            draw.rectangle((x,y,x+cw-2,y+rh-3),fill=color,outline="#fff")
    legend=[("not visible","#eeeeee"),("correct IoU>=.5",COLORS["D_correct_iou50"]),("missed",COLORS["A_missed"]),("wrong class",COLORS["B_wrong_class_iou50"]),("low-IoU",COLORS["C_correct_class_low_iou"])]
    for i,(lab,color) in enumerate(legend):
        x=20+i*210; draw.rectangle((x,height-42,x+18,height-24),fill=color); draw.text((x+25,height-44),lab,fill="#111",font=font(11))
    image.save(path)


def crop_panel(frame, gt, pred, label):
    source=Image.open(DRONE_ROOT/"src"/"helsinki"/"images"/f"frame_{frame:06d}.png").convert("RGB")
    source=source.resize((L0_W,L0_H),Image.Resampling.BOX)
    if gt:
        box=[v/4 for v in gt["bbox"]]
    else:
        box=[v/4 for v in pred["bbox"]]
    pad=max(35,min(130,max(box[2]-box[0],box[3]-box[1])*2.5))
    cx=(box[0]+box[2])/2;cy=(box[1]+box[3])/2
    crop_box=(max(0,int(cx-pad)),max(0,int(cy-pad)),min(L0_W,int(cx+pad)),min(L0_H,int(cy+pad)))
    crop=source.crop(crop_box).resize((330,250),Image.Resampling.NEAREST)
    draw=ImageDraw.Draw(crop)
    def transform(b):
        b=[v/4 for v in b]; sx=330/(crop_box[2]-crop_box[0]);sy=250/(crop_box[3]-crop_box[1])
        return [(b[0]-crop_box[0])*sx,(b[1]-crop_box[1])*sy,(b[2]-crop_box[0])*sx,(b[3]-crop_box[1])*sy]
    if gt: draw.rectangle(transform(gt["bbox"]),outline="#00e5ff",width=3)
    if pred: draw.rectangle(transform(pred["bbox"]),outline="#ff2bd6",width=3)
    canvas=Image.new("RGB",(330,300),"white");canvas.paste(crop,(0,50));d=ImageDraw.Draw(canvas)
    d.text((5,5),label,fill="#111",font=font(13));d.text((5,27),f"frame {frame}; cyan=GT, magenta=prediction",fill="#444",font=font(10))
    return canvas


def save_representatives(path, matches, pred_rows):
    by_pred={p["pred_id"]:p for p in pred_rows}
    correct=max((r for r in matches if r["failure_type"]=="D_correct_iou50"),key=lambda r:r["confidence"])
    tiny=min((r for r in matches if r["failure_type"]=="A_missed"),key=lambda r:r["l0_short_side"])
    # Use the explicitly important large_tower counterexample: its physical
    # label is "large", yet every L0 appearance is missed.
    large=max((r for r in matches if r["object_id"]=="large_tower"),key=lambda r:r["l0_short_side"])
    wrong=max((r for r in matches if r["failure_type"] in ("B_wrong_class_iou50","G_wrong_class_low_iou")),key=lambda r:r["confidence"],default=None)
    loc=max((r for r in matches if r["failure_type"]=="C_correct_class_low_iou"),key=lambda r:r["confidence"],default=None)
    fp=max((p for p in pred_rows if p["prediction_status"]=="E_background_fp"),key=lambda p:p["confidence"])
    specs=[(correct,by_pred.get(correct["matched_pred_id"]),f"Strong success: {correct['object_id']} {correct['confidence']:.2f}"),
           (tiny,None,f"Tiny miss: {tiny['object_id']} {tiny['l0_short_side']:.1f}px"),
           (large,by_pred.get(large["matched_pred_id"]),f"Large failure: {large['object_id']} {large['l0_short_side']:.1f}px"),
           (wrong,by_pred.get(wrong["matched_pred_id"]) if wrong else None,f"Wrong class: {wrong['object_id']}→{wrong['predicted_class']}" if wrong else "Wrong class: none"),
           (loc,by_pred.get(loc["matched_pred_id"]) if loc else None,f"Localization failure: {loc['object_id']} IoU {loc['iou']:.2f}" if loc else "Localization failure: none"),
           (None,fp,f"Background FP: {fp['object_id']} {fp['confidence']:.2f}")]
    panels=[]
    for gt,pred,label in specs:
        if gt is None and pred is None:
            panel=Image.new("RGB",(330,300),"white");ImageDraw.Draw(panel).text((10,10),label,fill="#111",font=font(14))
        else:
            frame=gt["frame"] if gt else pred["frame"]; panel=crop_panel(frame,gt,pred,label)
        panels.append(panel)
    canvas=Image.new("RGB",(990,600),"#dddddd")
    for i,panel in enumerate(panels):canvas.paste(panel,((i%3)*330,(i//3)*300))
    canvas.save(path)


def main() -> int:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    evaluation,gts,preds=load_inputs(); matches,pred_rows=assign_matches(gts,preds)
    class_rows=summarize_groups(matches,"object_id",list(CLASSES))
    ap=evaluation["non_realtime"]["per_class_ap50"]
    for row in class_rows: row["ap50"] = ap[row["object_id"]]
    size_order=["<4","4-<8","8-<16","16-<32",">=32"]
    size_rows=summarize_groups(matches,"size_bin",size_order)
    area_order=["<32","32-<128","128-<512","512-<2048",">=2048"]
    area_rows=summarize_groups(matches,"area_bin",area_order)
    temp_rows=temporal_summary(matches); conf_rows=confidence_summary(pred_rows); train=training_summary()

    gt_fields=["gt_id","frame","object_id","bbox","l0_bbox","l0_short_side","l0_area","size_bin","area_bin","boundary","cx_norm","cy_norm"]
    match_fields=gt_fields+["failure_type","matched_pred_id","predicted_class","confidence","iou","correct_class","iou_ge_050","any_class_iou50","correct_class_iou50","best_any_iou","best_any_class","best_any_confidence","best_correct_iou","best_correct_confidence","best_wrong_iou","best_wrong_class","best_wrong_confidence"]
    write_csv(AUDIT_DIR/"gt_object_inventory.csv",gts,gt_fields)
    write_csv(AUDIT_DIR/"prediction_matches.csv",matches,match_fields)
    write_csv(AUDIT_DIR/"prediction_inventory.csv",pred_rows,["pred_id","frame","object_id","bbox","confidence","prediction_status","matched_gt_id","best_gt_iou","best_gt_class","is_true_positive"])
    write_csv(AUDIT_DIR/"failure_cases.csv",[r for r in matches if r["failure_type"]!="D_correct_iou50"],match_fields)
    write_csv(AUDIT_DIR/"class_summary.csv",class_rows)
    write_csv(AUDIT_DIR/"size_summary.csv",size_rows)
    write_csv(AUDIT_DIR/"area_summary.csv",area_rows)
    write_csv(AUDIT_DIR/"temporal_summary.csv",temp_rows)
    write_csv(AUDIT_DIR/"confidence_summary.csv",conf_rows)

    confusion_cols=list(CLASSES)+["no_detection"]
    confusion=[]; confusion_csv=[]
    for actual in CLASSES:
        row=[]; out={"actual_class":actual}
        for predicted in confusion_cols:
            count=sum(r["object_id"]==actual and r["predicted_class"]==predicted for r in matches)
            row.append(count);out[predicted]=count
        confusion.append(row);confusion_csv.append(out)
    write_csv(AUDIT_DIR/"confusion_matrix.csv",confusion_csv,["actual_class"]+confusion_cols)

    failure_gt=Counter(r["failure_type"] for r in matches); failure_pred=Counter(r["prediction_status"] for r in pred_rows)
    tp=[p for p in pred_rows if p["is_true_positive"]]; incorrect=[p for p in pred_rows if not p["is_true_positive"]]
    boundary={str(v):{"count":sum(r["boundary"]==v for r in matches),"correct_iou50":sum(r["boundary"]==v and r["correct_class_iou50"] for r in matches)} for v in (0,1)}
    for val in boundary.values():val["recall"]=val["correct_iou50"]/val["count"]
    zero_classes=[name for name in CLASSES if ap[name]==0]
    zero_preds=[p for p in pred_rows if p["object_id"] in zero_classes]
    protected=[]
    for directory in [DRONE_ROOT/"evaluator_probes",SCENE,DRONE_ROOT/"architecture_experiments"/"motion_oracle",DRONE_ROOT/"architecture_experiments"/"motion_similarity_oracle",DRONE_ROOT/"architecture_experiments"/"motion_affine_oracle"]:
        protected.extend(
            p for p in directory.rglob("*")
            if p.is_file()
            and not any(part.startswith(".") or part == "__pycache__" for part in p.relative_to(directory).parts)
        )
    key_inputs=[EVAL_PATH,ARTIFACTS/"validation_results.json",ARTIFACTS/"training_metadata.json",TRAIN_PATH,MODEL_PATH]
    source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO_ROOT,text=True).strip()
    manifest={"aggregate_protected_sha256":aggregate_hash(protected),"protected_file_count":len(protected),"key_inputs":{p.relative_to(REPO_ROOT).as_posix():{"sha256":sha256(p),"bytes":p.stat().st_size} for p in key_inputs}}
    write_json(AUDIT_DIR/"preservation_manifest.json",manifest)

    taxonomy = [
        {"failure_mode":"L0 information loss","evidence":"60/60 GT objects below 8 px short side had no IoU>=0.50 localization; recall rises to 65/66 at 16-<32 px.","severity":"CRITICAL","diagnosis":"CONFIRMED (local); hosted contribution SUPPORTED","can_baseline1_prove":"Local size dependence yes; hosted causality no","likely_remedy":"L1/L2 observation for recognition","next_experiment_required":"Oracle crop x resolution recognition"},
        {"failure_mode":"Physical-instance/reference memorization","evidence":"One physical instance per class; same-sequence mAP 0.3837 versus hosted raw 0.001619.","severity":"CRITICAL","diagnosis":"SUPPORTED","can_baseline1_prove":"Cannot separate instance, scene, and domain factors","likely_remedy":"Transfer-oriented frozen/exemplar recognition","next_experiment_required":"Identical-crop representation comparison"},
        {"failure_mode":"Background/context memorization","evidence":"Catastrophic external gap is compatible, but no controlled context intervention was run.","severity":"POTENTIALLY CRITICAL","diagnosis":"PLAUSIBLE","can_baseline1_prove":"No","likely_remedy":"Object-centric representation and context robustness","next_experiment_required":"Controlled object/context perturbation"},
        {"failure_mode":"Class confusion","evidence":"10 GT outcomes are localized at IoU>=0.50 but wrong-class: 8 tank and 2 spacecraft.","severity":"HIGH FOR AFFECTED CLASSES","diagnosis":"CONFIRMED","can_baseline1_prove":"Yes on Helsinki only","likely_remedy":"Transfer-oriented recognition / prototypes","next_experiment_required":"Identical-crop representation comparison"},
        {"failure_mode":"Localization weakness","evidence":"Only 1 GT has correct class with 0.10<=IoU<0.50; localized successes have high IoU. 126/259 GT are localized by some class.","severity":"LOW AS PRIMARY CAUSE","diagnosis":"NOT SUPPORTED AS PRIMARY","can_baseline1_prove":"For detected Helsinki objects","likely_remedy":"Do not prioritize box-loss tuning","next_experiment_required":"L0 localization-only/objectness test"},
        {"failure_mode":"Confidence/ranking weakness","evidence":"13 true positives are below 0.10 confidence, but 107 incorrect predictions are also below 0.10; no prediction uses any of the nine AP-zero class IDs at the 0.01 floor.","severity":"MODERATE / SECONDARY","diagnosis":"CONFIRMED","can_baseline1_prove":"Yes locally; not hosted","likely_remedy":"Calibrated quality gates after perception improves","next_experiment_required":"No threshold sweep yet"},
        {"failure_mode":"Class imbalance","evidence":"Class count versus AP Pearson r=0.279; 25-example classes include both perfect and zero-recall outcomes.","severity":"SECONDARY/UNKNOWN","diagnosis":"NOT SUPPORTED AS PRIMARY","can_baseline1_prove":"Weak correlation only","likely_remedy":"Only revisit after representation test","next_experiment_required":"No dedicated experiment yet"},
        {"failure_mode":"Boundary truncation","evidence":"Correct IoU>=0.50 recall is 6/23 (26.1%) at boundary versus 110/236 (46.6%) nonboundary; size/class are confounders.","severity":"MODERATE","diagnosis":"SUPPORTED","can_baseline1_prove":"Association, not causal effect","likely_remedy":"Temporal state and uncertainty near boundaries","next_experiment_required":"Stratify later perception experiment"},
        {"failure_mode":"Temporal intermittency","evidence":"Recognized classes have gaps of 1-2 frames, but nine classes are never correctly recognized.","severity":"MODERATE; NOT A RESCUE FOR ZERO CLASSES","diagnosis":"CONFIRMED","can_baseline1_prove":"Yes locally","likely_remedy":"Temporal state only after identity signal exists","next_experiment_required":"Perception-first; then temporal simulation"},
        {"failure_mode":"Runtime skipping","evidence":"Local realtime sent 8/25 frames; sent-frame predictions exactly match offline.","severity":"HIGH FOR LOCAL REALTIME SCORE","diagnosis":"CONFIRMED","can_baseline1_prove":"Yes locally","likely_remedy":"Later latency/camera scheduling work","next_experiment_required":"Not before perception experiments"},
        {"failure_mode":"API/infrastructure","evidence":"Hosted endpoint connected, predicted, validation completed, and reported no errors.","severity":"LOW","diagnosis":"NOT SUPPORTED","can_baseline1_prove":"Rules out gross failure, not all platform differences","likely_remedy":"None indicated","next_experiment_required":"None"},
        {"failure_mode":"Hosted GT-level attribution","evidence":"No hosted GT, frames, classes, boxes, or historical request log is available.","severity":"LIMITATION","diagnosis":"NOT OBSERVABLE","can_baseline1_prove":"No","likely_remedy":"None within current rules","next_experiment_required":"Use allowed local discriminating experiments"},
    ]
    write_csv(AUDIT_DIR/"failure_taxonomy.csv", taxonomy)

    next_experiments = [
        {"priority":1,"experiment":"Oracle crop x L0/L1/L2 recognition","question":"Does true source resolution restore class separability?","decision":"Determines whether zoom is necessary and sufficient for identity.","minimum_design":"Same GT crop geometry and deterministic padding at each true source level; report per-class recognition, not detection."},
        {"priority":2,"experiment":"Identical-crop representation comparison","question":"Which representation transfers class evidence without one-instance fine-tuning?","decision":"Select supervised YOLO features, frozen pretrained embeddings, prototypes, or a justified hybrid.","minimum_design":"Exactly the same crops and folds for all methods; instance/scene-held controls where possible."},
        {"priority":3,"experiment":"L0 localization-only/objectness","question":"Can L0 say where to zoom even when it cannot identify the class?","decision":"Determines whether active zoom can be proposal-driven.","minimum_design":"Class-agnostic recall versus size at a fixed proposal budget; do not optimize identity."},
        {"priority":4,"experiment":"Controlled context-dependence probe","question":"Does confidence follow the target or Helsinki background/context?","decision":"Separates object evidence from context memorization.","minimum_design":"Small deterministic object translations, object masking, and matched context controls; compare original detections."},
    ]
    architecture = [
        {"component":"L0 discovery/objectness","addresses":"131/259 misses and need for zoom proposals","evidence":"Identity fails below 8 px; localization can still be evaluated independently.","gate":"L0 objectness reaches useful recall at a fixed proposal budget.","fallback":"Sparse deterministic scan/refresh schedule."},
        {"component":"Active L1/L2 zoom + scheduler","addresses":"Severe L0 information loss","evidence":"60/60 sub-8 px objects are unlocalized locally; prior scene audit establishes more source information at L1/L2.","gate":"Oracle resolution test shows a material, repeatable recognition gain.","fallback":"L1-only bounded refresh; selective L2 only if its marginal gain is proven."},
        {"component":"Transfer-oriented recognition (frozen embeddings/prototypes; supervised hybrid only if earned)","addresses":"Hosted collapse, nine never-emitted classes, tank/spacecraft confusion","evidence":"Same-reference fit does not transfer; only 7/16 class IDs are emitted locally.","gate":"Wins identical-crop external/held-control comparison with calibrated ranking.","fallback":"Best frozen representation plus nearest prototype; omit supervised branch."},
        {"component":"Affine GMC + temporal object state with staleness/uncertainty","addresses":"1-2 frame gaps, boundary loss, skipped observations","evidence":"Reviewed affine oracle is 97.94% one-step IoU>=0.50; perception sometimes fires intermittently for 7 classes.","gate":"Perception produces correct seeds and an offline temporal simulation improves recall without identity contamination.","fallback":"Short TTL hold-last state using affine only; no identity creation."},
        {"component":"Confidence quality gates","addresses":"Low-confidence TP/FP mixture and two high-confidence spacecraft confusions","evidence":"TP median 0.879 versus incorrect 0.0178, but 13 TPs are below 0.10 and 2 wrong-class results are near 0.79.","gate":"Held-control calibration improves risk/coverage without erasing rare correct classes.","fallback":"Conservative uncertainty flag; preserve raw evidence for temporal fusion."},
    ]

    results={
        "audit_version":1,"source_commit":source_commit,
        "matching_policy":{"stage_order":["correct class IoU>=0.50","wrong class IoU>=0.50","correct class 0.10<=IoU<0.50","wrong class 0.10<=IoU<0.50","missed"],"one_to_one_within_frame":True,"weak_overlap_threshold":IOU_WEAK},
        "inputs":{"gt_objects":len(gts),"predictions":len(preds),"frames":25,"local_map50":evaluation["non_realtime"]["map50"],"hosted_raw_score":0.0016190390038119638,"hosted_attempt_uuid":"7b96ded345424b96b6f43ab28ee14bbf","hosted_evidence_source":"handoff-provided; no repository request log found"},
        "outcomes":{"gt_failure_counts":dict(failure_gt),"prediction_status_counts":dict(failure_pred),"correct_iou50_recall":failure_gt["D_correct_iou50"]/len(gts),"any_class_iou50_recall":sum(r["any_class_iou50"] for r in matches)/len(gts),"background_fp_count":failure_pred["E_background_fp"],"duplicate_or_alternative_count":failure_pred["F_duplicate_or_alternative"]},
        "confidence":{"true_positive":dist([p["confidence"] for p in tp]),"incorrect":dist([p["confidence"] for p in incorrect]),"confidence_iou_pearson":pearson([p["confidence"] for p in pred_rows],[p["best_gt_iou"] for p in pred_rows]),"zero_ap_class_prediction_count":len(zero_preds),"zero_ap_class_prediction_confidence":dist([p["confidence"] for p in zero_preds])},
        "boundary":boundary,"size_summary":size_rows,"class_summary":class_rows,"temporal_summary":temp_rows,
        "training":train,"failure_taxonomy":taxonomy,"next_experiments":next_experiments,"architecture_v2_hypothesis":architecture,
        "hosted_observability":"Hosted GT-level failure attribution is not observable from the available evidence.",
        "environment":{"python":sys.version.split()[0],"platform":platform.platform(),"numpy":np.__version__,"pillow":importlib.metadata.version("pillow")},
        "preservation":manifest,
    }
    write_json(AUDIT_DIR/"audit_results.json",results)

    save_bar(FIGURES/"01_gt_count_by_class.png","GT object count by class",[r["object_id"] for r in class_rows],[r["gt_count"] for r in class_rows],"GT count")
    save_grouped_bar(FIGURES/"02_ap_recall_by_class.png","Same-sequence AP and direct recall by class",[r["object_id"][:8] for r in class_rows],[("AP@0.50",[r["ap50"] for r in class_rows],"#2f81f7"),("correct IoU>=.50 recall",[r["correct_class_iou50_recall"] for r in class_rows],"#22863a")])
    save_grouped_bar(FIGURES/"03_recall_by_size.png","Recall versus L0 short-side bin",size_order,[("any-class localization",[r["any_class_iou50_recall"] for r in size_rows],"#8250df"),("correct class + IoU",[r["correct_class_iou50_recall"] for r in size_rows],"#22863a")])
    save_bar(FIGURES/"04_confidence_correct_incorrect.png","Prediction confidence: true positives versus incorrect",["TP median","incorrect median","TP mean","incorrect mean"],[dist([p["confidence"] for p in tp])["median"],dist([p["confidence"] for p in incorrect])["median"],dist([p["confidence"] for p in tp])["mean"],dist([p["confidence"] for p in incorrect])["mean"]],"confidence",["#22863a","#d73a49","#22863a","#d73a49"])
    assoc=[r for r in matches if r["confidence"] is not None]
    save_scatter(FIGURES/"05_confidence_vs_size.png","Matched prediction confidence versus L0 target size",[r["l0_short_side"] for r in assoc],[r["confidence"] for r in assoc],[r["failure_type"]=="D_correct_iou50" for r in assoc],"L0 short side (px)","confidence")
    failure_labels=["D correct","A missed","B wrong@.5","C loc<.5","G wrong<.5","E bg FP","E loose FP","F duplicate"]
    failure_values=[failure_gt["D_correct_iou50"],failure_gt["A_missed"],failure_gt["B_wrong_class_iou50"],failure_gt["C_correct_class_low_iou"],failure_gt["G_wrong_class_low_iou"],failure_pred["E_background_fp"],failure_pred["E_low_overlap_fp"],failure_pred["F_duplicate_or_alternative"]]
    save_bar(FIGURES/"06_failure_type_counts.png","GT outcomes and prediction-only errors",failure_labels,failure_values,"count",["#22863a","#d73a49","#6f42c1","#f66a0a","#dbab09","#b31d28","#e36209","#8250df"])
    save_heatmap(FIGURES/"07_confusion_no_detection.png",confusion,list(CLASSES),confusion_cols,"Assigned class outcome (no detection separated)")
    save_timeline(FIGURES/"08_temporal_timeline.png",matches)
    save_representatives(FIGURES/"09_representative_overlays.png",matches,pred_rows)

    print(json.dumps({"status":"PASS","gt":len(gts),"predictions":len(preds),"gt_outcomes":dict(failure_gt),"prediction_status":dict(failure_pred),"source_commit":source_commit},indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
