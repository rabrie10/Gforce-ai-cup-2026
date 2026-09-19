"""V6 multiscale active-perception pipeline.

Camera observation (L0/L1/L2) -> fine-tuned one-class discovery -> predictive
object memory with global-motion compensation -> cross-level class evidence
(DINOv2) -> global-retention emission -> target-blind coverage + targeted L2
re-observation. Strictly causal. Official 16 classes only.

Injected into nothing: this is a standalone pipeline with .predict(request) matching
the endpoint contract, reusing v2 geometry + the DINOv2 RejectingExpert. Does NOT edit
existing v5 files.
"""
import os, time, threading, logging
from collections import OrderedDict
import numpy as np

from dtos import (OBJECT_CLASSES, DroneFlybyPredictResponseDto, DroneFlybyPredictionDto,
                  ALLOWED_RESOLUTION_LEVELS, MAXIMUM_CENTER_DELTA_PIXELS, SOURCE_REGION_SIZES,
                  FULL_FRAME_CENTER)
from utils import decode_view
from v2.geometry import iou, view_to_source
from v5.gpu.discovery_v54 import RejectingExpert
from ultralytics import YOLO

W, H = 3840, 2160
VW, VH = 960, 540


def region_for(level, cx, cy):
    w, h = SOURCE_REGION_SIZES[level]
    x1 = int(min(max(cx - w // 2, 0), W - w)); y1 = int(min(max(cy - h // 2, 0), H - h))
    return (x1, y1, x1 + w, y1 + h)


def center_bounds(level):
    w, h = SOURCE_REGION_SIZES[level]
    return (w // 2, W - w // 2, h // 2, H - h // 2)


class Track:
    __slots__ = ("id", "cx", "cy", "w", "h", "vx", "vy", "ev", "target", "best_level",
                 "last_idx", "misses", "feat", "unc")

    def __init__(self, tid, cx, cy, w, h, idx, level):
        self.id = tid; self.cx = cx; self.cy = cy; self.w = w; self.h = h
        self.vx = 0.0; self.vy = 0.0
        self.ev = np.zeros(16, np.float32)  # accumulated class evidence
        self.target = 0.5; self.best_level = level; self.last_idx = idx
        self.misses = 0; self.feat = None; self.unc = 60.0

    def box(self):
        return [self.cx - self.w / 2, self.cy - self.h / 2, self.cx + self.w / 2, self.cy + self.h / 2]


class State:
    def __init__(self):
        self.tracks = []
        self.next_id = 0
        self.last_idx = -1
        self.gvx = 0.0; self.gvy = 0.0  # estimated global scene drift per frame-index
        self.g_init = False
        self.cam_level = 0; self.cam_cx = FULL_FRAME_CENTER[0]; self.cam_cy = FULL_FRAME_CENTER[1]
        self.tour = 0
        self.region_seen = {}  # waypoint idx -> last frame idx observed


class Config:
    def __init__(self):
        self.assets = os.getenv("V5_ASSETS", "/workspace/assets")
        self.device = os.getenv("V5_DEVICE", "cuda:0")
        self.detector = os.getenv("V6_DETECTOR", "/workspace/training/ms1/weights/best.pt")
        self.det_conf = float(os.getenv("V6_DET_CONF", "0.15"))
        self.det_imgsz = int(os.getenv("V6_DET_IMGSZ", "960"))
        self.max_tracks = 300
        self.ttl = int(os.getenv("V6_TTL", "8"))
        self.active_camera = os.getenv("V6_ACTIVE_CAMERA", "1") == "1"
        self.min_emit_conf = float(os.getenv("V6_MIN_EMIT", "0.05"))
        self.classify_min_px = 22


# Broad L1 coverage tiles over the whole 3840x2160 frame (fine L2 handled via track refine).
COVERAGE_WAYPOINTS = [(1, 960, 540), (1, 2880, 540), (1, 2880, 1620), (1, 960, 1620)]


class V6Pipeline:
    def __init__(self, config=None):
        self.cfg = config or Config()
        self.det = YOLO(self.cfg.detector)
        self.dev = 0 if self.cfg.device.startswith("cuda") else "cpu"
        self.expert = RejectingExpert(self.cfg.assets, device=self.cfg.device)
        self.states = OrderedDict()
        self.lock = threading.RLock()
        self.last_diagnostics = {}
        self.manifest = {"pipeline": "v6", "detector": self.cfg.detector}
        # warmup
        z = np.zeros((VH, VW, 3), np.uint8)
        self._detect(z, (0, 0, W, H))
        self.expert.classify(z, [[100, 100, 140, 140]])

    def empty(self, r):
        return DroneFlybyPredictResponseDto(request_id=r.request_id, frame=r.frame, annotations=[])

    def _detect(self, view, region):
        r = self.det.predict(view, conf=self.cfg.det_conf, imgsz=self.cfg.det_imgsz,
                             verbose=False, device=self.dev)[0]
        out = []
        if r.boxes is not None and len(r.boxes):
            for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                lb = [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
                sb = list(view_to_source(lb, region))
                out.append({"local_box": lb, "source_box": sb, "score": float(c)})
        return out

    def predict(self, request):
        with self.lock:
            start = time.perf_counter()
            try:
                return self._predict(request, start)
            except Exception as exc:
                logging.exception("V6 frame failed")
                self.states.pop(request.sequence_id, None)
                self.last_diagnostics = {"error": f"{type(exc).__name__}: {exc}"}
                return self.empty(request)

    def _predict(self, r, start):
        st = self.states.setdefault(r.sequence_id, State())
        self.states.move_to_end(r.sequence_id)
        while len(self.states) > 4:
            self.states.popitem(last=False)
        if r.frame_index <= st.last_idx:
            self.last_diagnostics = {"reason": "out_of_order"}
            return self.empty(r)
        image = decode_view(r.view)
        if image.shape[:2] != (VH, VW):
            raise ValueError("view must be 960x540")
        region = list(r.view.source_region_xyxy)
        level = r.view.resolution_level
        gap = (r.frame_index - st.last_idx) if st.last_idx >= 0 else 1

        # 1. predict tracks forward (per-track velocity + global scene drift)
        for t in st.tracks:
            t.cx += (t.vx + st.gvx) * gap
            t.cy += (t.vy + st.gvy) * gap
            t.unc += 25.0 * gap

        # 2. discovery on this observation
        props = self._detect(image, region)
        # 3. classify sufficiently large proposals (cross-level evidence)
        cls_idx = [i for i, p in enumerate(props)
                   if min(p["local_box"][2]-p["local_box"][0], p["local_box"][3]-p["local_box"][1]) >= self.cfg.classify_min_px]
        results = {}
        if cls_idx:
            res, feats, _ = self.expert.classify(image, [props[i]["local_box"] for i in cls_idx])
            for j, i in enumerate(cls_idx):
                results[i] = (res[j], feats[j])

        # 4. associate proposals to predicted tracks (global geometry + gating)
        level_w = {0: 0.3, 1: 1.0, 2: 1.5}[level]
        used = set(); observed = []
        residuals = []
        for i, p in enumerate(props):
            sb = p["source_box"]; pcx = (sb[0]+sb[2])/2; pcy = (sb[1]+sb[3])/2
            best = None; best_score = 0.0
            for t in st.tracks:
                if t.id in used:
                    continue
                ov = iou(sb, t.box())
                dist = ((pcx-t.cx)**2 + (pcy-t.cy)**2) ** 0.5
                if ov > 0.2 or dist < max(t.unc, (t.w+t.h)/2):
                    s = ov + max(0.0, 1.0 - dist / max(1.0, t.unc))
                    if s > best_score:
                        best_score = s; best = t
            res = results.get(i)
            target_p = res[0]["target_probability"] if res else 0.5
            if best is not None:
                # update: residual vs predicted gives velocity contribution + global drift signal
                residuals.append((pcx - best.cx, pcy - best.cy))
                best.cx, best.cy = pcx, pcy
                best.w = sb[2]-sb[0]; best.h = sb[3]-sb[1]
                best.unc = max(40.0, min(best.w, best.h))
                best.misses = 0; best.last_idx = r.frame_index
                if res:
                    w = level_w * float(target_p)
                    best.ev += w * np.array(res[0]["class_scores"], np.float32)
                    best.target = max(best.target, float(target_p)) if level >= best.best_level else best.target
                    if level >= best.best_level:
                        best.best_level = level; best.feat = res[1]
                used.add(best.id); observed.append(best.id)
            else:
                if target_p >= 0.5 and len(st.tracks) < self.cfg.max_tracks:
                    t = Track(st.next_id, pcx, pcy, sb[2]-sb[0], sb[3]-sb[1], r.frame_index, level)
                    st.next_id += 1
                    if res:
                        t.ev += level_w * float(target_p) * np.array(res[0]["class_scores"], np.float32)
                        t.target = float(target_p); t.feat = res[1]
                    st.tracks.append(t); used.add(t.id); observed.append(t.id)

        # 5. update global drift estimate from residuals (median), causal EMA
        if residuals:
            rdx = float(np.median([d[0] for d in residuals])) / max(1, gap)
            rdy = float(np.median([d[1] for d in residuals])) / max(1, gap)
            if not st.g_init:
                st.gvx, st.gvy, st.g_init = rdx, rdy, True
            else:
                st.gvx = 0.7*st.gvx + 0.3*rdx; st.gvy = 0.7*st.gvy + 0.3*rdy
            # per-track residual velocity (relative to global) small correction
            for t in st.tracks:
                if t.id in observed:
                    t.vx *= 0.5; t.vy *= 0.5  # damp; global handles bulk motion

        # 6. misses + expiry
        alive = []
        for t in st.tracks:
            if t.id not in observed:
                t.misses += gap
            cx_ok = -200 < t.cx < W+200 and -200 < t.cy < H+200
            if t.misses <= self.cfg.ttl and cx_ok and (t.ev.sum() > 0 or t.misses == 0):
                alive.append(t)
        st.tracks = alive

        # 7. emit global-retained predictions
        annotations = []
        emitted = []
        ranked = sorted(st.tracks, key=lambda t: float(t.ev.max()) * (0.85 ** t.misses), reverse=True)
        for t in ranked:
            if t.ev.sum() <= 0:
                continue
            cls = int(t.ev.argmax())
            post = t.ev / max(1e-6, t.ev.sum())
            conf = float(np.clip(t.target * post[cls] * (0.85 ** t.misses), 0, 1))
            if conf < self.cfg.min_emit_conf:
                continue
            box = np.clip(np.array([t.cx-t.w/2, t.cy-t.h/2, t.cx+t.w/2, t.cy+t.h/2]) / [W, H, W, H], 0, 1)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            if any(cls == c and iou(box, b) > 0.5 for c, b in emitted):
                continue
            emitted.append((cls, box))
            annotations.append(DroneFlybyPredictionDto(object_id=OBJECT_CLASSES[cls], bbox=box.tolist(), confidence=conf))
            if len(annotations) >= 500:
                break

        # 8. camera policy (target-blind coverage + targeted L2 refine)
        requested = self._plan_camera(st, r, observed) if self.cfg.active_camera else None
        st.last_idx = r.frame_index
        resp = DroneFlybyPredictResponseDto(request_id=r.request_id, frame=r.frame,
                                            annotations=annotations, requested_view=requested)
        self.last_diagnostics = {"frame_index": r.frame_index, "level": level, "n_props": len(props),
                                 "n_tracks": len(st.tracks), "n_emitted": len(annotations),
                                 "global_drift": [round(st.gvx, 1), round(st.gvy, 1)],
                                 "requested": None if requested is None else [requested.resolution_level, requested.center_x, requested.center_y],
                                 "total_ms": (time.perf_counter()-start)*1000}
        return resp

    def _legal_step(self, st, goal_level, goal_cx, goal_cy):
        from dtos import RequestedViewDto
        cur = st.cam_level
        allowed = ALLOWED_RESOLUTION_LEVELS[cur]
        # choose next level progressing toward goal_level through legal transitions
        if goal_level in allowed:
            nl = goal_level
        else:
            # step through L1 (hub)
            nl = 1 if 1 in allowed else cur
        if nl == 0:
            st.cam_level, st.cam_cx, st.cam_cy = 0, FULL_FRAME_CENTER[0], FULL_FRAME_CENTER[1]
            return RequestedViewDto(resolution_level=0, center_x=FULL_FRAME_CENTER[0], center_y=FULL_FRAME_CENTER[1])
        limit = MAXIMUM_CENTER_DELTA_PIXELS[cur] - 3.0  # safe margin vs strict evaluator limit
        mnx, mxx, mny, myy = center_bounds(nl)
        # clamp goal into legal bounds FIRST, then cap the step to the limit
        tcx = min(max(goal_cx, mnx), mxx); tcy = min(max(goal_cy, mny), myy)
        dx, dy = tcx - st.cam_cx, tcy - st.cam_cy
        d = (dx*dx + dy*dy) ** 0.5
        if d > limit:
            f = limit / d
            tcx = st.cam_cx + dx * f; tcy = st.cam_cy + dy * f
        icx = int(round(min(max(tcx, mnx), mxx))); icy = int(round(min(max(tcy, mny), myy)))
        # guarantee within limit after rounding
        while ((icx - st.cam_cx) ** 2 + (icy - st.cam_cy) ** 2) ** 0.5 > MAXIMUM_CENTER_DELTA_PIXELS[cur] - 1:
            icx = int(round(st.cam_cx + (icx - st.cam_cx) * 0.98))
            icy = int(round(st.cam_cy + (icy - st.cam_cy) * 0.98))
        st.cam_level, st.cam_cx, st.cam_cy = nl, icx, icy
        return RequestedViewDto(resolution_level=nl, center_x=icx, center_y=icy)

    def _plan_camera(self, st, r, observed):
        # sync camera state to what the evaluator actually applied
        st.cam_level = r.view.resolution_level
        st.cam_cx = int(r.view.center_x); st.cam_cy = int(r.view.center_y)
        fi = r.frame_index
        # targeted L2 refine of the most uncertain/aging track with class mass, every 3rd frame
        if fi % 3 == 2:
            cands = [t for t in st.tracks if t.ev.sum() > 0]
            if cands:
                refine = max(cands, key=lambda t: t.misses + 1.0 / (1.0 + float(t.ev.max())))
                return self._legal_step(st, 2, int(refine.cx), int(refine.cy))
        # 3. systematic coverage: alternate L1 broad tiles and L2 fine sub-tiles
        wp = COVERAGE_WAYPOINTS[st.tour % len(COVERAGE_WAYPOINTS)]
        lvl, gx, gy = wp
        if st.cam_level == lvl and abs(st.cam_cx - gx) < 220 and abs(st.cam_cy - gy) < 220:
            st.region_seen[st.tour % len(COVERAGE_WAYPOINTS)] = fi
            st.tour += 1
            lvl, gx, gy = COVERAGE_WAYPOINTS[st.tour % len(COVERAGE_WAYPOINTS)]
        return self._legal_step(st, lvl, gx, gy)
