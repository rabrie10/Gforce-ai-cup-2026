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
from v5.gpu.discovery_v54 import RejectingExpert, MergedDiscovery
from v5.gpu.diagnostic_capture import DiagnosticCapture
from v5.gpu.diagnostic_capture_adapter import record_v6_diagnostics
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


def command_is_legal(from_level, fcx, fcy, to_level, tcx, tcy):
    """Independent final legality guard, mirroring local_evaluator.Camera.apply exactly:
    allowed transition, destination bounds, and movement <= the CURRENT level's limit
    (L0 full-frame reset is exempt from the delta limit)."""
    if to_level not in ALLOWED_RESOLUTION_LEVELS[from_level]:
        return False
    mnx, mxx, mny, myy = center_bounds(to_level)
    if not (mnx <= tcx <= mxx and mny <= tcy <= myy):
        return False
    if to_level == 0:
        return (tcx, tcy) == tuple(FULL_FRAME_CENTER)  # full-view reset, delta-exempt
    if ((tcx - fcx) ** 2 + (tcy - fcy) ** 2) ** 0.5 > MAXIMUM_CENTER_DELTA_PIXELS[from_level]:
        return False
    return True


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
        # believed = the evaluator's AUTHORITATIVE camera derived from our own issued-command
        # chain (reconciled via camera_command_feedback), because the hosted evaluator applies
        # commands asynchronously so r.view can be STALE relative to its true camera.
        self.believed = None            # (level, cx, cy) or None (bootstrap from r.view)
        self.last_issued = None         # (level, cx, cy) of our last emitted requested_view, or None (no-op)


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
        # V7: admission gates made configurable (audit 2eec8ec found three gates dropping
        # candidates that had usable proposals). Defaults reproduce V6 exactly.
        self.classify_min_px = int(os.getenv("V6_CLASSIFY_MIN_PX", "22"))
        self.target_min = float(os.getenv("V6_TARGET_MIN", "0.5"))


# Broad L1 coverage tiles over the whole 3840x2160 frame (fine L2 handled via track refine).
COVERAGE_WAYPOINTS = [(1, 960, 540), (1, 2880, 540), (1, 2880, 1620), (1, 960, 1620)]


class V6Pipeline:
    def __init__(self, config=None):
        self.cfg = config or Config()
        self.det = YOLO(self.cfg.detector)
        self.dev = 0 if self.cfg.device.startswith("cuda") else "cpu"
        self.expert = RejectingExpert(self.cfg.assets, device=self.cfg.device)
        # Ensemble: the Helsinki-fine-tuned ms1 detector has ~0 recall on hosted-domain objects,
        # so add the pretrained open-vocab OBB + YOLO-World detectors as a complementary proposal
        # source that generalizes to the hosted scene. Env-gated (V6_ENSEMBLE=1 default on).
        self.ensemble = None
        if os.getenv("V6_ENSEMBLE", "0") == "1":  # default OFF: ms1-only best on Helsinki (0.265 vs 0.206)
            self.ensemble = MergedDiscovery(self.cfg.assets, budget=int(os.getenv("V6_ENSEMBLE_BUDGET", "48")),
                                            device=self.cfg.device)
        self.states = OrderedDict()
        self.lock = threading.RLock()
        self.capture = DiagnosticCapture()
        self.last_diagnostics = {}
        self.manifest = {"pipeline": "v6", "detector": self.cfg.detector,
                 "ensemble": bool(self.ensemble),
                 "diagnostic_capture": self.capture.enabled}
        # warmup
        z = np.zeros((VH, VW, 3), np.uint8)
        self._detect(z, (0, 0, W, H))
        self.expert.classify(z, [[100, 100, 140, 140]])

    def empty(self, r):
        return DroneFlybyPredictResponseDto(request_id=r.request_id, frame=r.frame, annotations=[])

    def close(self):
        self.capture.close()

    def _detect(self, view, region):
        r = self.det.predict(view, conf=self.cfg.det_conf, imgsz=self.cfg.det_imgsz,
                             verbose=False, device=self.dev)[0]
        out = []
        if r.boxes is not None and len(r.boxes):
            for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                lb = [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
                out.append({"local_box": lb, "source_box": list(view_to_source(lb, region)),
                            "score": float(c), "src": "ms1"})
        if self.ensemble is not None:
            _, sel, _ = self.ensemble.propose(view, list(region))
            for c in sel:
                out.append({"local_box": c["local_box"], "source_box": c["source_box"],
                            "score": float(c["discovery_score"]), "src": c.get("candidate_source", "ens")})
        return self._merge_props(out)

    @staticmethod
    def _merge_props(props, thr=0.6):
        """Greedy NMS union across detectors: keep highest score, drop near-duplicates."""
        props = sorted(props, key=lambda p: p["score"], reverse=True)
        kept = []
        for p in props:
            if all(iou(p["local_box"], q["local_box"]) <= thr for q in kept):
                kept.append(p)
        return kept

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
                if target_p >= self.cfg.target_min and len(st.tracks) < self.cfg.max_tracks:
                    t = Track(st.next_id, pcx, pcy, sb[2]-sb[0], sb[3]-sb[1], r.frame_index, level)
                    st.next_id += 1
                    if res:
                        t.ev += level_w * float(target_p) * np.array(res[0]["class_scores"], np.float32)
                        t.target = float(target_p); t.feat = res[1]
                    st.tracks.append(t); used.add(t.id); observed.append(t.id)

        # 5. update global drift estimate. residual = observed - predicted = (true - gvx)*gap,
        #    so the correction is ADDITIVE: gvx += alpha * residual/gap  (converges to true drift).
        if residuals:
            rdx = float(np.median([d[0] for d in residuals])) / max(1, gap)
            rdy = float(np.median([d[1] for d in residuals])) / max(1, gap)
            alpha = 0.7 if not st.g_init else 0.5
            st.gvx += alpha * rdx; st.gvy += alpha * rdy
            st.g_init = True
            for t in st.tracks:
                if t.id in observed:
                    t.vx *= 0.5; t.vy *= 0.5  # damp per-track; global handles bulk motion

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
        # props is the post-merge detector candidate list (after cross-detector
        # greedy NMS), before classification, tracking, and emission filtering.
        record_v6_diagnostics(self.capture, r, image, props, resp)
        return resp

    def _legal_step(self, st, goal_level, goal_cx, goal_cy):
        from dtos import RequestedViewDto
        cur = st.cam_level
        allowed = ALLOWED_RESOLUTION_LEVELS[cur]
        nl = goal_level if goal_level in allowed else (1 if 1 in allowed else cur)
        if nl == 0:
            st.cam_level, st.cam_cx, st.cam_cy = 0, FULL_FRAME_CENTER[0], FULL_FRAME_CENTER[1]
            return RequestedViewDto(resolution_level=0, center_x=FULL_FRAME_CENTER[0], center_y=FULL_FRAME_CENTER[1])
        cx, cy = st.cam_cx, st.cam_cy
        limit = MAXIMUM_CENTER_DELTA_PIXELS[cur]

        def feasible(nlvl):
            mnx, mxx, mny, myy = center_bounds(nlvl)
            gx = min(max(goal_cx, mnx), mxx); gy = min(max(goal_cy, mny), myy)
            # farthest point on segment cam->goal that is IN bounds AND within (limit-2)
            best = None
            for i in range(20, -1, -1):
                t = i / 20.0
                px = int(round(cx + t * (gx - cx))); py = int(round(cy + t * (gy - cy)))
                if mnx <= px <= mxx and mny <= py <= myy and ((px-cx)**2 + (py-cy)**2) ** 0.5 <= limit - 2:
                    best = (px, py); break
            return best

        pt = feasible(nl)
        if pt is None:  # cannot legally reach nl this frame; keep current level, best legal move toward goal
            nl = cur; pt = feasible(cur)
        if pt is None:  # no move possible: stay put (legal no-op)
            pt = (cx, cy)
        st.cam_level, st.cam_cx, st.cam_cy = nl, pt[0], pt[1]
        return RequestedViewDto(resolution_level=nl, center_x=pt[0], center_y=pt[1])

    def _reconcile_believed(self, st, r):
        """Update believed authoritative camera from our own issued-command chain.
        The hosted evaluator applies commands asynchronously, so r.view can be stale;
        the evaluator validates our command against the state AFTER applying our previous
        (accepted) command. So believed = last_issued unless feedback says it was refused."""
        if st.believed is None:
            st.believed = (int(r.view.resolution_level), int(r.view.center_x), int(r.view.center_y))
            return
        if st.last_issued is None:
            return  # we issued no move last time -> evaluator camera unchanged
        rejected = False
        fb = getattr(r, "camera_command_feedback", None)
        rv = getattr(fb, "requested_view", None) if fb is not None else None
        if rv is not None and (int(rv.resolution_level), int(rv.center_x), int(rv.center_y)) == st.last_issued:
            rejected = True
        if not rejected:
            st.believed = st.last_issued  # applied in order by the evaluator

    def _plan_camera(self, st, r, observed):
        # Plan against the BELIEVED authoritative camera (our issued chain), NOT the possibly
        # stale r.view. Detection/tracking already uses r.view's region separately.
        self._reconcile_believed(st, r)
        st.cam_level, st.cam_cx, st.cam_cy = st.believed
        fi = r.frame_index
        cmd = None
        # targeted L2 refine of the most uncertain/aging track with class mass, every 3rd frame
        if fi % 3 == 2:
            cands = [t for t in st.tracks if t.ev.sum() > 0]
            if cands:
                refine = max(cands, key=lambda t: t.misses + 1.0 / (1.0 + float(t.ev.max())))
                cmd = self._legal_step(st, 2, int(refine.cx), int(refine.cy))
        if cmd is None:
            wp = COVERAGE_WAYPOINTS[st.tour % len(COVERAGE_WAYPOINTS)]
            lvl, gx, gy = wp
            if st.cam_level == lvl and abs(st.cam_cx - gx) < 220 and abs(st.cam_cy - gy) < 220:
                st.region_seen[st.tour % len(COVERAGE_WAYPOINTS)] = fi
                st.tour += 1
                lvl, gx, gy = COVERAGE_WAYPOINTS[st.tour % len(COVERAGE_WAYPOINTS)]
            cmd = self._legal_step(st, lvl, gx, gy)
        # Guard 1: legality against the believed authoritative state.
        bl, bx, by = st.believed
        if cmd is not None and not command_is_legal(bl, bx, by, cmd.resolution_level, cmd.center_x, cmd.center_y):
            cmd = None  # illegal -> no move (evaluator keeps its camera; always legal)

        # Guard 2 (V7): legality against the RECEIVED view. If a response never reaches the
        # evaluator, it does not apply that command, so `believed` silently diverges and every
        # later step is planned from a camera the evaluator is not at. Hosted attempt a23ca0fb
        # frames 196/202: 1041px and 1492px moves requested while the evaluator sat at
        # L2 (2001,1609) -> ignored, and the camera stayed stuck there.
        # The evaluator validates against the camera it reports in the request, so re-plan one
        # legal step from THAT camera toward the same goal, and drop the command if even that
        # is not legal. V6_RECV_CLAMP=0 restores the old behaviour.
        if cmd is not None and os.getenv("V6_RECV_CLAMP", "1") == "1":
            rl, rx, ry = int(r.view.resolution_level), int(r.view.center_x), int(r.view.center_y)
            if not command_is_legal(rl, rx, ry, cmd.resolution_level, cmd.center_x, cmd.center_y):
                goal = (cmd.resolution_level, cmd.center_x, cmd.center_y)
                st.believed = (rl, rx, ry)
                st.cam_level, st.cam_cx, st.cam_cy = st.believed
                cmd = self._legal_step(st, goal[0], goal[1], goal[2])
                if cmd is not None and not command_is_legal(rl, rx, ry, cmd.resolution_level,
                                                            cmd.center_x, cmd.center_y):
                    cmd = None
        st.last_issued = None if cmd is None else (cmd.resolution_level, cmd.center_x, cmd.center_y)
        return cmd
