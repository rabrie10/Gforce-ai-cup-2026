"""Model-free regression tests for V6 camera legality under a LAGGED (pipelined) evaluator.
Reproduces the reported hosted illegal commands and verifies the fix emits zero illegal
commands validated against the evaluator's authoritative (lagged) camera. No GPU/models."""
import types, random
from dtos import ALLOWED_RESOLUTION_LEVELS, MAXIMUM_CENTER_DELTA_PIXELS, FULL_FRAME_CENTER
from v5.gpu.pipeline_v6 import V6Pipeline, State, command_is_legal, center_bounds

FAILS = 0
def check(cond, msg):
    global FAILS
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond: FAILS += 1

# --- A. reproduce the 6 reported hosted pairs: illegal vs the evaluator's TRUE current ---
reported = [
    ("frame6",  1, 2016, 540, 2, 600, 1531),   # L1(2016,540)->L2(600,1531) 1728>1102
    ("frame14", 2, 2040, 1012, 1, 2880, 1620),  # L2(2040,1012)->L1(2880,1620) 1037>551
]
for name, fl, fx, fy, tl, tx, ty in reported:
    check(not command_is_legal(fl, fx, fy, tl, tx, ty), f"{name}: reported command is illegal vs true current")

# --- transitions legal cases ---
check(command_is_legal(0, 1920, 1080, 1, 960, 540), "L0->L1 small move legal")
check(command_is_legal(1, 960, 540, 2, 600, 1531), "L1->L2 within 1102 legal")
check(command_is_legal(2, 2040, 1012, 1, 2100, 1050), "L2->L1 within 551 legal")
check(command_is_legal(1, 2016, 540, 0, 1920, 1080), "L1->L0 reset legal")
check(not command_is_legal(2, 2040, 1012, 0, 1920, 1080), "L2->L0 illegal (not allowed transition)")

# --- lagged/pipelined evaluator simulation with the FIXED planner ---
def fake_request(fi, view, feedback):
    v = types.SimpleNamespace(resolution_level=view[0], center_x=view[1], center_y=view[2])
    return types.SimpleNamespace(frame_index=fi, view=v, camera_command_feedback=feedback)

def fb_for(cmd):
    rv = types.SimpleNamespace(resolution_level=cmd[0], center_x=cmd[1], center_y=cmd[2])
    return types.SimpleNamespace(requested_view=rv, reason="refused")

def run_sim(n, lag=1, skips=False, seed=0):
    rng = random.Random(seed)
    pipe = V6Pipeline.__new__(V6Pipeline)  # model-free: only camera methods used
    st = State()
    true_cam = (0, FULL_FRAME_CENTER[0], FULL_FRAME_CENTER[1])
    sent_view = true_cam
    inflight = []  # commands awaiting application (models lag)
    feedback = None
    illegal = 0; applied = 0; noop = 0
    fi = 0
    for k in range(n):
        r = fake_request(fi, sent_view, feedback)
        cmd = pipe._plan_camera(st, r, [])
        # evaluator applies the oldest inflight command (lagged application), in order
        inflight.append(None if cmd is None else (cmd.resolution_level, cmd.center_x, cmd.center_y))
        feedback = None
        if len(inflight) > lag:
            due = inflight.pop(0)
            if due is None:
                noop += 1
            elif command_is_legal(true_cam[0], true_cam[1], true_cam[2], due[0], due[1], due[2]):
                true_cam = due; applied += 1
            else:
                illegal += 1; feedback = fb_for(due)
        # r.view the evaluator sends next reflects its CURRENT true_cam (which lags the inflight cmds)
        sent_view = true_cam
        fi += (rng.choice([1, 1, 1, 2, 3]) if skips else 1)
    # drain
    for due in inflight:
        if due is None: continue
        if command_is_legal(true_cam[0], true_cam[1], true_cam[2], due[0], due[1], due[2]):
            true_cam = due
        else:
            illegal += 1
    return illegal, applied, noop

for lag in (1, 2):
    for skips in (False, True):
        ill, app, noop = run_sim(160, lag=lag, skips=skips, seed=lag*10 + int(skips))
        check(ill == 0, f"lagged replay (lag={lag}, skips={skips}, 160 frames): {ill} illegal (applied={app}, noop={noop})")

# --- guard catches corrupted internal state ---
pipe = V6Pipeline.__new__(V6Pipeline); st = State()
st.believed = (2, 2040, 1012); st.last_issued = None
r = fake_request(2, (1, 2880, 540), None)  # r.view stale/inconsistent
cmd = pipe._plan_camera(st, r, [])
ok = cmd is None or command_is_legal(st.believed[0], st.believed[1], st.believed[2], cmd.resolution_level, cmd.center_x, cmd.center_y)
check(ok, "guard: emitted command legal vs believed even with stale r.view")

print("\nRESULT:", "ALL PASS" if FAILS == 0 else f"{FAILS} FAILURES")
