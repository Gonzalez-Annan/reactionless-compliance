"""Headless run of the feedback study's drive loop, cue "none": a scripted operator steers the dot onto each pink
ball and sends. One goal deep inside the envelope, one far outside it. The first must be reached; on the second the
fence must bring the arm back, and the operator then gives it up. Also checks the slice rings. Run from the repo root."""
import sys; sys.path.insert(0, "src")
import types, numpy as np, teleop as T

_, goals, dep, _, truth = T.pool("medium")
G, D, TR = goals.reshape(-1, 3), dep.ravel(), truth.ravel()
near, far = np.flatnonzero((D < 0.8) & TR)[0], np.flatnonzero(D > 1.3)[0]
assert not TR[far]

n, at, s = [0], [None, 0, False], {"v": types.SimpleNamespace(cam=types.SimpleNamespace(azimuth=90.0, elevation=-45.0), close=lambda: None)}


def hook(m, d, pts):
    n[0] += 1
    return n[0] < 30000


def axes():                                             # sx, sy, dz, cx, cy, go, snap, fine
    if "tgt" not in s or "link" not in s:
        return 0, 0, 0, 0, 0, False, False, False
    if s["tgt"] is not at[0]:
        at[:] = [s["tgt"], n[0], False]
    e = s["tgt"] - s["link"][1]
    az, el = np.radians(90.0), np.radians(-45.0)        # the inverse of T.move at the stub camera
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = np.array([np.sin(az), -np.cos(az), 0])
    c = np.clip(np.array([e @ right, e @ np.cross(right, fwd), e @ fwd]) / 0.0015, -1, 1)
    go = np.linalg.norm(e) < 1e-3 and not at[2]
    at[2] = at[2] or go
    return c[0], c[1], c[2], 0, 0, go, False, False


axes.park = lambda: n[0] - at[1] > 6000                 # 60 s into a goal: give it up
m, d, sid, _ = T.setup("medium")
home = d.site_xpos[sid].copy()
T.viewer_hook = lambda: (s, hook)
rows = T.free_drive(axes, "medium", 5, "3d", targets=G[[near, far, near]] - home, aid="none")
for r in rows:
    print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
a, b, c = rows
assert a["reached"] and not a["parked"] and a["trips"] == 0 and a["sends"] == 1
assert b["parked"] and not b["reached"] and b["trips"] == 1 and b["time_s"] > 60 + T.REPARK_S
assert c["reached"] and abs(c["time_s"] - a["time_s"]) < 0.5, "after a reset the same goal must take the same time"
assert "cage" not in s and "cuts" not in s, "cue none must draw no envelope"

rays = T.pool("medium")[0]
seg = T.cut([r[-1] for r in rays], home, np.array([0, 0, 1.0]))
assert len(seg) > 10 and np.abs(seg[..., 2] - home[2]).max() < 1e-9, "a slice lies in its plane"
ends = np.round(seg.reshape(-1, 3), 6)
assert (np.unique(ends, axis=0, return_counts=True)[1] == 2).all(), "a slice of a closed cage is a closed ring"
print("ok")
