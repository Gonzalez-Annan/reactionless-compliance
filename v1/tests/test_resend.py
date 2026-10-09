"""Feedback study: a second send on the same goal must start from rest (D36). A scripted operator sends the hand
half way, waits for it, then sends to the goal. Right after the second send the hand has to be back at home.
Run from the repo root."""
import sys; sys.path.insert(0, "src")
import types, numpy as np, teleop as T

_, goals, dep, _, truth = T.pool("medium")
G, D, TR = goals.reshape(-1, 3), dep.ravel(), truth.ravel()
m, d, sid, _ = T.setup("medium")
home = d.site_xpos[sid].copy()
g = G[np.flatnonzero((D < 0.8) & TR & (np.linalg.norm(G - home, axis=1) > 0.1))[0]]
mid = (home + g) / 2
n, st, seen = [0], {"phase": 0, "n2": None}, {}
s = {"v": types.SimpleNamespace(cam=types.SimpleNamespace(azimuth=90.0, elevation=-45.0), close=lambda: None)}


def hook(m, d, pts):
    n[0] += 1
    if st["n2"] is not None and n[0] == st["n2"] + 2:
        seen["after"] = np.linalg.norm(d.site_xpos[sid] - home)
    return n[0] < 20000


def axes():                                             # sx, sy, dz, cx, cy, go, snap, fine
    if "link" not in s:
        return 0, 0, 0, 0, 0, False, False, False
    hand, dot = s["link"]
    if st["phase"] == 1:                                # wait for the hand to get half way
        if np.linalg.norm(hand - mid) < 2e-3:
            st["phase"], seen["before"] = 2, np.linalg.norm(hand - home)
        return 0, 0, 0, 0, 0, False, False, False
    if st["phase"] == 3:
        return 0, 0, 0, 0, 0, False, False, False
    e = (mid if st["phase"] == 0 else g) - dot
    az, el = np.radians(90.0), np.radians(-45.0)        # the inverse of T.move at the stub camera
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = np.array([np.sin(az), -np.cos(az), 0])
    c = np.clip(np.array([e @ right, e @ np.cross(right, fwd), e @ fwd]) / 0.0015, -1, 1)
    go = np.linalg.norm(e) < 1e-3
    if go:
        st["phase"] += 1
        if st["phase"] == 3:
            st["n2"] = n[0]
    return c[0], c[1], c[2], 0, 0, go, False, False


T.viewer_hook = lambda: (s, hook)
rows = T.free_drive(axes, "medium", 5, "3d", targets=np.array([g - home]), aid="none")
print("hand from home: %.3f m before the second send, %.4f m just after; row %s" % (seen["before"], seen["after"], rows))
assert seen["before"] > 0.04 and seen["after"] < 0.01, "the second send must start from rest"
assert rows[0]["reached"] and rows[0]["sends"] == 2 and rows[0]["trips"] == 0
print("ok")
