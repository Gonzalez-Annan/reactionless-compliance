"""Headless run of the set-and-go teleop with the geofence: push the dot outward for 8 s, send, wait 25 s.
The dot must never pass the fence, and the base must stay inside the tilt limit. Run from the repo root."""
import sys; sys.path.insert(0, "src")
import types, numpy as np, teleop as T

n, seen = [0], []
s = {"v": types.SimpleNamespace(cam=types.SimpleNamespace(azimuth=90.0, elevation=-45.0), close=lambda: None)}


def hook(m, d, pts):
    n[0] += 1
    if len(s.get("rays", ())) == len(T.DIRS):
        seen.append(T.depth([r[-1] for r in s["rays"]], home[0], pts[0]))
    return n[0] < 3300


def axes():                                             # sx, sy, dz, cx, cy, go, snap, fine
    return (1.0 if n[0] < 800 else 0.0), 0, 0, 0, 0, 800 <= n[0] < 810, False, False


home = []
setup = T.setup
T.setup = lambda size: (lambda r: (home.append(r[1].site_xpos[r[2]].copy()), r)[1])(setup(size))
T.viewer_hook = lambda: (s, hook)
T.free_drive(axes, "medium", 5, "3d")
k = np.array(seen)
print("dot depth: max %.4f, at the end %.4f, zone %d; rehearsal said %s" % (k.max(), k[-1], T.zone(k[-1]), s["ans"][0]))
assert k.max() <= T.FENCE[1] + 1e-6 and k.max() > T.FENCE[0], "the dot must reach the caution zone and stop at the fence"
print("ok")


def test_keys_off_windows(monkeypatch):
    """Linux / macOS path: a key the viewer reported just now is down, an old one is not."""
    import time
    import teleop as T
    monkeypatch.setattr(T, "WIN", False)
    monkeypatch.setattr(T, "PRESSED", {T.GLFW["up"]: time.time(), T.GLFW["left"]: time.time() - 1})
    assert T.down("up") and not T.down("left") and not T.down("enter")
    assert T.keyboard()[:2] == (0, 1)
