"""Headless run of the maps study's drive loop with a scripted operator: two goals per block, the first from rest and
the second from where the first left the arm.
gate: a near goal is sent and reached, a far one is refused by the rehearsal and never moves the arm.
guaranteed: the dot cannot leave the small cage, so a goal outside it is given up without a send.
live: after the first move the cage is probed again from where the hand stopped and swapped in.
Run from the repo root."""
import sys; sys.path.insert(0, "src")
import types, numpy as np, teleop as T


def drive(aid, G, steps=30000):
    n, at, s = [0], [None, 0, False], {"v": types.SimpleNamespace(cam=types.SimpleNamespace(azimuth=90.0, elevation=-45.0), close=lambda: None)}
    seen = []

    def hook(m, d, pts):
        n[0] += 1
        if not seen or s["rays"] is not seen[-1]:
            seen.append(s["rays"])                      # every cage the operator was shown
        return n[0] < steps

    def axes():                                         # sx, sy, dz, cx, cy, go, snap, fine
        if "tgt" not in s or "link" not in s:
            return 0, 0, 0, 0, 0, False, False, False
        if s["tgt"] is not at[0]:
            at[:] = [s["tgt"], n[0], False]
        e = s["tgt"] - s["link"][1]
        az, el = np.radians(90.0), np.radians(-45.0)    # the inverse of T.move at the stub camera
        fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        right = np.array([np.sin(az), -np.cos(az), 0])
        c = np.clip(np.array([e @ right, e @ np.cross(right, fwd), e @ fwd]) / 0.0015, -1, 1)
        go = np.linalg.norm(e) < 1e-3 and not at[2]
        at[2] = at[2] or go
        return c[0], c[1], c[2], 0, 0, go, False, False

    # give a goal up 30 s in; live: as soon as the new cage is in
    axes.park = lambda: len(seen) > 1 if aid == "live" else n[0] - at[1] > 3000
    T.viewer_hook = lambda: (s, hook)
    rows = T.free_drive(axes, "medium", 5, "3d", targets=G - home, aid=aid)
    for r in rows:
        print(aid, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
    return rows, s, len(seen) > 1


def test_maps():
    global home
    _, goals, dep, _, truth = T.pool("medium")
    G, D, TR = goals.reshape(-1, 3), dep.ravel(), truth.ravel()
    m, d, sid, _ = T.setup("medium")
    home = d.site_xpos[sid].copy()
    near, far = np.flatnonzero((D < 0.6) & TR)[0], np.flatnonzero(D > 1.3)[0]

    (a, b), s, _ = drive("gate", G[[near, far]])
    assert a["reached"] and a["true_go"] and a["sends"] == 1 and a["refused"] == 0 and a["trips"] == 0
    assert b["parked"] and not b["true_go"] and b["refused"] == 1 and b["trips"] == 0 and b["path_m"] < 0.01
    assert "cage" not in s

    rays = T.pool("medium")[0]
    small = [r[-1] for r in T.guaranteed(rays, home, "medium")]
    kg = np.array([T.depth(small, home, g) for g in G])
    assert max(T.depth([r[-1] for r in rays], home, t) for t in small) <= 1 + 1e-9, "the small cage lies inside the rest cage"
    out = np.flatnonzero((kg > 1.2) & TR)[0]            # reachable from rest, outside the small cage
    (a, b), s, _ = drive("guaranteed", G[[np.flatnonzero(kg < 0.8)[0], out]])
    assert a["reached"] and a["trips"] == 0
    assert b["parked"] and b["sends"] == 0, "the dot stops at the small cage, so nothing is sent"

    (a, b), s, swapped = drive("live", G[[near, far]], 300000)
    assert a["reached"] and swapped, "the cage is probed again once the hand has stopped away from rest"


if __name__ == "__main__":
    test_maps()
    print("ok")
