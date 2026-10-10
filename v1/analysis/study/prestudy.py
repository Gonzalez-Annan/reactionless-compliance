"""Scripted pre-study for the maps study. NOT participant data: a fixed rule plays the operator, so this only says
whether the four maps can differ at all on these goals before people are used. Run from the repo root:
python analysis/study/prestudy.py            ids 1 2 3 4, one per block order
python analysis/study/prestudy.py 1          one id
-> data/scripted/maps_<id>.csv (never data/participants) and a table per map on the scored goals (leg 2).

The rule: steer the dot to the ball. Go once when the dot is on it (live map: only when the map is current).
Give the goal up when the dot has been stuck short of the ball for 2 s (the cage holds it), when the go was
refused (red dot), or when the ball is still there 40 s after a go (the fence brought the arm back).
A person would plan, hesitate and learn. The rule does none of that, so the times are a floor, not a forecast."""
import sys; sys.path.insert(0, "src")
import builtins, pathlib, time, types
import numpy as np, pandas as pd
import teleop as T

STUCK, AFTER_GO = 200, 4000          # steps of 0.01 s


def operator():
    n, at, last, sync = [0], [None, 0, None], [0.0, 0], [None]      # at: goal, step it appeared, step of the go
    s = {"v": types.SimpleNamespace(cam=types.SimpleNamespace(azimuth=90.0, elevation=-45.0), close=lambda: None)}

    def hook(m, d, pts):
        n[0] += 1
        if s.get("text") == "updating":      # the live map costs wall time, so the simulation waits for the clock here
            sync[0] = sync[0] or (time.time(), d.time)
            time.sleep(max(0, (d.time - sync[0][1]) - (time.time() - sync[0][0])))
        else:
            sync[0] = None
        return True

    def err():
        return s["tgt"] - s["link"][1]

    def axes():                                          # sx, sy, dz, cx, cy, go, snap, fine
        if "tgt" not in s or "link" not in s:
            return 0, 0, 0, 0, 0, False, False, False
        if s["tgt"] is not at[0]:
            at[:] = [s["tgt"], n[0], None]
        e = err()
        if abs(np.linalg.norm(e) - last[0]) > 1e-6:
            last[:] = [np.linalg.norm(e), n[0]]
        az, el = np.radians(90.0), np.radians(-45.0)     # the inverse of T.move at the stub camera
        fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        right = np.array([np.sin(az), -np.cos(az), 0])
        c = np.clip(np.array([e @ right, e @ np.cross(right, fwd), e @ fwd]) / 0.0015, -1, 1)
        go = np.linalg.norm(e) < 1e-3 and at[2] is None and s.get("text") != "updating"
        if go:
            at[2] = n[0]
        return c[0], c[1], c[2], 0, 0, go, False, False

    def park():
        if "tgt" not in s or s["tgt"] is not at[0] or s.get("text") == "updating":
            return False
        if at[2] is None:
            return last[0] > 1e-3 and n[0] - last[1] > STUCK
        return list(s.get("rgba", [])) == T.RED or n[0] - at[2] > AFTER_GO

    axes.park = park
    def fresh():                                         # a new block: nothing left over from the last one but the camera
        v = s["v"]; s.clear(); s["v"] = v
        return s, hook

    return axes, fresh


if __name__ == "__main__":
    T.OUT = pathlib.Path("data/scripted")
    builtins.input = lambda q="": ""
    T.ask = lambda q, n, lo, hi: [float("nan")] * n      # a rule has no workload and no trust
    for pid in [int(a) for a in sys.argv[1:]] or [1, 2, 3, 4]:
        axes, T.viewer_hook = operator()
        T.feedback(axes, pid, "medium", T.MAPS, "maps")
    df = pd.concat(pd.read_csv(f) for f in sorted(T.OUT.glob("maps_*.csv")))
    df = df[df.leg == 2]
    g = df.groupby("cue")
    out = pd.DataFrame({"goals": g.size(), "s_per_goal": g.time_s.mean().round(1), "reached": g.reached.sum(),
                        "wrong_goes": g.trips.sum(), "refused": g.refused.sum(),
                        "gave_up_reachable": g.apply(lambda x: (x.parked & x.true_go).sum()),
                        "gave_up_unreachable": g.apply(lambda x: (x.parked & ~x.true_go).sum()),
                        "worst_tilt_deg": g.peak_base_deg.max().round(2)}).reindex(T.MAPS)
    print("SCRIPTED OPERATOR, not participants. Scored goals (leg 2), ids", sorted(df.pid.unique()))
    print(out.to_string())
