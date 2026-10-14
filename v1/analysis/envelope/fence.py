"""Runtime tilt fence: go without a rehearsal; when the base passes 2 deg, the hand stalls, or the move takes
longer than T_MOVE, the hand goes home. Run on goals of v4.npz that a rehearsal says are unreachable.
T_MOVE is set from the slowest arrival among reachable goals. Simulation only. Run from the repo root."""
import sys; sys.path.insert(0, "src")
import pathlib, copy, numpy as np, mujoco, teleop
z = np.load(pathlib.Path(__file__).parent / "v4.npz"); g, truth = z["g"], z["truth"]
m, d0, sid, Rb0 = teleop.setup("medium"); home = d0.site_xpos[sid].copy()
dt = teleop.DECIM * m.opt.timestep


def go(goal, trip=teleop.BASE_MAX, t_move=np.inf, t_end=90.0):
    d = copy.copy(d0); target, peak, t, last, lp, why = goal, 0.0, 0.0, 0.0, home, ""
    while t < t_end:
        p, base = teleop.ctrl(m, d, sid, Rb0, target, 5, "3d"); peak = max(peak, base)
        if not why:
            if np.linalg.norm(p - goal) < teleop.TOL:
                return "arrived", peak, base, t
            if base > trip:
                why = "tilt"
            elif t > t_move:
                why = "slow"
            elif t - last >= 0.5:
                if np.linalg.norm(p - lp) < teleop.STALL:
                    why = "stall"
                last, lp = t, p
            if why:
                target = home
        elif np.linalg.norm(p - home) < teleop.TOL:
            return why, peak, base, t
        for _ in range(teleop.DECIM):
            mujoco.mj_step(m, d)
        t += dt
    return ("never home after " + why) if why else "never resolved", peak, base, t


if __name__ == "__main__":
    ok = [go(x) for x in g[truth][::4]]
    ta = np.array([x[3] for x in ok if x[0] == "arrived"])
    print("reachable goals: %d of %d arrive without a time limit; time median %.1f, 95th pct %.1f, max %.1f s" % (len(ta), len(ok), np.median(ta), np.percentile(ta, 95), ta.max()), flush=True)
    T = 1.5 * ta.max()                                  # ponytail: one number from 42 goals on one bus; make it distance-based if moves vary more
    print("T_MOVE = %.1f s" % T, flush=True)
    assert all(go(x, t_move=T)[0] == "arrived" for x in g[truth][::20])    # the limit must not stop a move that works
    r = [go(x, t_move=T) for x in g[~truth][::3]]       # every third unreachable goal: 45 of 134
    for kind in sorted({x[0] for x in r}):
        a = np.array([x[1:] for x in r if x[0] == kind])
        print("%-24s %2d | peak tilt max %.3f | tilt at the end median %.3f max %.3f deg | time median %.1f max %.1f s" % (kind, len(a), a[:, 0].max(), np.median(a[:, 1]), a[:, 1].max(), np.median(a[:, 2]), a[:, 2].max()), flush=True)
