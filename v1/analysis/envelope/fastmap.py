"""How far off is the fast map (livemap.fast, no physics) from the simulated one (T.ray)? Run from the repo root.
From rest and from the first 4 starts of wide.py: 98 lines each, both ways.
Error = fast reach minus simulated reach, m. Over = the fast map promises more than 0.05 m too much.
-> data/fastmap_medium.npz and a table."""
import sys; sys.path.insert(0, "src"); sys.path.insert(0, "analysis/envelope")
import time, numpy as np
from multiprocessing import Pool
import teleop as T, livemap
from wide import arrive

if __name__ == "__main__":
    S = np.load(T.OUT.parent / "wide98_medium.npz")["S"]
    pl = Pool(10, livemap.init)
    states = [T.setup("medium")] + [arrive(s)[:4] for s in S[:4]]
    out = {}
    for n, (m, d, sid, Rb0) in enumerate(states):
        p = d.site_xpos[sid].copy()
        true = np.linalg.norm(np.array(livemap.ask(pl, d).get()) - p, axis=1)
        for dt in (0.02, 0.05, 0.1, 0.2):
            t = time.time()
            f = np.linalg.norm([livemap.fast(m, d, sid, Rb0, u, dt) - p for u in T.DIRS], axis=1)
            e = f - true; out["%d_%g" % (n, dt)] = f
            print("start %d dt %.2f: %.2f s for 98 lines, error median %.3f, 95%% %.3f, worst %.3f m, over by 0.05: %d, "
                  "short by 0.05: %d" % (n, dt, time.time() - t, np.median(abs(e)), np.percentile(abs(e), 95),
                                         abs(e).max(), (e > 0.05).sum(), (e < -0.05).sum()), flush=True)
        out["true%d" % n] = true
    pl.terminate()
    np.savez(T.OUT.parent / "fastmap_medium.npz", **out)
