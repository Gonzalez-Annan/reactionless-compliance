"""Pick the ready pose on a measured basis instead of by hand. Run from the repo root (about 5 min, 10 processes).

The old rest pose had the hand 1 cm above the bus, so the bus cut 62 of 98 probe directions (D29).
Stage 1: N random arm poses that start with the hand 30 cm or more above the bus (link 2 is always ~6 cm from it), of the arm itself and of the joint limits; 14 probe
         directions each (axes + cube corners); score = volume inside the 2 deg shell (mean r^3 * 4 pi / 3).
Stage 2: the best 6 with all 98 directions. Pick: largest 2 deg volume among poses whose shortest 2 deg reach is
         at least MIN_R, so no direction is a dead end for the operator.
Also the kinematic workspace (hand positions any collision-free pose gives, base held level) for scale.
-> analysis/envelope/ready.npz, and the line to paste into src/task.py."""
import sys; sys.path.insert(0, "src")
import numpy as np, mujoco
from multiprocessing import Pool
import teleop as T

N, MIN_R, SEED = 160, 0.15, 0
AX = [np.array(u, float) / np.linalg.norm(u) for u in
      [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)] + [(a, b, c) for a in (1, -1) for b in (1, -1) for c in (1, -1)]]


def selfclear(m, d):
    """Least distance between arm links that are not neighbours, m. ponytail: capsule geoms only, as modelled."""
    g = [i for i in range(m.ngeom) if m.geom_bodyid[i] >= 2]
    return min(mujoco.mj_geomDistance(m, d, a, b, 1.0, None) for a in g for b in g
               if m.geom_bodyid[b] - m.geom_bodyid[a] >= 3)      # links two apart share a short wrist segment and always touch


def probe(a):
    q, dirs = a
    T.Q_REST = q                                   # setup() and the controller's posture pull both read it
    m, d, sid, Rb0 = T.setup("medium")
    p0, out = d.site_xpos[sid].copy(), []
    for u in dirs:
        mk = []
        best = T.ray(m, d, sid, Rb0, u, 5, "3d", marks=mk)
        out.append(np.array((mk + [best] * 3)[:3]))
    return np.linalg.norm(np.array(out) - p0, axis=2)          # (dirs, 3 shells) m


def vol(r):
    return 4 * np.pi / 3 * (r ** 3).mean(0)        # star-shaped set sampled along rays, per shell


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    m, d, sid, _ = T.setup("medium")
    lo, hi = T.Q_LIM[:, 0], T.Q_LIM[:, 1]
    cand, hand = [np.array(T.Q_REST)], []
    for i in range(200000):
        q = rng.uniform(lo, hi)
        d.qpos[7:] = q
        mujoco.mj_forward(m, d)
        if T.clear(m, d) < T.CLEAR or selfclear(m, d) < 0.0:
            continue
        hand.append(d.site_xpos[sid].copy())       # kinematic workspace: any collision-free pose
        if len(cand) < N and T.clear(m, d) > 0.05 and d.site_xpos[sid][2] > 0.8 and selfclear(m, d) > 0.05 and (np.minimum(q - lo, hi - q) > 0.4).all():
            cand.append(q)
        if len(hand) >= 60000:
            break
    hand = np.array(hand)
    vk = len({tuple(v) for v in np.floor(hand / 0.05).astype(int)}) * 0.05 ** 3
    print(f"kinematic workspace: {vk:.2f} m^3 (5 cm voxels, {len(hand)} collision-free poses), "
          f"hand up to {np.linalg.norm(hand - [0, 0, 0.5], axis=1).max():.2f} m from the arm mount", flush=True)
    with Pool(10) as pl:
        r1 = np.array(pl.map(probe, [(q, AX) for q in cand]))              # (N, 14, 3)
        v1, mn1 = np.array([vol(r)[2] for r in r1]), r1[:, :, 2].min(1)
        ok = mn1 >= MIN_R
        print(f"stage 1: {len(cand)} poses; old rest pose 2 deg volume {v1[0]:.3f} m^3, shortest reach {mn1[0] * 100:.0f} cm; "
              f"{ok.sum()} poses have every direction past {MIN_R * 100:.0f} cm", flush=True)
        top = sorted(np.flatnonzero(ok if ok.sum() >= 6 else mn1 >= np.sort(mn1)[-6]), key=lambda i: -v1[i])[:6]
        r2 = np.array(pl.map(probe, [(cand[i], T.DIRS) for i in top]))     # (6, 98, 3)
    for i, r in zip(top, r2):
        v = vol(r)
        print(f"pose {i}: 2 deg volume {v[2]:.3f} m^3 ({100 * v[2] / vk:.0f}% of kinematic), 0.5 deg {v[0]:.3f}; "
              f"2 deg reach {r[:, 2].min() * 100:.0f}-{r[:, 2].max() * 100:.0f} cm, median {np.median(r[:, 2]) * 100:.0f}; "
              f"0.5 deg median {np.median(r[:, 0]) * 100:.0f}; q = {np.round(cand[i], 2).tolist()}", flush=True)
    full_ok = r2[:, :, 2].min(1) >= MIN_R
    j = max(range(len(top)), key=lambda k: (full_ok[k], vol(r2[k])[2]))
    d.qpos[7:] = cand[top[j]]
    mujoco.mj_forward(m, d)
    print(f"pick: pose {top[j]} (all 98 directions past {MIN_R * 100:.0f} cm: {bool(full_ok[j])}); hand at {np.round(d.site_xpos[sid], 2)}, "
          f"arm {T.clear(m, d) * 100:.0f} cm from the bus")
    print(f"Q_REST = np.array({np.round(cand[top[j]], 2).tolist()})")
    np.savez("analysis/envelope/ready.npz", cand=np.array(cand), r1=r1, top=np.array(top), r2=r2, vk=vk, pick=cand[top[j]])
