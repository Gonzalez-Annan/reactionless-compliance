"""The same question on the textbook system: a planar two-link arm on a free-floating base, no MuJoCo, exact momentum.
Run from the repo root. Three things:
 A. the known result (Papadopoulos and Dubowsky): with no tilt limit, a straight hand move can only fail in the band of
    radii where a dynamic singularity exists. Outside the band it works from any start.
 B. ours: with a 2 deg tilt budget counted from rest, does the map drawn from rest hold from other starts?
 C. a first-order guess of the tilt at the goal (tilt spent + tilt per metre at the start x distance): how often right?
ponytail: two joints, so the base cannot be held level at all, tilt builds from the first centimetre. The 7-joint arm
holds it level until it runs out of joints. Same question, harder case for the map.
-> a table, nothing saved."""
import sys, numpy as np

M, I = np.array([200.0, 10.0, 10.0]), np.array([50.0, 0.9, 0.9])     # base, link 1, link 2: kg, kg m2 about own centre
R0, L = 0.5, (1.0, 1.0)                                               # base centre to shoulder, link lengths, m
BUDGET, REST = np.radians(2.0), np.array([0.0, 0.9, -1.8])            # tilt limit; rest state (base angle, q1, q2)
A = np.tril(np.ones((3, 3)))                                          # body angle rates per state rate


def pos(x):
    """-> body centres (3, 2) and hand (2,), about the system centre of mass."""
    e = lambda a: np.array([np.cos(a), np.sin(a)])
    a0, a1, a2 = x[0], x[0] + x[1], x[0] + x[1] + x[2]
    j1 = R0 * e(a0); j2 = j1 + L[0] * e(a1)
    C = np.array([0 * j1, j1 + L[0] / 2 * e(a1), j2 + L[1] / 2 * e(a2)])
    cm = M @ C / M.sum()
    return C - cm, j2 + L[1] * e(a2) - cm


def kin(x, h=1e-6):
    """-> w (base rate per joint rate, from zero angular momentum) and Jstar (hand rate per joint rate)."""
    C, _ = pos(x); dC, dp = np.zeros((3, 3, 2)), np.zeros((3, 2))
    for j in range(3):
        d = np.eye(3)[j] * h
        (Cp, pp), (Cm, pm) = pos(x + d), pos(x - d)
        dC[j], dp[j] = (Cp - Cm) / (2 * h), (pp - pm) / (2 * h)
    H = I @ A + np.array([M @ (C[:, 0] * dC[j, :, 1] - C[:, 1] * dC[j, :, 0]) for j in range(3)])
    w = -H[1:] / H[0]
    return w, dp[1:].T + np.outer(dp[0], w)


def line(x, goal, budget=BUDGET, ds=0.004, cap=40.0):
    """Straight hand move to goal. -> (arrived, state). Stops at the tilt budget or a singularity (joint rate over cap)."""
    x = x.copy()
    for _ in range(int(6 / ds)):
        e = goal - pos(x)[1]; n = np.linalg.norm(e)
        if n < ds:
            return True, x
        w, J = kin(x); qd = np.linalg.solve(J, e / n)
        xm = x + np.r_[w @ qd, qd] * ds / 2                            # midpoint step
        w, J = kin(xm); qd = np.linalg.solve(J, e / n)
        if abs(qd).max() > cap:
            return False, x
        x = x + np.r_[w @ qd, qd] * ds
        if abs(x[0] - REST[0]) > budget:
            return False, x
    return False, x


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    w, J = kin(REST); w2, J2 = kin(REST + [1.0, 0, 0])
    assert abs(np.linalg.det(J) - np.linalg.det(J2)) < 1e-6 and np.allclose(w, w2, atol=1e-6)   # turning the whole thing changes nothing

    # A. radii where a dynamic singularity exists
    g = np.linspace(-np.pi, np.pi, 121)
    D = np.array([[np.linalg.det(kin(np.array([0, a, b]))[1]) for b in g] for a in g])
    rad = np.array([[np.linalg.norm(pos(np.array([0, a, b]))[1]) for b in g] for a in g])
    cut = (np.sign(D[:, 1:]) != np.sign(D[:, :-1]))
    lo, hi = rad[:, 1:][cut].min(), rad[:, 1:][cut].max()
    print("reach: radius %.2f to %.2f m from the centre of mass. dynamic singularities at radius %.2f to %.2f m" % (
        rad.min(), rad.max(), lo, hi))
    fail = {"clear of the band": [0, 0], "through the band": [0, 0]}
    def draw(inner):
        while True:
            x = np.r_[0, rng.uniform(-np.pi, np.pi, 2)]
            if not inner or np.linalg.norm(pos(x)[1]) < lo - 0.03:
                return x

    for i in range(300):
        x, y = draw(i % 2), draw(i % 2)                               # every other pair inside the band's inner radius
        a, b = pos(x)[1], pos(y)[1]
        r = np.linalg.norm(a + np.linspace(0, 1, 50)[:, None] * (b - a), axis=1)
        if r.min() < rad.min() + 0.02 or r.max() > rad.max() - 0.02:
            continue                                                  # leaves the reach, not the question
        k = "clear of the band" if (r.max() < lo - 0.02 or r.min() > hi + 0.02) else "through the band"
        fail[k][0] += 1; fail[k][1] += not line(x, b, budget=np.inf)[0]
    for k, (n, f) in fail.items():
        print("A. no tilt limit, straight moves %s: %d of %d failed" % (k, f, n))

    if "A" in sys.argv:
        sys.exit()

    # B. the rest map under the budget, then the same tips from 40 one-move starts
    p0 = pos(REST)[1]; U = [np.array([np.cos(a), np.sin(a)]) for a in np.linspace(0, 2 * np.pi, 36, endpoint=False)]
    tips = []
    for u in U:
        lo_, hi_ = 0.0, 3.0
        for _ in range(10):
            mid = (lo_ + hi_) / 2
            lo_, hi_ = (mid, hi_) if line(REST, p0 + mid * u)[0] else (lo_, mid)
        tips.append(lo_ * u)
    tips = np.array(tips); n = np.linalg.norm(tips, axis=1)
    print("B. rest map, 36 lines, %.0f deg budget: reach %.2f to %.2f m, median %.2f" % (np.degrees(BUDGET), n.min(), n.max(), np.median(n)))
    starts = [line(REST, p0 + rng.uniform(0.3, 0.8) * tips[rng.integers(36)])[1] for _ in range(40)]
    for k in (1.0, 0.8, 0.5):
        ok = np.array([[line(x, p0 + k * t)[0] for t in tips] for x in starts])
        # C. first-order guess from the start state alone
        guess = []
        for x in starts:
            w, J = kin(x); c = w @ np.linalg.inv(J)                   # tilt per metre of hand travel, rad/m
            guess.append([abs(x[0] - REST[0] + c @ (p0 + k * t - pos(x)[1])) <= BUDGET for t in tips])
        guess = np.array(guess)
        print("   map pulled in to %.1f: holds for %.0f%% of start-tip pairs, every tip from %d of 40 starts. "
              "C. guess agrees on %.0f%%, says yes when it is no on %.1f%%" % (
                  k, 100 * ok.mean(), ok.all(1).sum(), 100 * (guess == ok).mean(), 100 * (guess & ~ok).mean()))
    print("tilt spent at the 40 starts: median %.2f, most %.2f deg" % tuple(np.degrees([np.median(abs(np.array(starts)[:, 0])), abs(np.array(starts)[:, 0]).max()])))
