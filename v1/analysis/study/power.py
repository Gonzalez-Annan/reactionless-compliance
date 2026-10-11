"""Maps study, sample size: power of the main contrast, written before any participant. Numpy only, one process, seeded.
Run from v1: python analysis/study/power.py        (about two minutes)
The first model was too kind in five places. Each is now a knob with its own row: see ASSUMED.
Main contrast: test goals with a wrong go (a command the fence had to bring back), rest map against the map from
the current state, one count per participant and map, exact two-sided Wilcoxon signed-rank test, zeros dropped.
A wrong go needs a trap: a test goal the map on screen allows and the arm cannot reach from where the lead goal
left it. So a block is its traps, and a participant is what they do on a trap.

FROM THE RECORDS (simulation and the scripted rule, no people):
  rest map, goals 0.5 to 1.0 of the way out: it allows a goal the arm cannot reach on 8% of 3920 start and line
  pairs (13% for goals 0.8 to 1.0), and on 3 of the 49 scripted scored goals inside it.   wide98_medium.npz, data/scripted
  As built a scored goal is a trap 3 times in 112. The four scripted rest blocks happened to hold 2 in 28.
  The rule sent on both, 2 wrong goes. It waits for the current-state map by construction and met no trap there,
  so the scripted run says nothing about that map.
  Held map and no map: no wrong go in the scripted run (0 of 56). Looked-up map: 0.05 m too much on 1.4% of pairs
  against the full simulation (D55), not simulated here.
ASSUMED, because human rates are unknown. Every one of these is a guess:
  a    chance a participant sends on a trap under the rest map (the rule: 1)
  b    chance a participant sends on a trap before the current-state map is current (the rule: 0)
  SD   how far participants differ, on the logit scale
  rho  how a person's lean under the rest map goes with their lean under the current-state map. The first model had +1:
       who sends under one sends under the other. 0: no link. -1: who follows a map sends under the rest map and
       waits for the current one. +1 is the kind case, it takes the person out of the difference.
  met  the design plants 2 traps among the 10 test goals of a block (8 of them inside the rest map). Truth is
       rehearsed from where the participant's own lead move ends, so a planted trap is one for them with chance met.
  e    chance of a wrong go on a test goal inside the rest map that is not a trap (a dot put down short of the goal
       or past it), the same under both maps. The rule: 0. It adds noise and no difference.
  once after a wrong go the participant sends on no other trap of that block, under either map (learning in a block)
  a2   a for the half who met the current-state map in an earlier block than the rest map (carry-over)
  As built the traps fall by chance, 7 scored goals a block.
  The wrong goes per block that a row assumes are printed next to it."""
import numpy as np

ALPHA, REPS, SD, SEED, IN = 0.05, 4000, 1.0, 0, 8
memo = {}


def wilcoxon(d):
    """Exact two-sided p of the signed-rank test on the paired differences d, zeros dropped, ties share a rank.
    -> (p, the + side is the heavier one)."""
    d = np.sort(d[d != 0])
    if d.tobytes() not in memo:
        _, inv, c = np.unique(abs(d), return_inverse=True, return_counts=True)
        r = (2 * np.cumsum(c) - c + 1)[inv]             # twice the mid-rank, so it stays a whole number
        f = np.ones(1)                                  # null: every sign is a coin. f[k] = ways the + ranks add up to k
        for x in r:
            f = np.convolve(f, np.r_[1, np.zeros(x - 1), 1])
        far = r[d > 0].sum() - r.sum() / 2
        memo[d.tobytes()] = f[abs(np.arange(len(f)) - r.sum() / 2) >= abs(far) - 1e-9].sum() / f.sum(), far > 0
    return memo[d.tobytes()]


def power(n, a, b, traps, alpha=ALPHA, sd=SD, rho=1.0, e=0.0, once=False, a2=None):
    """Share of REPS made-up studies of n participants in which the rest map has more wrong goes at p < alpha.
    traps per block: a fixed number, or (planted or scored goals, chance each is a trap for this participant).
    -> (power, wrong goes per block under the rest map, under the current-state map)."""
    rng, more = np.random.default_rng(SEED), np.random.default_rng(SEED + 1)    # more: draws power.py did not make
    z = rng.normal(0, sd, (REPS, n))                    # the same made-up people in every row
    t = rng.binomial(*traps, (2, REPS, n)) if isinstance(traps, tuple) else np.full((2, REPS, n), traps)
    la, lb = (np.log(q / (1 - q)) for q in np.clip((a, b), 1e-9, 1 - 1e-9))
    if a2:
        la = np.where(np.arange(n) < n // 2, np.log(a2 / (1 - a2)), la)
    zb = rho * z + np.sqrt(1 - rho ** 2) * more.normal(0, sd, (REPS, n))
    p = 1 / (1 + np.exp(-la - z)), 1 / (1 + np.exp(-lb - zb))
    w = [rng.binomial(t[i], p[i]) for i in (0, 1)]
    w = [(np.minimum(x, 1) if once else x) + more.binomial(IN - t[i], e) for i, x in enumerate(w)]
    return np.mean([q < alpha and up for q, up in map(wilcoxon, w[0] - w[1])]), w[0].mean(), w[1].mean()


BUILT, MET = (7, 3 / 112), (2, 0.8)
HARD = dict(rho=0, e=0.05)                              # with traps=MET: every knob at its less kind setting but once and a2
ROWS = [    # what is assumed, a, b, traps per block, the rest
    ("AS power.py (rho 1, every planted trap met, e 0)", ),
    ("design, 2 traps: people do as the scripted rule", 1.0, 0.0, 2, {}),
    ("design, 2 traps: most send, most wait", 0.9, 0.2, 2, {}),
    ("design, 2 traps: most send, half do not wait", 0.9, 0.5, 2, {}),
    ("design, 2 traps: some hold back, a third do not wait", 0.7, 0.35, 2, {}),
    ("  the same, people differ twice as much (SD 2)", 0.7, 0.35, 2, dict(sd=2 * SD)),
    ("  the same, if H1 shared 0.05 with 3 other tests", 0.7, 0.35, 2, dict(alpha=ALPHA / 4)),
    ("design with 1 trap: most send, most wait", 0.9, 0.2, 1, {}),
    ("design with 1 trap: some hold back, a third do not wait", 0.7, 0.35, 1, {}),
    ("as built, trap 3 in 112: people do as the scripted rule", 1.0, 0.0, BUILT, {}),
    ("as built, trap 3 in 112: most send, most wait", 0.9, 0.2, BUILT, {}),
    ("as built, trap 2 in 28: people do as the scripted rule", 1.0, 0.0, (7, 2 / 28), {}),
    ("ONE KNOB AT A TIME, on a 0.7 b 0.35, 2 traps", ),
    ("  a planted trap is one for the person 4 times in 5", 0.7, 0.35, MET, {}),
    ("  leans under the two maps not linked (rho 0)", 0.7, 0.35, 2, dict(rho=0)),
    ("  leans opposed (rho -1)", 0.7, 0.35, 2, dict(rho=-1)),
    ("  wrong go on 1 other goal in 20 (e 0.05)", 0.7, 0.35, 2, dict(e=0.05)),
    ("  no second trap after a wrong go (once)", 0.7, 0.35, 2, dict(once=True)),
    ("  half met the current map first and send at 0.4 (a2)", 0.7, 0.35, 2, dict(a2=0.4)),
    ("TOGETHER: trap met 4 in 5, rho 0, e 0.05", ),
    ("  people do as the scripted rule", 1.0, 0.0, MET, HARD),
    ("  most send, most wait", 0.9, 0.2, MET, HARD),
    ("  most send, half do not wait", 0.9, 0.5, MET, HARD),
    ("  most send, most do not wait", 0.9, 0.7, MET, HARD),
    ("  some hold back, a third do not wait", 0.7, 0.35, MET, HARD),
    ("    and no second trap after a wrong go", 0.7, 0.35, MET, dict(HARD, once=True)),
    ("    and 3 traps planted, not 2", 0.7, 0.35, (3, 0.8), HARD),
    ("  half hold back, most wait", 0.5, 0.2, MET, HARD),
    ("  most hold back, nearly all wait", 0.3, 0.1, MET, HARD),
]

if __name__ == "__main__":
    assert wilcoxon(np.ones(6, int))[0] == 2 / 64 and abs(wilcoxon(np.array([1, 2, 3, -1]))[0] - 6 / 16) < 1e-12
    print(__doc__)
    print("\nalpha %g two-sided, %d made-up studies per cell (so +-0.02), SD %g, seed %d. Power = rest map worse at p < alpha." % (
        ALPHA, REPS, SD, SEED))
    print("%-58s %-13s %s" % ("", "wrong goes a", "power at n ="))
    print("%-58s %-13s %s" % ("assumed", "block rest live", "  12    20    30"))
    for s, *row in ROWS:
        if not row:
            print(s)
            continue
        a, b, t, kw = row
        got = [power(n, a, b, t, **kw) for n in (12, 20, 30)]
        print("%-58s %.2f  %.2f     %.2f  %.2f  %.2f" % ((s,) + got[2][1:] + tuple(g[0] for g in got)))
    for s, t, kw in (("check, as power.py: nobody waits, the maps are the same", 2, {}), ("check, together: the same", MET, HARD)):
        same = [power(n, 0.9, 0.9, t, **kw)[0] for n in (12, 20, 30)]      # no difference to find
        assert max(same) < ALPHA / 2 + 0.01, same       # one side of a two-sided test, so about 0.025 at most
        print("%-58s                %.2f  %.2f  %.2f   (must stay under %.3f)" % ((s,) + tuple(same) + (ALPHA / 2 + 0.01,)))
    print("Four conditions were decided (D63). With five: 8 test goals a block, still 2 traps, the same two blocks and the same n,")
    print("so the rows with e 0 would not change. With e there would be 6 goals inside the rest map and not 8, so a little less noise.")
