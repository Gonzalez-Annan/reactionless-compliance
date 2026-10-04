"""Session runner: participant ID -> counterbalanced condition order -> one CSV per trial. Owner: P1.

    python -m src.session --participant P01
    python -m src.session --participant P01 --dry-run        # every scheme replaced by "demo"
    python -m src.session --participant P01 --start-at 4     # resume after a crash / Esc

Design (plan, Week 8): within-subject, 5 schemes x 2 task modes, 2 closed
loops per trial, medium servicer.
  * Task modes run as two blocks; odd participants do 3-D first, even 6-D first.
  * Within a block the scheme order is a row of a Williams Latin square
    (n = 5 -> 10 rows; balanced for first-order carry-over). The second block
    uses the row 5 further on, so it differs from the first.
  * The controller name is hidden from the participant.

Output: data/participants/<ID>/<ID>_c<k>_<mode>_<scheme>.csv plus order.csv.
data/participants/ is git-ignored. Back the folder up after every participant.
"""
import argparse
import csv
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "participants"

# Names P3 must use as keys in src/schemes.py SCHEMES (plan Section 3, schemes 1-5).
SCHEMES = ["dls", "grad_proj", "impedance", "rns", "impedance_rns"]
MODES = ["3d", "6d"]
LOOPS = 2
MODEL = "medium"


def williams_square(n):
    """2n rows for odd n (n rows for even n); each row is an order of 0..n-1.
    Every condition appears once per position, and each ordered pair of
    neighbours appears equally often."""
    first, lo, hi = [0], 1, n - 1
    while len(first) < n:
        first.append(lo)
        lo += 1
        if len(first) < n:
            first.append(hi)
            hi -= 1
    rows = [[(c + r) % n for c in first] for r in range(n)]
    if n % 2:
        rows += [row[::-1] for row in rows]
    return rows


def participant_number(pid):
    digits = "".join(ch for ch in pid if ch.isdigit())
    if not digits:
        raise ValueError(f"participant ID needs a number, e.g. P01 (got {pid!r})")
    return int(digits)


def condition_order(pid):
    """List of (mode, scheme) for this participant, in running order."""
    k = participant_number(pid)
    square = williams_square(len(SCHEMES))
    modes = MODES if k % 2 else MODES[::-1]
    rows = [square[(k - 1) % len(square)], square[(k - 1 + len(SCHEMES)) % len(square)]]
    return [(mode, SCHEMES[i]) for mode, row in zip(modes, rows) for i in row]


def run_session(pid, dry_run=False, start_at=1):
    from src.teleop import Teleop  # imports pygame/OpenGL only when actually running

    order = condition_order(pid)
    out = DATA_DIR / pid
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "order.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "mode", "scheme"])
        for i, (mode, scheme) in enumerate(order, 1):
            w.writerow([i, mode, scheme])

    print(f"{pid}: {len(order)} conditions -> {out}")
    for i, (mode, scheme) in enumerate(order, 1):
        if i < start_at:
            continue
        controller = "demo" if dry_run else scheme
        log = out / f"{pid}_c{i:02d}_{mode}_{scheme}.csv"
        print(f"  condition {i:2d}/{len(order)}: {mode} {scheme}" + ("  (dry run: demo)" if dry_run else ""))
        tele = Teleop(
            MODEL, mode, controller, log_path=log, task_loops=LOOPS, blind=True,
            banner=f"Trial {i} of {len(order)}  -  {mode.upper()} task",
            meta={"participant": pid, "condition": i, "scheme": scheme, "dry_run": dry_run},
        )
        tele.run()
        if not tele.task.done:
            print(f"  stopped during condition {i}. Resume with: --start-at {i}")
            return 1
        print("    done. Give the questionnaire now; the next trial waits for SPACE.")
    print(f"{pid}: session complete. Back up {out} now.")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Run one participant's session")
    ap.add_argument("--participant", required=True, help="e.g. P01")
    ap.add_argument("--dry-run", action="store_true", help='use "demo" for every scheme (pipeline test)')
    ap.add_argument("--start-at", type=int, default=1, help="first condition to run (resume)")
    ap.add_argument("--show-order", action="store_true", help="print the condition order and exit")
    args = ap.parse_args()
    if args.show_order:
        for i, (mode, scheme) in enumerate(condition_order(args.participant), 1):
            print(f"{i:2d}  {mode}  {scheme}")
        return 0
    return run_session(args.participant, args.dry_run, args.start_at)


if __name__ == "__main__":
    sys.exit(main())
