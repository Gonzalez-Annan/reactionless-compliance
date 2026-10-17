"""A gamepad read through Windows itself (winmm), no extra installs. Any pad Windows lists in joy.cpl works:
a PS4 or PS5 pad, an Xbox-style pad, a cheap PSP-style one with one stick. No rumble: winmm has none.

    python src/pad.py --learn       # press each control once, saves data/pad_map.json
    python src/pad.py               # print what the pad gives, to check the map. Ctrl+C to stop

Every control is learned on its own, so a stick, a trigger, a button or a D-pad direction can do any job.
Anything skipped is left to the keyboard. Without data/pad_map.json the layout is a guess at a PS4 pad on
DirectInput. NOT TESTED on a real pad yet: no pad was plugged in when this was written."""
import ctypes
import json
import sys
import time
from pathlib import Path

MAP = Path(__file__).parent.parent / "data" / "pad_map.json"
DEAD = 0.1                              # stick deadband: tune in the pilot
CENTRE = 65535                          # hat at rest
FULL = 65535                            # ponytail: winmm scales nearly every pad to 0..65535. A pad where every step
                                        # says "nothing held" needs the ranges from joyGetDevCapsW here
STEPS = (("right", "the left stick right"), ("left", "the left stick left"), ("up", "the left stick up"),
         ("down", "the left stick down"), ("away", "R2 (dot away from you)"), ("toward", "L2 (dot toward you)"),
         ("cam_right", "the right stick right (camera)"), ("cam_left", "the right stick left"),
         ("cam_up", "the right stick up"), ("cam_down", "the right stick down"), ("go", "Cross (send, start)"),
         ("snap", "Circle (dot back on the hand)"), ("fine", "L1 (fine, hold)"), ("view", "Square (cage view)"),
         ("park", "Triangle (give the goal up)"))
DS4 = {"right": ["axis", 0, 32767, 65535], "left": ["axis", 0, 32767, 0], "up": ["axis", 1, 32767, 0],
       "down": ["axis", 1, 32767, 65535], "away": ["axis", 5, 0, 65535], "toward": ["axis", 4, 0, 65535],
       "cam_right": ["axis", 2, 32767, 65535], "cam_left": ["axis", 2, 32767, 0], "cam_up": ["axis", 3, 32767, 0],
       "cam_down": ["axis", 3, 32767, 65535], "go": ["button", 1], "snap": ["button", 2], "fine": ["button", 4],
       "view": ["button", 0], "park": ["button", 3]}


def level(c, axes, buttons, hat):
    """How far control c is pushed, 0 to 1. axes: the six winmm axes, buttons: their bits, hat: hundredths of a degree."""
    if c is None:
        return 0.0
    if c[0] == "axis":
        _, i, r, e = c
        return min(max((axes[i] - r) / (e - r), 0.0), 1.0)
    if c[0] == "button":
        return float(buttons >> c[1] & 1)
    return float(hat != CENTRE and abs((hat - c[1] + 18000) % 36000 - 18000) < 9000)    # a diagonal counts for both sides


def state(mp, axes, buttons, hat):
    """-> (sx, sy, depth, cam_x, cam_y, go, snap, fine, view, park), the keyboard's tuple plus view and park."""
    v = {k: level(mp.get(k), axes, buttons, hat) for k, _ in STEPS}
    two = [v[a] - v[b] for a, b in (("right", "left"), ("up", "down"), ("away", "toward"), ("cam_right", "cam_left"), ("cam_down", "cam_up"))]
    return tuple(0.0 if abs(x) < DEAD else x for x in two) + tuple(v[k] > 0.5 for k in ("go", "snap", "fine", "view", "park"))


def pick(rest, held):
    """What changed between letting go of everything and holding one control -> its entry in the map, or None."""
    (a0, b0, h0), (a1, b1, h1) = rest, held
    if b1 & ~b0:
        return ["button", (b1 & ~b0).bit_length() - 1]
    if h1 != h0 and h1 != CENTRE:
        return ["hat", h1]
    i = max(range(6), key=lambda i: abs(a1[i] - a0[i]))
    return ["axis", i, a0[i], a1[i]] if abs(a1[i] - a0[i]) > 0.3 * FULL else None


class INFO(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint32) for n in ("dwSize", "dwFlags", "dwXpos", "dwYpos", "dwZpos", "dwRpos", "dwUpos",
                                               "dwVpos", "dwButtons", "dwButtonNumber", "dwPOV", "dwReserved1", "dwReserved2")]


def raw():
    """-> read(): (six axes, button bits, hat) of the first pad Windows lists, None while it is unplugged."""
    if not hasattr(ctypes, "windll"):
        raise SystemExit("The pad is read through Windows (winmm). Off Windows, use the keyboard.")
    w, j = ctypes.windll.winmm, INFO(dwSize=ctypes.sizeof(INFO), dwFlags=255)     # 255: JOY_RETURNALL
    n = next((n for n in range(16) if w.joyGetPosEx(n, ctypes.byref(j)) == 0), None)
    if n is None:
        raise SystemExit("No pad found. Plug it in and check it moves in joy.cpl (Win+R, joy.cpl), then try again.")

    def read():
        if w.joyGetPosEx(n, ctypes.byref(j)):
            return None
        return (j.dwXpos, j.dwYpos, j.dwZpos, j.dwRpos, j.dwUpos, j.dwVpos), j.dwButtons, j.dwPOV
    return read


def reader(path=MAP):
    """-> now(): the state() of the pad, all at rest while it is unplugged."""
    mp, read = json.loads(path.read_text()) if path.exists() else DS4, raw()
    print(f"pad map: {path}" if path.exists() else "no pad map, guessing a PS4 pad. Run python src/pad.py --learn", flush=True)

    def now():
        r = read()
        return state(mp, *r) if r else (0.0,) * 5 + (False,) * 5
    return now


def learn(read, path=MAP):
    input("Let go of everything on the pad, then press Enter.")
    rest, mp = read(), {}
    for k, what in STEPS:
        input(f"Hold {what}, or whatever you want for it, and press Enter. Nothing held skips it: ")
        c = pick(rest, read())
        print(f"  {k}: {c or 'skipped, the keyboard does it'}")
        if c in mp.values():
            print("  that one is taken already: run --learn again")
        if c:
            mp[k] = c
    path.write_text(json.dumps(mp, indent=1))
    print(f"saved {path}")


if __name__ == "__main__":
    if "--learn" in sys.argv:
        learn(raw())
    now = reader()
    while True:
        print(" ".join(f"{x:+.2f}" if isinstance(x, float) else str(int(x)) for x in now()), end="   \r", flush=True)
        time.sleep(0.05)
