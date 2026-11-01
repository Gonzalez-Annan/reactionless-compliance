"""Watch trials on our screen, in real time.

    python src/view.py                      # small servicer, position task: DLS, then Imp.+RNS
    python src/view.py medium 6d 1 4 5      # size, mode, schemes...
"""
import copy
import ctypes
import functools
import sys
import threading
import time
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

from schemes import NAMES
from task import run_trial, DECIM


SHELL = ([0.2, 1, 0.3, 0.22], [1, 0.55, 0.1, 0.22], [0.2, 0.5, 1, 0.08])   # faces: base tilt under 0.5, 1, 2 deg
W, H, FPS = 1280, 720, 30               # ponytail: fixed window, 42 fps measured at this size; size it from the screen if a study laptop needs it
EYE = np.eye(3).flatten()   # orientation of every mark: they are spheres, lines and triangles laid by their points
HITCH = 0.25                            # s behind real time that counts as a hitch, not a slow frame
WIN = hasattr(ctypes, "windll")
INK, DIM, CARD = (235, 240, 248), (150, 160, 182), (12, 16, 28, 200)
TK = dict(Left=263, Up=265, Right=262, Down=264, Prior=266, Next=267, Home=268, End=269, Return=257,
          Delete=261, Shift_L=340, Insert=260, Control_R=345)        # Tk key name -> the GLFW code teleop's key_callback takes


class BIH(ctypes.Structure):            # BITMAPINFOHEADER of a frame: 24 bit, top row first
    _fields_ = [("size", ctypes.c_uint32), ("w", ctypes.c_int32), ("h", ctypes.c_int32), ("planes", ctypes.c_uint16),
                ("bits", ctypes.c_uint16), ("rest", ctypes.c_uint32 * 6)]


@functools.cache
def font(size, bold=False):
    try:
        return ImageFont.truetype(f"C:/Windows/Fonts/{'seguisb' if bold else 'segoeui'}.ttf", size)
    except OSError:                     # not Windows
        return ImageFont.load_default(size)


def draw_envelope(scn, n, tips, tris, edges):
    """Teleop work envelope through the probe rays (tips[direction][shell]): a blue cage where the base reaches
    2 deg, and inside it a yellow and a green volume for 1 and 0.5 deg (filled, no lines: as wireframes the
    three were a tangle). tips grows while the rays are run; only finished corners are drawn. -> next free geom index.
    The triangle frames are worked out in numpy: one by one they took 85 ms for the full cage, now 15."""
    k, eye = len(tips), EYE
    P = np.asarray(tips, dtype=float).reshape(k, len(SHELL), 3)
    for i, j in edges:
        if max(i, j) < k and n < scn.maxgeom:
            mujoco.mjv_initGeom(scn.geoms[n], mujoco.mjtGeom.mjGEOM_SPHERE, [0.001, 0, 0], P[i, -1], eye, [0.4, 0.85, 1, 0.9])
            mujoco.mjv_connector(scn.geoms[n], mujoco.mjtGeom.mjGEOM_LINE, 2, P[i, -1], P[j, -1])
            n += 1
    t = np.array([x for x in tris if max(x) < k], dtype=int).reshape(-1, 3)
    for sh, rgba in enumerate(SHELL):
        a, b, c = (P[t[:, q], sh] for q in range(3))
        e1, e2 = b - a, c - a                                    # both windings: seen from inside and outside
        M = np.stack([np.stack([e1, e2, np.cross(e1, e2)], 2), np.stack([e2, e1, np.cross(e2, e1)], 2)], 1).reshape(-1, 2, 9)
        for p, ms in zip(a, M):
            if n + 2 > scn.maxgeom:
                break
            for mm in ms:
                mujoco.mjv_initGeom(scn.geoms[n], mujoco.mjtGeom.mjGEOM_TRIANGLE, [1, 1, 1], p, mm, rgba)
                n += 1
    return n


key_callback = None                     # teleop sets this off Windows to get key presses from the window


def marks(scn, s, ref, kept=None):
    """What the hook adds to the scene after the model: the cage, the dot, the maps, the hand ball, the hand to goal
    line. kept: a list the screen holds; the cage is drawn first and left in the scene while its rays and n stay the
    same (update_scene writes only the model's geoms), so most frames skip its 1440 geoms."""
    n = scn.ngeom
    if "cage" in s and s.get("show", 2):   # teleop work envelope; show: 0 slices, 1 both, 2 shells
        rays = s.get("rays", [])
        tips = list(rays)                   # the probe thread may append while we draw
        key = (n, id(rays), len(tips), id(s["cage"]))
        if kept and kept[0] == key:
            n = kept[1]
        else:
            n = draw_envelope(scn, n, tips, *s["cage"])
            if kept is not None:
                kept[:] = [key, n, rays, s["cage"]]     # holding both keeps their ids theirs
    elif kept:
        kept.clear()                        # the marks below write over the kept cage
    mujoco.mjv_initGeom(scn.geoms[n], mujoco.mjtGeom.mjGEOM_SPHERE, [0.012, 0, 0],   # green = where the hand should be
                        ref[0], EYE, s.get("rgba", [0, 1, 0, 0.6]))
    n += 1
    if s.get("far") is not None:        # teleop: how far the hand would get toward an unreachable dot
        mujoco.mjv_initGeom(scn.geoms[n], mujoco.mjtGeom.mjGEOM_SPHERE, [0.012, 0, 0], s["far"],
                            EYE, [0.2, 0.5, 1, 0.9])
        n += 1
    if s.get("tgt") is not None:        # pilot: the goal to hold the hand on
        mujoco.mjv_initGeom(scn.geoms[n], mujoco.mjtGeom.mjGEOM_SPHERE, [0.02, 0, 0], s["tgt"],
                            EYE, [1, 0.2, 0.9, 0.5])
        n += 1
    if s.get("show", 0) < 2:               # the shells cut by the three planes through the dot
        for seg, rgba in s.get("cuts", ()):
            for a, b in seg:
                if n + 8 < scn.maxgeom:
                    mujoco.mjv_initGeom(scn.geoms[n], mujoco.mjtGeom.mjGEOM_SPHERE, [0.001, 0, 0], a, EYE, rgba)
                    mujoco.mjv_connector(scn.geoms[n], mujoco.mjtGeom.mjGEOM_LINE, 3, a, b)
                    n += 1
    if "hand" in s and n < scn.maxgeom:    # teleop: live singularity margin, a ball on the hand
        mujoco.mjv_initGeom(scn.geoms[n], mujoco.mjtGeom.mjGEOM_SPHERE, [0.03, 0, 0], s["hand"][0],
                            EYE, s["hand"][1])
        n += 1
    if "link" in s and n < scn.maxgeom:    # white = hand to goal (teleop)
        g = scn.geoms[n]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, [0.001, 0, 0], s["link"][1], EYE, [1, 1, 1, 0.9])
        if np.linalg.norm(s["link"][1] - s["link"][0]) > 1e-4:
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, 2, *s["link"])
        n += 1
    scn.ngeom = n


@functools.lru_cache(maxsize=32)
def cards(cap, val, rgb, hud, width):
    """The two cards, margin or map state top left and score top right, as pictures and where they go. Kept: their
    text changes about once a second, and drawn every frame it held the sim's thread back by a fifth (one GIL)."""
    out, m = [], ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    if cap is not None:
        a, b = font(15), font(28, True)
        w = round(max(m.textlength(cap, font=a), m.textlength(val, font=b) + 30) + 38)
        c = Image.new("RGBA", (w + 1, 87))
        g = ImageDraw.Draw(c)
        g.rounded_rectangle((0, 0, w, 86), 14, fill=CARD)
        g.text((18, 12), cap, font=a, fill=DIM)
        g.ellipse((18, 49, 34, 65), fill=rgb)
        g.text((46, 36), val, font=b, fill=INK)
        out.append(((20, 20), c))
    if hud:
        f, fb = font(18), font(18, True)
        rows = list(zip(*(x.split("\n") for x in hud)))
        w = round(max(m.textlength(n, font=f) + m.textlength(v, font=fb) for n, v in rows) + 76)
        c = Image.new("RGBA", (w + 1, 25 + 30 * len(rows)))
        g = ImageDraw.Draw(c)
        g.rounded_rectangle((0, 0, w, 24 + 30 * len(rows)), 14, fill=CARD)
        for i, (n, v) in enumerate(rows):
            g.text((18, 12 + 30 * i), n, font=f if v else fb, fill=DIM if v else INK)
            g.text((w - 18, 12 + 30 * i), v, font=fb, fill=INK, anchor="ra")
        out.append(((width - 20 - w, 20), c))
    return out


def hud(im, s):
    """The text on the scene, the same as the viewer had, in cards: margin or map state top left, score top right,
    a screen to read in the middle with the scene dimmed behind it."""
    hand = s.get("hand")
    for xy, c in cards(s.get("label", "margin (1 = ready pose)") if hand else None, s.get("text") or "",
                       tuple(int(255 * c) for c in hand[1][:3]) if hand else None, tuple(s.get("hud") or ()), im.width):
        im.paste(c, xy, c)
    g = ImageDraw.Draw(im, "RGBA")
    if s.get("say"):
        head, body = s["say"]
        fh, fb = font(34, True), font(20)
        g.rectangle((0, 0, *im.size), fill=(0, 0, 0, 120))
        bx = g.multiline_textbbox((0, 0), body, font=fb, spacing=8)
        w, h = max(bx[2], g.textlength(head, font=fh)) + 80, bx[3] + 124
        x0, y0 = (im.width - w) / 2, (im.height - h) / 2
        g.rounded_rectangle((x0, y0, x0 + w, y0 + h), 18, fill=(14, 18, 32, 245))
        g.text((x0 + 40, y0 + 30), head, font=fh, fill=INK)
        g.multiline_text((x0 + 40, y0 + 92), body, font=fb, fill=(205, 212, 228), spacing=8)
    return im


class Screen:
    """Our window in place of MuJoCo's viewer (D78): the scene drawn offscreen, the text on it in a real font, and no
    panel on it that can change the physics mid-trial. Keeps what callers used of the viewer: .cam, .close(), .is_running().
    window=False draws frames and opens nothing (tests, proof frames)."""

    def __init__(self, m, d, window=True):
        m.vis.global_.offwidth, m.vis.global_.offheight = max(W, m.vis.global_.offwidth), max(H, m.vis.global_.offheight)
        self.r, self.cam, self.root, self.kept = mujoco.Renderer(m, H, W), mujoco.MjvCamera(), None, []
        mujoco.mjv_defaultFreeCamera(m, self.cam)      # azimuth 90, elevation -45: what the stick mapping and the scripted rule assume
        self.cam.lookat[:] = (d.xpos[m.body("l1").id] + d.site_xpos[m.site("ee").id]) / 2 + [0, 0, .2]
        self.cam.distance = 2.0                         # opens on the arm, the bus low in the frame, the same for every size
        if not window:
            return
        import tkinter as tk
        from PIL import ImageTk
        if WIN:
            ctypes.windll.user32.SetProcessDPIAware()  # one screen pixel per frame pixel
        self.root = tk.Tk()
        self.root.title("Reactionless Compliance")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.ph = ImageTk.PhotoImage("RGB", (W, H))
        self.box = tk.Label(self.root, image=self.ph, bd=0, bg="black")
        self.box.pack()
        self.root.bind("<KeyPress>", lambda e: key_callback and key_callback(TK.get(e.keysym, 0)))
        self.root.bind("<ButtonPress-1>", lambda e: setattr(self, "xy", (e.x, e.y)))
        self.root.bind("<B1-Motion>", self.turn)
        self.root.bind("<MouseWheel>", lambda e: setattr(self.cam, "distance", self.cam.distance * 0.9 ** (e.delta / 120)))
        self.root.update()
        self.dc = ctypes.windll.user32.GetDC(self.box.winfo_id()) if WIN else None

    def turn(self, e):                                  # drag to turn the camera, the wheel zooms
        (x, y), self.xy = self.xy, (e.x, e.y)
        self.cam.azimuth -= 0.3 * (e.x - x)
        self.cam.elevation = np.clip(self.cam.elevation - 0.3 * (e.y - y), -89, 89)

    def frame(self, d, s, ref):
        """-> the picture: the scene, the marks when ref is given, the text."""
        self.r.update_scene(d, self.cam)
        if ref is not None:
            marks(self.r.scene, s, ref, self.kept)
        return hud(Image.fromarray(self.r.render()), s)

    def show(self, im):
        if WIN:     # straight to the window: 4 ms against 13 ms for Tk's own copy, which keeps the screen at 30 fps
            ctypes.windll.gdi32.StretchDIBits(self.dc, 0, 0, W, H, 0, 0, W, H, im.tobytes("raw", "BGR"),
                                              ctypes.byref(BIH(40, W, -H, 1, 24)), 0, 0xCC0020)
        else:       # ponytail: not run off Windows, the team is on Windows
            self.ph.paste(im)
        self.root.update()

    def is_running(self):
        return self.root is not None

    def close(self):
        if self.root:
            if WIN:
                ctypes.windll.user32.ReleaseDC(self.box.winfo_id(), self.dc)
            self.ph = self.box = None                   # Tk objects die on the thread that made them
            self.root.destroy()
            self.root = None
        if self.r:
            self.r.close()
            self.r = None


class Live:
    """The screen on its own thread, the way MuJoCo's viewer ran: the sim posts a copy of its state, the thread draws
    the newest one. Drawn in the sim's thread a frame with the cage (45 to 100 ms) held up the steps and the arm ran at
    0.15 of real time. Tk, GL and the window's DC are made and used on this thread only."""

    def __init__(self, m, d):
        self.post, self.cam, self.new, self.up, self.stop = None, None, threading.Event(), threading.Event(), False
        self.th = threading.Thread(target=self.run, args=(m, d), daemon=True)
        self.th.start()
        self.up.wait()

    def run(self, m, d):
        v = None
        try:                                            # any error here ends the run, never a frozen window
            v = Screen(m, d)
            self.cam = v.cam                            # teleop turns the camera from the pad's right stick
            self.up.set()
            while not self.stop and v.is_running():
                if self.new.wait(1 / FPS):
                    self.new.clear()
                    v.show(v.frame(*self.post))
                elif v.root:
                    v.root.update()                     # the window stays live while nothing new comes
        finally:
            self.stop = True
            self.up.set()
            if v:
                v.close()

    def is_running(self):
        return not self.stop

    def close(self):
        self.stop = True
        self.th.join()


def viewer_hook(show_ref=True):
    """on_step callback for run_trial: opens the screen on the first step, posts the state to it at most FPS times a
    second, paces the sim to real time. A hitch over HITCH (a rehearsal holds a step up to 1.5 s) slips the clock, so
    the arm does not run fast to catch up while the operator holds the stick."""
    s, last = {}, [0.0]

    def on_step(m, d, ref):
        if "v" not in s:
            s["v"], s["t0"] = Live(m, d), time.time() - d.time
        v, now = s["v"], time.time()
        if now - last[0] >= 1 / FPS:
            last[0], v.post = now, (copy.copy(d), dict(s), ref if show_ref else None)
            v.new.set()
        if now - s["t0"] - d.time > HITCH:
            s["t0"] = now - d.time
        time.sleep(max(0, s["t0"] + d.time + DECIM * m.opt.timestep - time.time()))
        return v.is_running()

    return s, on_step


def summary(r):
    return (f"hand error {r['rms_pos_mm']:.1f} mm, peak base drift {r['max_base_deg']:.2f} deg, "
            f"a joint at its limit {r['limit_frac']:.0%} of the time")


if __name__ == "__main__":
    a = sys.argv[1:]
    for scheme in [int(x) for x in a[2:]] or [1, 5]:
        size, mode = a[0] if a else "small", a[1] if len(a) > 1 else "3d"
        print(f"{NAMES[scheme]} on the {size} servicer, {mode} task", flush=True)
        s, hook = viewer_hook()
        print("  " + summary(run_trial(size, scheme, mode, loops=1, on_step=hook)), flush=True)
        time.sleep(2)
        s["v"].close()
