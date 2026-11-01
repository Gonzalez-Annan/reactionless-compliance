"""Images and the demo video for the report and slides, rendered from the real models. Owner: P1/P5.

    python -m analysis.render stills       # figures/render_*.png (servicers, home pose, contact)
    python -m analysis.render video        # figures/demo_dls_vs_imprns.mp4 (plan: scheme 1 vs 5, 6-D,
                                           # antenna constraint visible, 20-30 s)

Off-screen MuJoCo rendering; the scripted ORU reference drives both arms identically, so the
only difference on screen is the redundancy-resolution scheme.
"""
import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.teleop import ANTENNA_LIMIT_DEG, Teleop, add_line, add_sphere, site_pose

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"


def _font(size):
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def big_buffer(model, w=4000, h=2400):
    model.vis.global_.offwidth, model.vis.global_.offheight = w, h
    return model


def camera(lookat, distance, azimuth=135, elevation=-22):
    c = mujoco.MjvCamera()
    c.lookat[:] = lookat
    c.distance, c.azimuth, c.elevation = distance, azimuth, elevation
    return c


def decorate(tele, scn, antenna=True, task=True, target=True, contacts=True):
    """The same overlays as the teleop window: antenna lines, task path, target, contacts."""
    if antenna:
        a_pos, a_R = site_pose(tele.model, tele.data, "antenna")
        err = tele.antenna_error_deg()
        L = 0.6 + 2.0 * tele.model.geom("base_geom").size[2]
        add_line(scn, a_pos, a_pos + L * tele.antenna_z0, [0.2, 0.9, 0.2, 1], width=4)
        add_line(scn, a_pos, a_pos + L * a_R[:, 2],
                 [0.95, 0.15, 0.15, 1] if err > ANTENNA_LIMIT_DEG else [0.95, 0.8, 0.1, 1], width=4)
    if task and tele.task is not None:
        wps = tele.task.waypoints
        for (_, p0, _), (_, p1, _) in zip([(None, *tele.task.start)] + wps[:-1], wps):
            add_line(scn, p0, p1, [0.7, 0.7, 0.7, 0.6], width=2)
        for i, (_, p, _) in enumerate(wps):
            add_sphere(scn, p, 0.014, [1.0, 0.55, 0.1, 0.9] if i == tele.task.index else [0.75, 0.75, 0.75, 0.7])
    if target:
        add_sphere(scn, tele.target_pos, 0.025, [0.3, 0.7, 1.0, 0.75])
    if contacts:
        for i in range(tele.data.ncon):
            add_sphere(scn, tele.data.contact[i].pos, 0.02, [1.0, 0.1, 0.1, 0.9])


def render(tele, renderer, cam, **kw):
    renderer.update_scene(tele.data, cam)
    decorate(tele, renderer.scene, **kw)
    return renderer.render()


def label(img, lines, pos=(14, 10), size=26, colors=None):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    y = pos[1]
    for i, t in enumerate(lines):
        c = (colors or {}).get(i, (255, 255, 255))
        d.text((pos[0] + 2, y + 2), t, font=_font(size), fill=(0, 0, 0))
        d.text((pos[0], y), t, font=_font(size), fill=c)
        y += int(size * 1.25)
    return np.asarray(im)


def stills(w=1600, h=1000):
    OUT.mkdir(exist_ok=True)
    shots = []
    view = {"small": ((0.55, 0, 0.12), 1.8), "medium": ((0.8, 0, 0.12), 2.3), "large": ((1.2, 0, 0.2), 3.8)}
    for size in ("small", "medium", "large"):
        t = Teleop(size, "6d", "hold", headless=True)
        from scipy.spatial.transform import Rotation
        from src.task import ClosedLoopTask
        p, R = site_pose(t.model, t.data, "end_effector")
        t.task = ClosedLoopTask(p, Rotation.from_matrix(R), "6d", 2)
        with mujoco.Renderer(big_buffer(t.model), h, w) as r:
            img = render(t, r, camera(*view[size]))
        m_base = t.model.body("base").mass[0] + t.model.body("mount").mass[0]
        img = label(img, [f"ff_{size}: {m_base:.0f} kg servicer + 6.5 kg Kinova Gen3 arm",
                          f"arm / base mass ratio {6.49 / m_base:.3g}"], size=34)
        Image.fromarray(img).save(OUT / f"render_{size}.png")
        shots.append(img)
        t.close()
    # the three side by side at the same scale (servicer size comparison): one renderer per
    # model, identical camera, composited on the non-background pixels
    t = {s: Teleop(s, "6d", "hold", headless=True) for s in ("small", "medium", "large")}
    for s, y in zip(t, (-1.55, -0.45, 1.35)):
        t[s].data.qpos[1] = y
        mujoco.mj_forward(t[s].model, t[s].data)
    cam = camera((0.5, 0.0, 0.05), 3.6, 205, -14)
    frames = []
    for s in t:
        with mujoco.Renderer(big_buffer(t[s].model), h, 2 * w) as r:
            r.update_scene(t[s].data, cam)
            frames.append(r.render())
    bg = frames[0][0, 0].astype(int)
    comp = frames[0].copy()
    for f in frames[1:]:
        mask = np.abs(f.astype(int) - bg).sum(axis=2) > 12
        comp[mask] = f[mask]
    ys, xs = np.nonzero(np.abs(comp.astype(int) - bg).sum(axis=2) > 12)      # crop to the content
    pad = 60
    comp = np.ascontiguousarray(comp[max(ys.min() - pad - 120, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad])
    comp = label(comp, ["Three servicers, one Kinova Gen3 arm: 20 kg / 150 kg / 1500 kg bases",
                        "arm / base mass ratio 0.30 / 0.043 / 0.0043"], size=44)
    Image.fromarray(comp).save(OUT / "render_lineup.png")
    print("saved", [p.name for p in OUT.glob("render_*.png")])


def video(size="small", schemes=("dls", "impedance_rns"), seconds=None, fps=30, w=800, h=720):
    """Split screen, same scripted 6-D reference, one loop."""
    import imageio.v2 as imageio
    from scipy.spatial.transform import Rotation
    from src.schemes import LABELS, make_scheme
    from src.task import ClosedLoopTask
    OUT.mkdir(exist_ok=True)
    teles, refs = [], []
    for s in schemes:
        t = Teleop(size, "6d", "hold", headless=True)
        t.controller = make_scheme(s)
        p, R = site_pose(t.model, t.data, "end_effector")
        t.task = ClosedLoopTask(p, Rotation.from_matrix(R), "6d", 1)
        T, ref = t.task.reference()
        teles.append(t)
        refs.append(ref)
    seconds = seconds or T + 2.0
    lookat = {"small": (0.55, 0, 0.1), "medium": (0.75, 0, 0.05), "large": (1.1, 0, 0)}[size]
    dist = {"small": 2.0, "medium": 2.6, "large": 4.2}[size]
    cam = camera(lookat, dist, 120, -20)
    path = OUT / f"demo_{schemes[0]}_vs_{schemes[1].replace('impedance_rns', 'imprns')}.mp4"
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=8)
    peak = [0.0] * len(teles)
    renderers = [mujoco.Renderer(big_buffer(t.model), h, w) for t in teles]
    n_frames = int(seconds * fps)
    ticks_per_frame = max(1, round(100 / fps))
    for k in range(n_frames):
        tiles = []
        for i, (t, ref, rr) in enumerate(zip(teles, refs, renderers)):
            for _ in range(ticks_per_frame):
                pos, rot, tw = ref(t.data.time)
                t.target_pos, t.target_rot = pos, rot
                t.control_tick(tw)
            err = t.antenna_error_deg()
            peak[i] = max(peak[i], err)
            img = render(t, rr, cam)
            over = err > ANTENNA_LIMIT_DEG
            img = label(img, [LABELS[schemes[i]], f"antenna error {err:5.2f} deg  (limit {ANTENNA_LIMIT_DEG:.0f})",
                              f"peak {peak[i]:5.2f} deg", t.task.status()],
                        colors={1: (255, 80, 80) if over else (255, 230, 120)})
            tiles.append(img)
        frame = np.concatenate([tiles[0], np.full((h, 6, 3), 255, np.uint8), tiles[1]], axis=1)
        if k < fps * 3:
            frame = label(frame, [f"ff_{size} servicer, 6-D ORU task, same scripted operator input"],
                          pos=(20, h - 50), size=28)
        writer.append_data(frame)
        if k == int(n_frames * 0.55):
            Image.fromarray(frame).save(OUT / "demo_frame.png")
    writer.close()
    for r in renderers:
        r.close()
    print("saved", path, f"{n_frames / fps:.1f} s; peak antenna", [round(p, 2) for p in peak])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=["stills", "video", "all"])
    ap.add_argument("--size", default="small")
    a = ap.parse_args()
    if a.what in ("stills", "all"):
        stills()
    if a.what in ("video", "all"):
        video(a.size)


if __name__ == "__main__":
    main()
