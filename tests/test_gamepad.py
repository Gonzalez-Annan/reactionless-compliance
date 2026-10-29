"""PS4 mapping in src/teleop.py, with a fake pad (no hardware needed in CI). Owner: P1/P3."""
import numpy as np
import pygame
import pytest

from src.teleop import FINE_SCALE, Gamepad

UP, LEFT = -1.0, -1.0      # SDL: stick pushed up / left reads -1


class FakePad:
    def __init__(self, axes=None, buttons=()):
        self.axes = list(axes or [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.buttons = set(buttons)

    def get_axis(self, i):
        return self.axes[i]

    def get_button(self, i):
        return int(i in self.buttons)

    def get_instance_id(self):
        return 7

    def rumble(self, low, high, ms):
        self.calls.append((round(low, 3), round(high, 3), ms))
        return True

    def stop_rumble(self):
        self.calls.append("stop")

    calls = None


def pad(layout="classic", **kw):
    g = Gamepad.__new__(Gamepad)          # skip pygame.joystick, no hardware here
    g.layout, g.js, g.trigger_seen = layout, FakePad(**kw), [False, False]
    g.js.calls = []
    return g


def test_no_pad_reads_zero():
    g = Gamepad.__new__(Gamepad)
    g.js, g.layout = None, "game"
    assert not g.read().any()
    assert g.camera() == (0.0, 0.0, 1.0)


# --- classic: world axes, mirrors the keyboard ---------------------------------------

def test_classic_sticks_mirror_the_keyboard():
    assert np.allclose(pad(axes=[0, UP, 0, 0, 0, 0]).read(), [1, 0, 0, 0, 0, 0])    # W
    assert np.allclose(pad(axes=[LEFT, 0, 0, 0, 0, 0]).read(), [0, 1, 0, 0, 0, 0])  # A
    assert np.allclose(pad(axes=[0, 0, 0, 1, 0, 0]).read(), [0, 0, 0, 0, 1, 0])     # mouse down
    assert np.allclose(pad(axes=[0, 0, LEFT, 0, 0, 0]).read(), [0, 0, 0, 0, 0, 1])  # mouse left


def test_classic_triggers_and_shoulders():
    assert np.allclose(pad(axes=[0, 0, 0, 0, -1, 1]).read(), [0, 0, 1, 0, 0, 0])
    assert np.allclose(pad(buttons={10}).read(), [0, 0, 0, 1, 0, 0])
    assert np.allclose(pad(buttons={9}).read(), [0, 0, 0, -1, 0, 0])


def test_classic_has_no_camera():
    assert pad(axes=[0, 0, 1, 1, 0, 0]).camera() == (0.0, 0.0, 1.0)


# --- shared --------------------------------------------------------------------------

@pytest.mark.parametrize("layout", Gamepad.LAYOUTS)
def test_deadband_and_untouched_triggers(layout):
    assert not pad(layout, axes=[0.05, -0.1, 0.08, 0.1, 0.0, 0.0]).read(0).any()


def test_buttons_become_keys():
    g = pad("game")
    ev = lambda b: pygame.event.Event(pygame.JOYBUTTONDOWN, button=b, instance_id=7)
    assert g.handle(ev(0)) == pygame.K_SPACE        # Cross: start
    assert g.handle(ev(1)) == pygame.K_g            # Circle: snap target
    assert g.handle(ev(3)) == pygame.K_v            # Triangle: workspace
    assert g.handle(ev(6)) == pygame.K_p            # Options: pause
    assert g.handle(ev(4)) == pygame.K_BACKSPACE    # Share: reset
    assert g.handle(ev(8)) == pygame.K_HOME         # R3: camera reset


# --- game: camera-relative, camera on the right stick --------------------------------

@pytest.mark.parametrize("az", [0, 90, 135, 250])
def test_game_stick_up_moves_away_from_the_viewer(az):
    u = pad("game", axes=[0, UP, 0, 0, 0, 0]).read(az)
    a = np.radians(az)
    assert np.allclose(u[:3], [np.cos(a), np.sin(a), 0], atol=1e-9) and not u[3:].any()


def test_game_stick_right_moves_right_on_screen():
    u = pad("game", axes=[1, 0, 0, 0, 0, 0]).read(0)     # looking along +x, right is -y
    assert np.allclose(u, [0, -1, 0, 0, 0, 0])


def test_game_triggers_are_world_up_down():
    assert np.allclose(pad("game", axes=[0, 0, 0, 0, -1, 1]).read(77), [0, 0, 1, 0, 0, 0])


def test_game_r1_is_rotate_mode():
    g = pad("game", axes=[0, UP, 0, 0, -1, 1], buttons={10})
    u = g.read(0)                                       # tip down = -rotation about right (-y)
    assert not u[:3].any()
    assert np.allclose(u[3:], [1, 1, 0])                 # +y (= -right) from the stick, +x roll from R2
    assert np.allclose(pad("game", axes=[LEFT, 0, 0, 0, 0, 0], buttons={10}).read(0)[3:], [0, 0, 1])


@pytest.mark.parametrize("fine_button", [9, 2])          # L1 or Square
def test_game_fine_mode(fine_button):
    u = pad("game", axes=[0, UP, 0, 0, 0, 0], buttons={fine_button}).read(0)
    assert u[0] == pytest.approx(FINE_SCALE)


def test_game_camera_on_right_stick_and_dpad_zoom():
    daz, de, zoom = pad("game", axes=[0, 0, 1, UP, 0, 0]).camera()
    assert daz < 0 and de > 0 and zoom == 1.0            # stick right = look right, up = look up
    assert pad("game", buttons={11}).camera()[2] < 1 < pad("game", buttons={12}).camera()[2]
    assert not pad("game", axes=[0, 0, 1, UP, 0, 0]).read(0).any()   # camera stick never moves the arm


def test_southpaw_swaps_the_sticks():
    sp = pad("southpaw", axes=[0, 0, 0, UP, 0, 0])      # right stick up -> move
    assert np.allclose(sp.read(0)[:3], [1, 0, 0])
    assert pad("southpaw", axes=[1, 0, 0, 0, 0, 0]).camera()[0] < 0   # left stick -> camera


# --- haptics -------------------------------------------------------------------------

def test_feel_buzzes_then_stops_once():
    g = pad("game")
    g.feel(0.5, 0.0, now_ms=0)
    g.feel(0.0, 0.3, now_ms=30)
    g.feel(0.0, 0.0, now_ms=60)
    g.feel(0.0, 0.0, now_ms=90)                       # already quiet: no second stop
    assert g.js.calls == [(0.5, 0.0, 120), (0.0, 0.3, 120), "stop"]


def test_event_pulse_wins_over_continuous_feel():
    g = pad("game")
    g.pulse(now_ms=1000)
    g.feel(0.4, 0.4, now_ms=1100)                     # inside the 250 ms pulse: ignored
    g.feel(0.4, 0.4, now_ms=1300)
    assert g.js.calls == [(0.9, 0.9, 250), (0.4, 0.4, 120)]


def test_no_pad_haptics_is_a_no_op():
    g = Gamepad.__new__(Gamepad)
    g.js = None
    g.feel(1, 1)
    g.pulse()


def test_haptic_signals_quiet_at_home_and_ramp_at_a_limit():
    import mujoco
    from src.teleop import HAPTIC_JOINT_ZONE, haptic_signals
    from tests.test_momentum import load
    m, d = load("medium")
    assert haptic_signals(m, d)[:2] == (0.0, 0.0)     # home: no contact, >15 deg from every limit
    highs = []
    for frac in (0.9, 0.5, 0.0):                      # j2 moving toward its upper limit
        d.qpos[8] = m.jnt_range[2, 1] - frac * HAPTIC_JOINT_ZONE
        mujoco.mj_forward(m, d)
        low, high, _, margin, j = haptic_signals(m, d)
        highs.append(high)
    assert j == 1 and margin == pytest.approx(0, abs=1e-9)
    assert 0 < highs[0] < highs[1] < highs[2] == 1.0


def test_haptic_signals_feel_the_arm_hit_the_base():
    import mujoco
    from src.teleop import haptic_signals
    from tests.test_momentum import load
    m, d = load("medium")
    best = (0.0, 0.0)
    for k in range(int(2.0 / m.opt.timestep)):        # full speed backwards: link4 hits the base after ~1.4 s
        d.ctrl[:] = -1.0
        mujoco.mj_step(m, d)
        if k % 40 == 0 and d.ncon:
            low, _, force, _, _ = haptic_signals(m, d)
            best = max(best, (low, force))
    assert best[0] >= 0.35 and best[1] > 0
