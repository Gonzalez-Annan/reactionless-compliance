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


def pad(layout="classic", **kw):
    g = Gamepad.__new__(Gamepad)          # skip pygame.joystick, no hardware here
    g.layout, g.js, g.trigger_seen = layout, FakePad(**kw), [False, False]
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
