"""PS4 mapping in src/teleop.py, with a fake pad (no hardware needed in CI). Owner: P1/P3."""
import numpy as np
import pygame

from src.teleop import FINE_SCALE, Gamepad


class FakePad:
    def __init__(self, axes=None, buttons=()):
        self.axes = axes or [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.buttons = set(buttons)

    def get_axis(self, i):
        return self.axes[i]

    def get_button(self, i):
        return int(i in self.buttons)

    def get_instance_id(self):
        return 7


def pad(**kw):
    g = Gamepad.__new__(Gamepad)          # skip pygame.joystick, no hardware here
    g.js, g.trigger_seen = FakePad(**kw), [False, False]
    return g


def test_no_pad_reads_zero():
    g = Gamepad.__new__(Gamepad)
    g.js = None
    assert not g.read().any()


def test_sticks_mirror_the_keyboard():
    # left stick pushed up (LY = -1) -> +x like W; pushed left (LX = -1) -> +y like A
    assert np.allclose(pad(axes=[0, -1, 0, 0, 0, 0]).read(), [1, 0, 0, 0, 0, 0])
    assert np.allclose(pad(axes=[-1, 0, 0, 0, 0, 0]).read(), [0, 1, 0, 0, 0, 0])
    # right stick down -> pitch +, left -> yaw +, like the mouse drag
    assert np.allclose(pad(axes=[0, 0, 0, 1, 0, 0]).read(), [0, 0, 0, 0, 1, 0])
    assert np.allclose(pad(axes=[0, 0, -1, 0, 0, 0]).read(), [0, 0, 0, 0, 0, 1])


def test_deadband():
    assert not pad(axes=[0.05, -0.1, 0.08, 0.1, 0, 0]).read().any()


def test_triggers_and_shoulders():
    g = pad(axes=[0, 0, 0, 0, -1, 1])     # L2 released, R2 fully pressed
    assert np.allclose(g.read(), [0, 0, 1, 0, 0, 0])
    assert np.allclose(pad(buttons={10}).read(), [0, 0, 0, 1, 0, 0])   # R1 roll +
    assert np.allclose(pad(buttons={9}).read(), [0, 0, 0, -1, 0, 0])   # L1 roll -


def test_untouched_triggers_do_not_drift():
    """SDL reports 0 (= half pressed) for a trigger never touched; that must read as released."""
    assert not pad(axes=[0, 0, 0, 0, 0.0, 0.0]).read().any()


def test_square_is_fine_mode():
    assert np.allclose(pad(axes=[0, -1, 0, 0, 0, 0], buttons={2}).read()[0], FINE_SCALE)


def test_buttons_become_keys():
    g = pad()
    ev = lambda b: pygame.event.Event(pygame.JOYBUTTONDOWN, button=b, instance_id=7)
    assert g.handle(ev(0)) == pygame.K_SPACE        # Cross: start
    assert g.handle(ev(1)) == pygame.K_g            # Circle: snap target
    assert g.handle(ev(3)) == pygame.K_v            # Triangle: workspace
    assert g.handle(ev(6)) == pygame.K_p            # Options: pause
    assert g.handle(ev(4)) == pygame.K_BACKSPACE    # Share: reset
