"""The scripted operator's give-up test (analysis/study/prestudy.py). On pair 155 its own steering shook the dot
on the fence, 15.0 to 17.5 mm from the ball, and the goal ran into the 90 s limit instead of being given up."""
import sys; sys.path[:0] = ["src", "analysis/study"]
import types
import numpy as np
import prestudy as P

SHAKE = [0.015 + 0.0025 * (i % 8) / 7 for i in range(P.STUCK + 50)]


def gave_up(dists, wait=0):
    """The rule's give-up answer at each step, with the dot at these distances from the ball. Same order as free_drive:
    axes, then this step's dot and label, then park, then the hook. For the first `wait` steps the cage is being
    drawn and the dot sits on the ball, as it did in the trace. Those steps are left out of the answer."""
    axes, fresh = P.operator()
    s, hook = fresh()
    steps = [("updating", 0.0)] * wait + [("current", d) for d in dists]
    s["tgt"], s["text"], s["link"], out = np.zeros(3), steps[0][0], (None, np.array([steps[0][1], 0.0, 0.0])), []
    for text, d in steps:
        axes()
        s["text"], s["link"] = text, (None, np.array([d, 0.0, 0.0]))
        out.append(axes.park())
        hook(None, types.SimpleNamespace(time=0.0), ())
    return out[wait:]


def test_shaking_dot_is_stuck():
    out = gave_up(SHAKE)
    assert not any(out[:P.STUCK]) and out[-1]


def test_dot_pulled_back_by_a_new_cage_is_stuck():
    out = gave_up(SHAKE, wait=300)
    assert not any(out[:P.STUCK]) and out[-1]


def test_closing_dot_is_not_stuck():
    assert not any(gave_up(np.linspace(0.3, 0.002, 3 * P.STUCK)))
