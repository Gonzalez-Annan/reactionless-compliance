"""The pad map, on made-up winmm readings: no pad is needed. Run from the repo root."""
import sys; sys.path.insert(0, "src")
import builtins, json, pad

C = 32767
REST = ((C, C, C, C, 0, 0), 0, pad.CENTRE)           # sticks centred, triggers out, nothing pressed, hat centred


def held(axis=None, to=0, buttons=0, hat=pad.CENTRE):
    a = list(REST[0])
    if axis is not None:
        a[axis] = to
    return tuple(a), buttons, hat


def test_state():
    assert pad.state(pad.DS4, *REST) == (0.0,) * 5 + (False,) * 5
    assert pad.state(pad.DS4, *held(0, 65535))[0] == 1.0 and pad.state(pad.DS4, *held(0, 0))[0] == -1.0
    assert pad.state(pad.DS4, *held(1, 0))[1] == 1.0                    # stick up is dot up
    assert pad.state(pad.DS4, *held(0, C + 3000))[0] == 0.0             # inside the deadband
    assert pad.state(pad.DS4, *held(buttons=2))[5:] == (True, False, False, False, False)     # Cross is go
    one = {"away": ["axis", 2, C, 0], "toward": ["axis", 2, C, 65535]}   # both triggers on one axis, as on many pads
    assert pad.state(one, *held(2, 0))[2] == 1.0 and pad.state(one, *held(2, 65535))[2] == -1.0
    hat = {"up": ["hat", 0], "right": ["hat", 9000]}                    # a one-stick pad steering with the D-pad
    assert pad.state(hat, *held(hat=4500))[:2] == (1.0, 1.0)            # a diagonal is both
    assert pad.state(hat, *held(hat=18000))[:2] == (0.0, 0.0)


def test_pick():
    assert pad.pick(REST, held(buttons=4)) == ["button", 2]
    assert pad.pick((REST[0], 1, REST[2]), held(buttons=1 | 8)) == ["button", 3]     # a bit stuck on at rest is not the one
    assert pad.pick(REST, held(hat=9000)) == ["hat", 9000]
    assert pad.pick(REST, held(5, 65535)) == ["axis", 5, 0, 65535]
    assert pad.pick(REST, held(0, C + 6000)) is None and pad.pick(REST, REST) is None
    for h in (held(buttons=4), held(hat=9000), held(1, 0)):             # what is picked reads full when held
        assert pad.level(pad.pick(REST, h), *h) == 1.0


def test_learn(tmp_path, monkeypatch):
    seq = iter([REST, held(0, 65535), REST, held(buttons=2)] + [REST] * (len(pad.STEPS) - 3))
    monkeypatch.setattr(builtins, "input", lambda q="": "")
    pad.learn(lambda: next(seq), tmp_path / "m.json")                 # right on the stick, left skipped, up on a button
    assert json.loads((tmp_path / "m.json").read_text()) == {"right": ["axis", 0, C, 65535], "up": ["button", 1]}
