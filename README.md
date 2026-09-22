# Reactionless Compliance

Integrable Redundancy Resolution and Contact-Safety Limits for
Teleoperated Free-Floating Space Manipulators.

MA6223 CA3 Group Project — see `paper/` for the manuscript and
`DECISIONS.md` for the running design-decision log.

## Reproduction
1. `pip install -r requirements.txt` (add this file once dependencies are pinned)
2. `pytest tests/test_momentum.py` — momentum-conservation gate, must pass
   before any other result is trusted.
