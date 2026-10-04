# Design Decisions Log

Format: `YYYY-MM-DD — decision — who — why`

- 2026-10-04 — Three servicer models (`ff_small/medium/large`) share one arm via `models/common/`; only base size and mass differ (20 / 150 / 1500 kg → arm/base ratio 0.33 / 0.044 / 0.0044) — P1 — the mass-ratio sweep must vary one thing; one arm definition prevents the three files drifting apart.
- 2026-10-04 — Shoulder mounted on the base +x face, so the shoulder offset from the base CoM scales with base size (0.2 / 0.4 / 0.8 m) — P1 — with a fixed offset the large base swallowed the arm and the small base left it floating; a real servicer mounts its arm on its hull.
- 2026-10-04 — Joint actuators are velocity servos (`ctrl` = commanded qdot, kv = 20, ±5 Nm torque limit), replacing torque motors — P1 — all five schemes output qdot. Actuator torques are internal, so momentum is still conserved and the base reacts through the real dynamics, which writing `qvel` directly would bypass. **Provisional: P2/P3 to confirm.**
- 2026-10-04 — Integrator RK4 at 0.25 ms (was implicitfast at 2 ms) — P1 — measured worst-case total-momentum drift over 10 s of random qdot steps: implicitfast/Euler @ 2 ms ~1e-3 (first-order, fails the 1e-6 gate); RK4 @ 0.5 ms, kv 10 ~6e-7 (marginal); RK4 @ 0.25 ms, kv 20 ~1.2e-7 (10× margin). Runs ~9× faster than real time.
- 2026-10-04 — Antenna site on the base top face; its z-axis is the boresight used for the ±5° pointing limit (R1) and the demo-video cone — P1.
