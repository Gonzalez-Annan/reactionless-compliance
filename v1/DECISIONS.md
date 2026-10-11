# Decisions

| # | Decision | Why |
|---|---|---|
| D1 | Simulation only, MuJoCo, gravity off, free-joint base + 7-joint arm. | Course scope. |
| D2 | Three servicers: 30 / 300 / 3000 kg base, same 12.5 kg arm. | Base-to-arm mass ratio is the variable that decides whether reaction matters. |
| D3 | Reactionless condition is `W q̇ = 0`, with `W` = angular rows of `-H_b⁻¹ H_bm`. **Changes the plan**, which used the null space of `H_bm,ang`. | `H_bm,ang q̇ = 0` only zeroes base rotation when the base frame sits at the system centre of mass. `W` is the actual base angular rate map; `test_jstar_predicts_hand_velocity` checks it against the simulator. |
| D4 | Rest posture `[-0.26, 0.88, -0.17, 1.5, 0.57, 0.83, -0.52]` rad. | Best of 40 random postures by smallest singular value of J* along the path. The first posture crossed a singularity. |
| D5 | Gains: DLS λ 0.02, null-space KP 1, manipulability KG 5, WLS γ 0.01, attitude KA 5, default β 10. Operator KX = KO = 4. | Hand-tuned on the medium servicer; λ 0.05 gave 30 mm tracking error. Not optimised per scheme, which is a limitation. |
| D6 | Joint velocity servos (kv 150), armature 0.5, dt 0.5 ms, RK4, control at 100 Hz. | With armature 0.1 and dt 2 ms the momentum residual was 1.4e-4 (integrator error from the stiff servo). Now below 1e-7. |
| D7 | Schemes 4 and 5 feed base attitude error back: `W q̇ = -KA·att`. | Open-loop `W q̇ = 0` lets servo lag accumulate as drift. |
| D8 | Joint rate limit 2 rad/s by scaling the whole vector, not clipping per joint. | Per-joint clipping changes the direction of q̇ and broke the reactionless constraint (43° drift on the small servicer). |
| D9 | Path: 0.10 m circle in the world y–z plane, 15 s per loop, smoothstep timing (peak speed 1.5× mean), 0.25 rad tilt in 6-D mode. | Starts and ends at rest, so joint-return and final-drift are measured at standstill. |
| D10 | Scripted operator (feed-forward + P) for all reported numbers. | Repeatable. The human study replaces it through the `operator` hook. |
| D11 | Safety map uses base-only effective mass (joints locked) and the chest limits F 280 N, k 25 kN/m, m_h 40 kg. | Locked joints is the conservative case. **The three body values are recalled from ISO/TS 15066 Annex A and must be checked against the standard (P4).** |
| D13 | Joint limits ±2.9 rad (roll joints) and ±2.0 rad (pitch joints), enforced twice: a controller guard that locks a joint 0.05 rad before its limit and re-solves with the rest, and hard stops in the physics. When joints lock, the hand task outranks the reactionless term (weight 100), so the base takes the leftover reaction. | Requested by the team. Limit values are generic (close to a Panda or iiwa), not from a specific arm: replace them if you pick one. The lock is abrupt, with no slow-down zone. |
| D12 | R4 gate: on the medium servicer the 3-D reactionless schemes must return the base to within 0.02° and keep peak drift below a third of DLS. | The plan assumed exactly zero drift. Measured peak is 0.28° against 1.15° for DLS. |
| D14 | Free-drive teleop is set-and-go: the operator moves a goal dot anywhere in 3-D and presses Enter; the move is rehearsed on a copy of the simulation and only executed if the hand arrives within 5 mm with the base drifting under 0.5°. Base attitude outranks the hand here (task weight 0.1), the reverse of D13. | Operator intent: pick a goal, base stays level. Goals outside the reactionless workspace are refused (red dot) instead of swinging the base. The benchmark and the study session keep D13 (hand first, velocity keys). The viewer draws the work envelope as 26 blue rays from the hand: each is a rehearsed hard pull in that direction, ending where the hand stalls or the base passes 0.5°; it is recomputed for the arrival pose after every accepted goal (about 17 s, in the background). Medium servicer from rest: 0.05 m (forward and down) to 0.55 m (up), most directions 0.25 to 0.50 m. The pull near the goal is stiff (20 1/s) so that accepted goals match the drawn edge; with the operator gain of 4 the hand stalled 22 mm short of a 0.10 m goal. Small servicer: about 0.05 m forward. So set-and-go defaults to the medium servicer. |

| D15 | Teleop: hand first, base within 2 degrees, envelope drawn as a cage

Supersedes the base-first weighting and the 0.5 degree limit of D14. Measured on the medium servicer at rest:
the hand Jacobian restricted to the exactly reactionless joint motions has singular values 0.51, 0.07, 0.04,
and the smallest falls to 0.0003 after 0.15 m forward. So with the base held exactly level the hand has one
usable direction, and no weighting fixes that (task weights 0.1 to 100 all give 0.14 to 0.35 m). Reach against
the base drift allowed, straight pulls along the six axes: 0.5 deg 0.14-0.34 m, 1 deg 0.25-0.45 m,
2 deg 0.32-0.85 m, no limit 0.32-0.96 m (then joint limits). Chosen: task weight 2 (hand first) and a 2 degree
limit, which gives nearly the whole kinematic reach; the base does NOT re-level at the goal (measured, D17); it comes back when the hand does.
The envelope is drawn as a cage through the tips of 26 probe pulls from the current pose and redone after
each move; the goal dot is green or red from the same rehearsal, about half a second behind.
Controls are camera-relative and laid out for a PS4 pad, mirrored on the keyboard (table in src/teleop.py).
The PS4 path is still untested.

| D16 | The level-base limit is the arm's geometry, kept as a result

analysis/singularity_map.py scores 6000 random poses of the medium servicer by how freely the hand moves with
the base held level (smallest singular value of the hand Jacobian on the base-level joint motions; the plain
Jacobian is 0.2 to 0.85). Median 0.01, 90th percentile 0.05, best after hill-climbing 0.10; the rest pose
scores 0.043. At rest the three servicer sizes score 0.037, 0.043, 0.048, so base mass does not change the exactly-level score (it does change reach under a drift limit, D17).
The best poses do not give more reach than the rest pose (0.14-0.68 m per axis at 0.5 deg against 0.14-0.34 m,
and less at 2 deg). So no starting pose fixes it: a level base costs 5 to 10 times of the hand's mobility
everywhere. Decision: keep the robot and the rest pose, report this as a trade-off (D15 table: drift allowed
against reach). Not done: changing mount point or link lengths, which would mean re-running the sweep.

| D17 | Drift is borrowed, not lost; base inertia sets the price

analysis/level_base.py, medium servicer unless said. Measured:
- Base turn per 10 cm of hand travel at rest, worst direction (W J+): 12.4 deg small, 0.62 medium, 0.021 large.
  Mean axis reach at 0.5 / 2 deg: small 0.09 / 0.20 m, medium 0.23 / 0.52 m, large 0.59 / 0.59 m (joint limits).
- With the hand held still the spare joints keep 10-20% of their authority over the base (median of 2000
  poses), near zero about one axis. The base does not re-level at the goal: 0.24 -> 0.25 deg, 0.87 -> 0.90,
  0.46 -> 0.47 over an 8 s hold. Corrects D15.
- The base comes home with the hand: 0.00 deg after 0.3 m out and straight back; 0.02, 0.03, 0.04 deg after
  1, 2, 3 laps of a 0.25 m square. So under scheme 5 base tilt is close to a function of hand position.
- The hand's free direction leans toward the line through the system centre of mass (median 30 deg off it,
  60 would be chance) and is 3.4 times freer than the next. A lean, not a rule.
Believed, not measured: because tilt is a function of position, ONE fixed envelope (nested 0.5 / 1 / 2 deg
shells) could replace the cage that is recomputed after every move. Not built.

limit, which gives nearly the whole kinematic reach; the base does NOT re-level at the goal (measured, D18); it comes back when the hand does.

analysis/sizing.py (fig6_sizing), nine base masses 30-3000 kg, same 12.5 kg arm. Measured:
- Full 0.59 m straight-line reach needs ~950 kg at a 0.5 deg limit, ~530 kg at 2 deg; 95 kg keeps 0.13-0.22 m.
  (53 kg at 2 deg reads 0.16 m, below the 30 kg figure of 0.20: stall detection noise, not a trend.)
- Wheels instead of drift: momentum h = K v, K = 6.2-9.2 kg m at rest across ALL masses, median 8.2 and
  90th pct 13.3 over 2000 poses at 300 kg. At 0.1 m/s hand speed: 0.6-0.9 N m s. It is the arm's momentum,
  so a bigger bus does not reduce it. Not checked against a wheel catalogue; rigid bodies, min-norm joints.
Teleop: --delay S (command reaches the robot S seconds late), --direct (rate control, no dot/envelope/rehearsal),
--pilot ID (4 blocks: direct/set-and-go x 0/2 s, five goals, hold 0.5 s within 2 cm; data/.../delay_ID.csv).
A scripted operator completes all four blocks; that run is a tool check, NOT participant data.

## D19 Wheels: torque binds, not momentum. Envelope is path-independent (measured)
- Catalogue wheels (from a web search summary of vendor datasheets, not the PDFs themselves: BCT RWP500 0.5 N m s / 25 mN m, BCT RW1 1.0 / 60, Rocket Lab RW-12 12 / 200). With K = 8 kg m: speed limit h/K = 0.06 / 0.12 / 1.5 m/s, hand acceleration tau/K = 0.003 / 0.008 / 0.025 m/s^2, i.e. 32 / 13 / 4 s to reach 0.1 m/s.
- ponytail: one wheel per axis, starting unloaded, K at one pose; a real sizing needs the wheel cluster geometry.
- Fixed envelope test (scratch fixed.py, 300 kg, 2 deg): 10 random directions x 3 goals at 0.6/0.9/1.15 of the edge, judged from rest and after a 0.15 m sideways detour: 30 of 30 agree. Edge distance by direction 0.32 to 1.01 m. Supports drawing ONE fixed envelope instead of a per-move cage. Not tested: longer detours, 30 kg base. Not built.
- Paper headline now reads 'inside the level-base envelope'.

## D20 Fixed nested envelope built; 26 probe lines were too few (measured)
- Teleop now draws ONE envelope from rest, three shells (base tilt under 0.5 / 1 / 2 deg), taken from a single pass per direction. On Enter the console says the tilt band the goal will cost. The rehearsal (check) still decides accept/refuse.
- My D19 reading 'fixed region of space' was too strong: that test only used goals ON the probed lines. Goals between lines, medium base, scratch t14-t16: 26-line cage, six hand positions, goals 0.3-1.3 of the edge: 34 of 42 agree, 6 of 12 near the edge; 26 lines from rest only, goals 0.7-1.3: 31 of 42; 98 lines from rest only: 37 of 42 (20 of 24 within 15% of the edge), 4 of the 5 misses are the cage saying 'out' for a goal that is reachable. So the error was mostly the straight join between too few lines, not path dependence.
- Tilt band predicted by the shells vs measured at arrival: 26 lines, six positions 16 of 25, every miss an over-estimate; 98 lines from rest 16 of 18.
- Not tested: 98 lines from hand positions other than rest; small and large base; a human reading the three cages.
- Pilot goals sit at 0.30-0.41 of the 2 deg edge: none can be refused.

## D21 Framing: version 2 of the ETS-VII reactionless experiment; 6 against 7 joints measured, and it does not support the joint-count claim
- Application chosen with the user: grapple reach of a small servicer in free drift; heritage = ETS-VII zero reaction maneuver (Yoshida, Hashizume, Abiko, ICRA 2001; 6-joint arm, 2.5 t). Paper NOT read, only its title and a search summary.
- Measured (scratch v2a.py, scheme 5, position only, 26 probe lines, one joint locked at rest), volume inside the 2 deg shell in m3, 7 joints / lock j3 / lock j5 / lock j7: large 1.11 / 1.24 / 1.80 / 1.10; medium 0.69 / 0.44 / 0.82 / 0.69; small 0.016 / 0.012 / 0.007 / 0.015. Worst lock is j4 (elbow): 0.16, 0.17, 0.003.
- So: a 6-joint arm already has a level-base VOLUME, not narrow paths (my claim to the user was wrong). Joint 7 (wrist roll) adds nothing to position reach. Locking j5 reaches FURTHER than 7 free joints on large and medium: the envelope is limited by the local controller's joint path, not by the arm.
- Rest pull is not the cause (scratch v2b.py): gain 0, 0.2, 1, 5 give mean reach 0.59-0.60 m medium, 0.68-0.69 m large.
- Not done: why a locked j5 helps; an envelope for an aligned grapple (position + approach axis = 5 task + 3 base = 8 > 7 joints); contact.

## D22 Literature check (search summaries only, no paper read): the dynamics is prior art, the operator side is where we can be new
- Path-independent / path-dependent workspace and dynamic singularities: Papadopoulos and Dubowsky, ASME JDSMC 1993 (nereus.mech.ntua.gr/Documents/pdf_ps/asme93.pdf). Our D19/D20/D21 path results are instances of this.
- Fixed-attitude-restricted Jacobian, and a 7-DOF free-flying robot under Cartesian velocity telecontrol: Nenchev and Yoshida (sice.jp/e-trans/papers/E1-16.pdf). This is our hand-plus-level-base stacking.
- Zero Reaction Workspace computed for different arm/base mass ratios: IAC-10 D3.3 (iafastro.directory/iac/archive/browse/IAC-10/D3/3/8979/). Prior art for the sizing chart; ours differs only in grading by allowed tilt.
- Contact: reaction null space with impact (Yoshida/Nenchev, IEEE TRA 1999), impedance for free-floating capture (Yoshida, ICRA 1997), null-reaction capture with a redundant arm (Cocuzza, IAC 2016).
- Operator displays under multi-second delay exist for ground robots (ISCTE study; a path/envelope visualisation paper at uhasselt). One search found none that shows a free-floating arm's base-reaction envelope to an operator: a hypothesis, not a finding.
- Framework from here: one claim (the display helps the operator, most on small buses); dynamics cited not claimed; three experiments only (display on/off with edge goals, display error rate, one grapple touch).

## D23 Kill-checks on problem A: gap narrowed, re-pose lever measured (2026-10-07)
- Prior work found (abstract only, full text not read; TUM mediatum page is behind bot protection): Pietras, Fleischner, Walter (TU Munich), IAC-10 B6.2.2, real-time evaluation of attitude constraints during telepresent space-robot operations; unplanned base attitude change breaks antenna pointing; operator-in-the-loop tests of 'different operator support concepts'. My D22 'no operator aid found' is withdrawn. What the concepts were is unknown: novelty of comparing help forms is NOT confirmed until someone reads it.
- Other prior art: 'best configuration' for capture (IAC-16 A6.5), minimum base disturbance pre-capture (Robotica); zero reaction workspace vs mass ratio is Cocuzza IAC-10 but 3-joint planar.
- Application: one docking patent (US 12097978) assumes approach precision +-25 cm; our level-reach at 100 kg is 0.13-0.22 m. Capture is normally supervised autonomy (DEOS, patents): the operator acts on anomalies, so frame the task as takeover in a marginal capture.
- Re-pose measured (scratch v3.py, scheme 5, 26 lines, 12 random alternative postures at the same hand position plus rest). No posture is better on average (mean reach within about 10% of rest on every bus). Choosing the posture per direction is what pays: best-of-13 / rest, median over directions, at 0.5 / 1 / 2 deg: small 2.2 / 2.0 / 1.5 (22, 23, 19 of 26 directions gain over 20%), medium 1.5 / 1.3 / 1.1 (22, 18, 3 of 26), large 1.0 (2 of 26, stall-limited). So lever 1 is real on light buses and tight tilt limits, and needs the goal direction known first.
- Caveats on re-pose: postures were set by teleporting the arm on a held base, the cost of getting there is not measured. With hand and base attitude both held only 1 of 7 joint freedoms is left (my '4-D family' ignored the base), so most re-poses must be done before free drift while attitude control is on. 12 samples is a lower bound on the gain, not an optimum.

## D24 Problem statement = the intersection; decision study redesigned around the rehearsal (2026-10-07)
- Ten searches (titles and abstracts only): reach displays for people exist but for fixed-base arms (ReachVox, capability calibration, AR limits); binary vs graded 'likelihood' alarm vs raw information at 70-90% reliability is established human factors; DLR 'Null-Space Wall' gives limits by haptics during the move; TUM 2010 (Pietras, Rems, Walter; I had the second author wrong) is the only operator-support paper on base attitude and has no visible follow-up. Claimed gap = the intersection: presenting an imperfect, tilt-dependent reach envelope to an operator who commits before the move. Still a hypothesis until the TUM paper is read.
- Why this aid is not a generic alarm: (1) its reliability is not one number, it falls from near 100% deep inside to about 5 in 6 at the edge, so margin is the aid's own confidence; (2) the operator has no independent evidence, people cannot see reaction dynamics, so the only way to catch an aid error is the rehearsal; (3) the rehearsal answers one goal, the map answers where to park.
- So the aid's job is triage: tell the operator when a rehearsal is worth its time. Study (teleop.py --decide): 4 kinds of help (none, binary, graded by tilt band, map), goals from 0.5 to 1.4 of the edge, actions go / re-park / rehearse (8 s). Measures: correct decisions, time, and rehearsal rate against depth. Prediction: graded and map concentrate rehearsals at the edge, binary does not.
- Planted aid errors dropped (the 1-in-4 pool of the first version): errors are now whatever the envelope gets wrong. The 8 s rehearsal cost is a design choice, not a measured operational figure.
- Pool as built (medium, 60 goals): aid wrong 0 of 24 inside 0.95 of the edge, 4 of 12 at 0.95-1.05, 1 of 12 at 1.05-1.2, 0 of 12 beyond; all 5 errors are a no-go for a reachable goal, no false go. So a 'go' can be trusted and only a no-go near the edge is worth a rehearsal. Grading by tilt band would add nothing there, so the graded verdict is graded by margin ('close to the edge' within 15%). 60 goals is one pool: the zero false-go count is not a guarantee.

## D25 A wrong go costs more; the envelope measured as a detector (2026-10-07)
- Scoring in teleop.py --decide: a go that works 0, re-park -5, rehearsal -1 (plus the 8 s), a go that fails -20. The outcome is shown only after a go. The numbers are a design choice, told to the participant, not mission costs. New column: points.
- Measured (scratchpad v4.py, medium bus, 300 goals at 0.75 to 1.25 of the edge, truth by rehearsal, 166 reachable). 98 lines, edge as drawn: 10 unreachable goals called go, 27 reachable called no-go, 263 of 300 right. 26 lines: 13 and 61, 226 right. More lines cut both errors; moving the edge only trades one for the other.
- CORRECTION to D24: the aid's errors are NOT one-sided. The 60-goal pool happened to hold no false go; 300 goals hold 10, at 0.78, 0.85, 0.92, 0.92, 0.95, 0.97, 0.98, 0.99, 0.99, 1.00 of the edge. Pulling the edge in to 0.85 still leaves 2. No setting of the edge makes the aid safe on its own.
- By zone: 0.75 to 0.90, 87 of 89 reachable; 0.90 to 1.10, 79 of 127; beyond 1.10, 0 of 84. With the points above a rehearsal pays only in the middle zone, on both sides of the edge (go side: 8 of 60 fail, so going blind averages -2.7 against -1.7 with a rehearsal; no-go side: 27 of 67 reachable, -4.0 against -5). With equal costs it would pay only on the no-go side. Trust the aid everywhere: -335 avoidable points per 300 goals. Rehearse everything: -300. Rehearse only the middle zone: -167.
- So 'close to the edge' in the graded aid is now within 10% (was 15%). The best policy here is a rule a program could follow; the person's job in a real mission is the costs outside the model. One bus size, one seed, no contact, and the rehearsal is itself a model.

## D26 Direction picked: the robust envelope (2026-10-08)
- Why the D24 frame was weak: the aid's errors there are mostly our own coarse sampling (26 to 98 lines cut both kinds), so 'use more lines' answers it. What no sampling removes: the map and the rehearsal share one model, so if the bus mass is off both are wrong together.
- Pick (mine, the user said to choose): draw the envelope over a spread of plausible bus masses. Core = reachable for every mass (go). Outside the shell = unreachable for every mass (re-park). Between = depends on what is not known; a rehearsal cannot settle it, only re-parking into the core or finding out the mass can. The core is also the box the servicer must park in, so the parking question and the crossover chart come from the same object.
- Kill condition: if the nominal map and rehearsal stay right for nearly all goals when the real bus is 10% off, the band is too thin to matter.
- RETRACTED: the first two runs (scratchpad v5.py, first v6.py) changed the mass with mj_setConst, which left the arm at the wrong posture; reach read 0.3 m against a true 1.4 m. The 'edge moves about 7%' figure is void. The harness now uses mj_forward and must reproduce the cached map within 1 mm before it measures anything.

## D27 Robust envelope is secondary; the problem lives in a band of bus sizes (2026-10-08)
Measured (scratchpad v6.py, v7.py; simulation only, no people).
- Bus mass x0.9 / x1.0 / x1.1, 120 goals near the edge: nominal map false go 6 / 4 / 4, false no-go 3 / 7 / 12. Core (reachable for every mass) false go 4 of 59 in all three: those are the coarse-map errors of D25, not mass. Rehearsal on the nominal model disagrees with reality on 6 and 5 of 120 goals. 3 of 98 directions change 2 deg reach by over 30%.
- Verdict: D26's kill condition is mostly met. Mass at +-10% moves the edge less than the 10% band D25 already rehearses. Robust envelope is a secondary result, not the headline.
- Parking box, shortest / median reach over 98 directions, and directions under 0.25 m: small 2 deg 0.035 / 0.133 m, 84; medium 0.5 deg 0.110 / 0.278, 44; medium 1 deg 0.194 / 0.386, 16; medium 2 deg 0.286 / 0.577, 0; large 0.291 / 0.618, 0 at all three limits (tilt never binds, arm length does).
- Reading: large bus needs no aid, small bus has no safe parking spot at 0.25 m error, medium is where the decision is real and the tilt limit chosen decides whether parking accuracy matters.
- Believed, not measured: the 0.25 m parking error (no source yet); band edges between 30 and 3000 kg (three points only).
- Next: crossover chart, bus mass against tilt limit, more masses. Headline = map + band where it matters + the D25 three-zone rule.
- PRD and plan doc: https://claude.ai/code/artifact/9e5e5bfe-47e2-4d95-9148-cddb0123f205

## D28 (2026-10-08): the go / no-go decision is not a safety problem; the fence and the band are the result

Measured (simulation only, scripts v8.py and v9.py in the session scratchpad, to be copied into the repo before the paper cites them):

- Runtime tilt fence, 45 goals a rehearsal calls unreachable, commanded anyway, medium bus, trip at 2.0 deg:
  tilt trip 22 (peak 2.011 deg at most, 0.039 deg left at the end, 14.2 s median); stall 17 (peak 1.959, left 0.005 median
  and 1.102 worst, 15.9 s); crept for 60 s without stalling or tripping 5 (left up to 1.836 deg); arrived 1 (peak 1.272).
- Band (medium model, bus mass and inertia scaled together, 98 directions; directions where tilt binds / under 0.25 m reach, at 0.5, 1, 2 deg):
  30 kg 98/89 96/82 90/57; 60 kg 98/83 97/63 88/42; 99 kg 98/69 98/53 90/31; 150 kg 97/61 97/43 88/13;
  300 kg 96/44 90/16 65/0; 600 kg 89/19 72/0 17/0; 990 kg 79/0 24/0 0/0; 3000 kg 8/0 0/0 0/0.

Verdict:

- FAILED: "a wrong go is dangerous" (the -20 in the D25 scoring). With the fence a wrong go costs about 15 s and no attitude.
  The D25 scoring is void; any study must be scored in seconds.
- WEAK: "the operator needs a rehearsal". A rehearsal is 8 s, a fenced wrong go about 15 s.
- HOLDS: the map as a guide (D25 numbers), and the problem existing only in a band of bus sizes that moves up as the tilt limit tightens.
- The rehearsal used as ground truth is itself wrong on 1 of 45.

Decision: the paper is about the geofence, the feedback it gives, and the band. The decision-aid study is demoted to a feedback
study (no cue / colour cue / colour plus hard fence), scored in time.

Built: teleop.py has FENCE = (0.9, 1.1), zone(), hard clamp of the dot at 1.1, yellow dot in the caution zone, rumble hook
(untested, no pygame), and a runtime tilt fence that sends the hand back to where the move started. Checked: zone rule and clamp
on a synthetic cage. Not checked: nobody has watched it run.

Open: (1) creeping moves need a time limit; not added because the right limit is unmeasured (a slow legitimate move near the edge
took 21 s). (2) coming home does not level the base after a stall (1.1 deg left). (3) inertia scaling in the band is linear in
mass, cruder than a real bus. (4) believed, not checked: real servicers hold attitude with wheels, so tilt should be re-read as
wheel momentum. (5) novelty still rests on the unread TUM paper; fences (virtual fixtures, envelope protection) are old ideas.

Addendum to D28, same day: open item (1) closed. analysis/envelope/fence.py: 42 of 42 reachable moves from rest arrive, median
4.6 s, slowest 10.2 s. With a time limit of 1.5 x that (15.3 s) the same 45 wrong goes end: tilt trip 22, stall 12, time limit 11;
all home, worst 22 s, peak tilt 2.011 deg, tilt left 0.116 deg at most except the one stall case (1.102 deg, still open).
teleop.py uses T_MOVE = 30 s (room for a move across the cage; not measured for such moves). Band chart: analysis/envelope/band.py
-> paper/figs/fig_band. Paper rewritten around D28 (old draft kept as paper/main_v1_schemes.tex); NOT compiled, no LaTeX here.
Study mode for the feedback conditions NOT built: the teleop has not been watched running yet, and that comes first.

## D29: the arm passed through the bus; envelope rebuilt with a clearance limit (2026-10-08)

- The model has no contacts and nothing checked the arm against the bus. Added `clear()` (2 cm, links 2-7 against the bus box) to the probes, the rehearsal and the runtime fence. No arm-on-arm check.
- Rest pose: hand 1 cm above the top face, arm 6 cm from the bus. With the limit, 39 / 43 / 62 of 98 directions are cut short at 0.5 / 1 / 2 deg; median reach 28 -> 14, 39 -> 22, 58 -> 31 cm. Old cache kept as `data/decide4_medium_nocollision.npz`.
- STALE until rerun: D25 detector table, v6 mass, v8 band, v9 and fence.py outcomes, fig_band, every table in paper/main.tex. All were measured with an arm free to pass through the bus.
- First-order physics does not predict the envelope (analysis/envelope/basis.py, 36 uncut directions, rest pose): rank correlation of measured reach with reactionless speed -0.07 / -0.18 / -0.51, with 1 / tilt-per-metre -0.29 / -0.41 / -0.76; first-order tilt at the measured 0.5 deg edge is 0.52-2.56 deg (median 1.14). The envelope is a property of start pose + controller + straight path, not of the local coupling.
- Feedback study mode built (`--feedback`, D28 follow-up): 3 cues, time-scored, 30 s give-up penalty (design choice), T_REST 15.3 s. Slice view (Insert) and `fig_slices`. Known flaw: goals in directions the bus cuts sit 2-6 cm from the hand, inside the 2 cm arrival radius.

## D30: ready pose chosen by measurement; the envelope edge is a dynamic singularity (2026-10-08)

- Ready pose: Q_REST = [1.59, -0.3, 0.07, -1.52, 0.07, 1.42, -2.41], the largest 2 deg tilt-bounded volume of 160 random collision-free poses with every direction past 15 cm (analysis/envelope/ready.py). Hand 56 cm above the bus. The best six are within 41-45 cm median reach, so the pick is not delicate.
- Envelope on it: 2 deg reach 24-75 cm, median 45 (old pose with the bus limit: 4-85, median 31); 0.5 deg median 26 (was 14). The bus ends 9 of 98 probes (was 67). 2 deg volume 0.585 m^3 = 29 percent of the 2.04 m^3 kinematic workspace.
- Basis (analysis/envelope/causes.py): the controller solves hand velocity and zero base rotation together, S = [J; W], 6 x 7. Where the base crosses 0.5 deg, the 6th singular value of S over the free joints is under 10 percent of its rest value, or two or more joints are at a limit, on 94 of 97 directions (1 deg: 92 of 93; 2 deg: 63 of 63). Median ratio at the crossing is under 0.01. Joint limits alone explain 12.
- Limits of that claim: tilt begins earlier and gradually (first 0.05 deg at 27 percent of the 2 deg reach, sigma6 ratio 0.24), so the wall is not thin. On the old pose 7 of 39 directions dipped under 10 percent without reaching 0.5 deg (the bus stopped them first). One servicer size, one controller, straight moves from one pose.
- Old caches kept: data/decide4_medium_oldpose.npz, data/decide4_medium_nocollision.npz. Everything listed STALE in D29 is still stale, now for the new pose too. Also stale: the sweep in data/sweep.csv (run_trial starts from Q_REST).

## D31: live singularity margin on the hand (2026-10-08)

- `teleop.sig6()` / its value at the ready pose, shown in free drive as a ball on the hand (green, yellow under 0.24, red under 0.05) and a number top left. Not shown in the feedback study, so its three conditions are unchanged.
- Check from other poses (analysis/envelope/margin.py): 6 starts 15 cm from the ready pose, 84 straight probes, 79 tilt the base 0.5 deg further. At that moment the margin is yellow or red on 79 of 79, red on 71. Yellow comes a median 26 cm of hand travel earlier, red 10 cm. From the ready pose: 13 of 13 red, 12 and 6 cm.
- Weak side: all 6 probes that never tilted 0.5 deg further also went red. Six is too few to give a false-alarm rate; red means the reactionless solution is gone, not that the base has tilted yet.
- Not seen on screen by me (the MuJoCo window cannot be captured here).

## D32: scripted loop moved to the x-z plane (2026-10-10)

- After D30 two gates failed (`test_3d_reactionless`). Cause: the scripted circle went out 20 cm along +y, and from the new ready pose that direction runs into the joint limits. Scheme 5 had joint 6 on its limit 51 percent of the run and joint 3 for 20 percent, so the base got 1.18 deg against 1.56 for DLS.
- Measured the same circle in all 12 axis-aligned placements (medium, 3-D, 1 loop, scheme 5 vs DLS). Any loop that goes out along y: DLS/scheme 5 ratio 1.0 to 1.6. Loops in the x-z plane never touch a limit: ratio 5.7 (out +x), 6.6 (out -x), 9.3 (out -z), 23.5 (out +z).
- Picked out +x in the x-z plane: it is the old loop with y swapped for x, not the best case. Scheme 5: peak 0.182 deg vs DLS 1.045, back to 0.006 deg, limits never hit, 0.18 mm rms hand error.
- Scheme 4 on the same loop: peak 0.179 deg, but it ends 0.066 deg off. It has no posture pull, so joint 5 wanders 3 rad in the null space and parks on its limit for 32 percent of the run. With limits off it returns to 0.000. So the gate now asks scheme 4 for under 0.1 deg at the end and scheme 5 for under 0.02 as before. This is a real weakness of plain RNS, worth a line in the paper.
- 14 of 14 gates pass. Still stale: data/sweep.csv and the paper tables (old pose and old loop).

## D33: sweep and paper tables redone on the new ready pose and loop (2026-10-10)

Re-ran `analysis/sweep.py` (84 runs, all stable, no limit violation), `analysis/figures.py`, `analysis/envelope/v4.py`,
`fence.py` and `analysis/level_base.py`, then put the numbers into both tex files. This closes the stale list in D32.

- Position-only, 300 kg: scheme 5 peaks at 0.20 deg against 1.22 for DLS (sixfold, was fourfold). Tracking 0.16 mm.
- 30 kg, position-only: schemes 4 and 5 sit on a joint limit 24 to 44% of the run, hand 47 to 50 mm behind.
- 6-D, 30 kg: schemes 1 to 3 drift 45 to 48 deg. Getting inside 5 deg needs beta >= 1 and costs 81 to 84 mm.
- Map: 162 of 300 targets reachable. 98 lines at depth < 1: 8 false go, 21 false no-go, 271 right.
- Fence: 46 refused commands. 33 tilt trips (peak 2.020 deg), 10 stalls, 1 time limit, 2 arrived. Time limit is now
  13.1 s (1.5 x the slowest of 41 reachable moves, 8.8 s), so `T_REST` in teleop.py went from 15.3 to 13.1.
- Rest-pose singular values of the hand Jacobian with the base held level: 0.36, 0.10, 0.08 (were 0.51, 0.07, 0.04).

One claim did not survive. We wrote that the tilt is borrowed and comes back with the hand (0.00 deg after 0.3 m out
and back). From the new ready pose 0.30 deg is left and the joints end 1.24 rad from the start. Square laps leave
0.67, 0.11, 0.14 deg. Both papers now say this.

The fence also showed the rehearsal is wrong about 2 of 46 targets: it refused them and the arm reached them
(base at 1.2 and 1.9 deg).

Not re-run, marked with a red TODO in the tex: `v6.py` (wrong mass), `v8.py` and `band.py` (fig_band), `sizing.py`
(fig6), the 0.0003 singularity number, the 30 of 30 / 37 of 42 / 16 of 18 counts, the 8 s rehearsal time.

## D34 The rest of the analyses redone on the new ready pose (2026-10-10)

All measured, 300 kg model unless said.
- Band (`v8.py`, `band.py`, fig_band). Directions where tilt binds / under 0.25 m reach, at 0.5, 1, 2 deg:
  30 kg 96,95,88 / 90,84,53. 150 kg 97,96,81 / 62,44,24. 300 kg 97,93,63 / 45,25,4. 600 kg 96,72,33 / 26,2,2.
  990 kg 80,48,1 / 3,2,2. 3000 kg 15,0,0 / 0,0,0. Same story as D28: the band is about 150 to 600 kg at 2 deg
  and moves up when the limit tightens.
- Wrong mass (`v6.py`, 120 goals, true mass x0.9 / x1 / x1.1): nominal map false go 4, 1, 1, false no-go 4, 6, 10.
  1 of 98 directions changes reach over 30%. Rehearsing on the wrong mass is wrong on 5 and 4 of 120. Still small.
- Sizing (`sizing.py`, fig6): full 0.61 m reach needs about 1700 kg at 0.5 deg and about 950 kg at 2 deg, which is
  a lot more bus than the old pose said (950 and 530). 100 kg keeps 0.17 to 0.23 m. K is 8 to 11 kg m across the
  nine masses, so 0.8 to 1.1 N m s at 0.1 m/s.
- Singularity: smallest restricted singular value is 0.077 at rest, 0.007 after 0.15 m in +x or -x, under 0.0001
  after 0.14 m in -y. +y and z stay near 0.06 to 0.08. The 6000 random postures are not tied to the rest pose.
- 8 s rehearsal time is `CHECK_S` in teleop.py, a number we set. The paper says so now.
- Start-independence re-tested with a new script, `analysis/envelope/start.py`: 12 of the 98 directions, goals at
  0.6, 0.9, 1.15 of the 2 deg edge, rehearsed from rest and after a 0.15 m detour to +y (leaves the base 0.80 deg off)
  and to +z (0.00 deg). Same answer on 72 of 72. From rest: 12, 12, 0 of 12 reachable. Limits of this test: goals sit
  ON probed lines, none between 0.9 and 1.15 where the map is unsure, and only two detours of 0.15 m.
- Harder version (`start.py hard`): 20 random directions between the lines, goals at 0.95, 1.0, 1.05 of the map's
  edge, four 0.15 m detours (+y, +z, -x, and 0.1 +x 0.1 +z). Same answer on 217 of 240. By depth: 79 of 80, 76 of 80,
  62 of 80. Flips go both ways: 14 go -> no-go, 9 no-go -> go. From rest 20, 20, 6 of 20 are reachable, so between
  lines the map sits a little inside the real edge.
  Reading: start-independence holds away from the edge and fails within about 5% of it. That is inside the caution
  zone (0.90 to 1.10) the fence already has, so the three zones stand, but 'drawn once' needs the margin said with it.
- Dropped: the 37 of 42 / 16 of 18 counts (scratch scripts gone). The schemes paper quotes 271 of 300 from the map table.

## D35 The cage is a guide, not the gate (2026-10-10)
- Bigger detours (`start.py far`): the same 20 between-line directions, goals at 0.6, 0.8, 0.9, 1.0, 1.05 of the map's
  edge, five 0.3 m detours (+z, -z, -x, +x, +y), all arrived. From rest 20, 20, 20, 20, 6 of 20 reachable.
  Same answer on 428 of 500. By depth: 99, 95, 94, 76, 64 of 100. By detour: +z 84, -z 88, -x 89, +x 91, +y 76 of 100.
  Flips: 51 go -> no-go, 21 no-go -> go.
- Reading: D34 said start matters only within about 5% of the edge. That was true for 0.15 m detours and is false for
  0.3 m ones: 5 to 6 of 100 flip inside the free zone, a quarter at the edge. 'Drawn once with a 5% margin' is withdrawn.
- What stands: the cage is drawn from rest and is a guide. The rehearsal runs from the true state and decides
  (as D19, D20 already said). The feedback study starts every target from rest, so its map is the one that was measured.
  Not covered: a second send on the same study target after a move that arrived part way.
- Not done: redrawing the cage from the current start. 98 probe moves take about 85 s, too slow to do live.
- Paper: P2 TODOs closed (hand to base weight 2 to 1 from TW_BASE, related work), the stale 'scores in points' TODO
  replaced (feedback() already scores seconds), operator view figure added (`analysis/envelope/opview.py`).

## D36 Every send in the feedback study starts from rest (2026-10-10)
- D35 left one hole: a second send on the same target started from wherever the first move ended, where the map is
  less right. Now a send with the hand away from rest first puts the arm back at rest (`free_drive`, aid mode only).
- The jump home costs no time. A miss still costs its own travel and the travel of the next send.
- Check: `tests/test_resend.py`, the hand is 0.085 m out before the second send and 0.0006 m from home just after.
- Analysis written before any participant: `analysis/study/feedback.py` (seconds per goal, wrong goes, give-ups,
  Friedman across cues, Wilcoxon pairs from 6 participants). `--check` runs it on made-up rows that are never saved.
  pid 0 is the self-test and is left out. NASA-TLX is not recorded by the tool yet.

## D37 New problem statement: the tilt-bounded workspace as something an operator works against (2026-10-10)
- Why: the old gap (show the operator a reach map) is too thin for a journal. Start-dependent workspaces are old news:
  Umetani and Yoshida already have a straight-path workspace from a given start posture and a guaranteed workspace,
  Papadopoulos and Dubowsky have path-dependent and path-independent workspaces.
- What is still open, after five searches (a hypothesis, not a proof): those workspaces are bounded by singularity,
  not by a base tilt budget, and no human works against them. Sato 2002 keeps the tilt in limits by slowing the arm
  silently, no user study. The JHU virtual fixture studies have users but the limits are fixed in space. The DLR
  null-space wall is a haptic wall that moves with the state, simulation only, no users.
- So the paper is two things. (1) The 2 deg workspace as an object: how its edge moves with where the hand starts,
  an inner set that holds from any start, and a live estimate cheap enough to redraw. (2) A study, about 20 people:
  guaranteed map (small, never wrong) against live map (bigger, sometimes wrong) against rehearsal only.
- Working guess for (1): tilt is counted from the rest attitude, so a detour leaves tilt behind and that eats the
  budget. If that is most of the shift, the live map is the rest map with the budget corrected. `analysis/envelope/guar.py`
  tests it (edge with the leftover tilt kept, and with it forgiven). Numbers go in the next entry.
- Analysis for (2) is written before any participant: `analysis/study/feedback.py --maps` (files `maps_*.csv`, same
  columns and tests as the cue study). `--maps --check` runs on made-up rows. The tool does not run this study yet.
- CA3 paper stays as it is. This is the follow-on.

## D38 The guess in D37 was wrong: leftover tilt does not explain the moved edge (2026-10-10)
- Run: `analysis/envelope/guar.py`, 20 directions, edge found by bisection (to 0.016 of the map edge) from rest and
  after five 0.3 m detours. Scripted, no participants. Saved in `data/guar_medium.npz`.
- Mostly the edge does not move. 77 of 100 direction and detour pairs are within 0.05 of the rest edge. Median edge
  1.03 from rest, 0.99 to 1.08 after a detour.
- A few collapse: worst 0.49, then 0.60, 0.66, 0.68, 0.77. Four of the five are hand at +y going far to -x, the fifth
  is hand at -x going to +y. Long cross moves. Replayed all five, every one stops on the 2 deg tilt, not the bus
  and not a stall.
- Guaranteed edge (worst of the six starts): median 0.97, worst 0.49. As volume it keeps 70% of the rest reach.
  Only over these six starts, so it is not a guarantee for any start yet.
- Forgiving the leftover tilt does not put the edge back: median distance to the rest edge 0.05 forgiven against
  0.03 kept. It moves the edge both ways depending on direction (tilt is a vector, the budget is a ball around the
  rest attitude). It does help the collapsed ones partly (0.49 to 0.61, 0.77 to 1.14). So "rest map with a corrected
  budget" is dead as the live estimate.
- Second guess, also weak: shift against the area of the triangle rest, start, goal. Big shifts only happen at big
  area (worst -0.29 m above 0.09 m2, -0.05 m below 0.03 m2) but a signed linear fit explains 7%. Not a predictor.
- Where that leaves the live map: no cheap formula yet. Rehearsal is still the only thing that is right.

## D39 Live cage timed, guaranteed set widened (2026-10-10)
- Live cage: the 98 probes shared over a warm pool of processes (`src/livemap.py`, timing in
  `analysis/envelope/live.py`). 12.6 s from rest and 7.4 s after a 0.3 m detour on 10 processes, against 85 s on one
  thread. 12 processes is no faster. Tips match the cached cage exactly. So the live map is right but 7 to 13 s old:
  it is for a hand that has settled, not one that is moving. Check: `tests/test_livemap.py`. Not drawn by the tool yet.
- Guaranteed set over 40 random one-leg starts (`analysis/envelope/wide.py`, `data/wide_medium.npz`, scripted, 18 min):
  median edge 0.74, worst 0.35, 44% of the rest reach by volume. The same at 20 starts as at 40 (59% at 5 and 10),
  so it has probably stopped shrinking for one-leg starts. Starts with several legs are not covered.
- 80% of pairs are within 0.05 of the rest edge, 7% are under 0.8 of it. The bad ones are spread out: 17 of 40 starts
  and 13 of 20 directions have at least one. More common on cross moves (8 to 10% against 3 to 4% along or against
  the start) and with more tilt left (15% above 0.8 deg against 5% below 0.4), but neither predicts which.
- So the trade for the study is real: guaranteed map = 44% of the reach and never wrong (as far as tested),
  rest map = all of it and wrong on about 1 pair in 14 near the edge, live map = right but seconds late.

## D40 Three maps in the study mode (2026-10-10)
- `python src/teleop.py --maps <pid>`: three blocks, one per map, same goal sets and block shuffle as the feedback
  study. Saves `data/participants/maps_<pid>.csv`, read by `analysis/study/feedback.py --maps`.
- guaranteed: the rest cage pulled in to the edge that held from all 40 one-leg starts. The dot stops at it.
  That edge is measured on 20 directions, the cage has 98 lines, so each line takes the worst of its 3 nearest
  measured directions. That costs reach: the drawn cage is 27% of the rest volume, not the 44% of D39
  (median scale 0.55, smallest 0.35). Sweep the 98 lines themselves before the paper calls it guaranteed.
- live: starts as the rest cage. Each time the hand stops somewhere new the 98 probes run from there on the pool
  and the cage is swapped in when they finish. Ball on the hand is yellow while the map is old, green when current.
  The dot stops at whichever cage is on screen. Only the 2 deg shell is drawn, the pool returns tips only.
- gate: no cage. Every go is rehearsed from where the arm is and refused if it cannot work. The wall time of the
  rehearsal (about 4 s on a far goal) is added to the goal time, otherwise the gate would look free.
- Unlike the feedback study nothing is put back to rest between sends. Every second goal starts from rest, the
  other starts where the one before left the arm. That is one leg, which is all the guaranteed set was tested on.
- `true_go` is rehearsed when a goal appears, from the state the arm is in then, not taken from the rest pool.
- Check: `tests/test_maps.py`, scripted operator, all three maps. 17 tests pass. No participant has run it.
- Open: the first goal of a pair can be unreachable, then the second also starts from rest and tests nothing.
  A dropped live probe keeps the pool busy, so the next map after a reset can be later than 13 s.

## D41 Guaranteed cage measured on its own 98 lines, a hole in it fixed, study mode made complete (2026-10-10)
- Run: `analysis/envelope/wide.py --98`, the same 40 one-leg starts as D39, edge searched on the 98 lines of the
  cage (1 = the tip from rest). 77 min on 10 processes, scripted. `data/wide98_medium.npz`, `paper/figs/fig_wide98`.
- 3920 pairs. 74% within 0.05 of the rest edge, 7% under 0.8. Same picture as the 20 random directions.
- Guaranteed cage: median line keeps 0.67, worst 0.24, 40% of the rest reach by volume. 28 of 98 lines keep less
  than half, 5 keep all of it. The tool now draws this one, not the 27% cage D40 had to borrow from 20 directions.
- A hole: on 22 pairs (7 starts, the 10 longest lines) even the near end at 0.3 was out of reach, the search gave
  up and saved nothing, and the cage code skipped empty pairs. It would have drawn those lines longer than
  anything measured. Fixed twice: `wide.py --98 --fill` searches those pairs below 0.3 (they came out 0.24 to
  0.29), and `guaranteed()` now counts any pair with no measured edge as 0.
- Not explained: the lowest edges pile up between 0.24 and 0.30 on the long lines, from different starts.
- Still only one-leg starts. It is "held from 40 starts", not a guarantee.
- Study mode: a fourth condition `rest` (the full cage from rest, dot stopped at it), which is the control the
  other three are compared against. Block order is a balanced Latin square over the participant id (4 orders).
  After each block the tool asks raw NASA-TLX (six numbers) and trust 1 to 7 and saves them. The live map uses
  cores minus 2 processes, at most 10, and saves the number.
- Analysis takes the four conditions, Holm on the Wilcoxon pairs. Protocol written before data:
  `analysis/study/PROTOCOL_maps.md`. Journal draft with an empty results section: `paper/journal/draft.tex`.
- `analysis/study/prestudy.py`: a fixed rule plays the operator through all four maps, saved to `data/scripted/`.
  It is there to see if the maps can differ at all on these goals. Not participant data.
- Check: 18 tests pass. No participant has run it.

## D42 The live map was drawn from the wrong place, fixed (2026-10-10)
- Found by the scripted pre-study, id 1: in the live block 4 of 7 scored goals that the rehearsal says are reachable
  were given up with no send, because the dot was held short of them.
- Cause: a pool worker got the joint state but never recomputed positions, so every line started from where the
  hand is at rest and stopped after about 0.05 m. From rest it was right, which is why D40 saw tips matching the
  cached cage. After any move the live cage was close to nothing.
- Fix: one `mj_forward` in `livemap.probe`. Checked on 4 states from that block: the 3 lines nearest the goal are
  the same from a worker and in process to the mm (0.31 to 0.86 m, not 0.05), and the goals sit at 0.95 to 1.05
  of the live cage against 0.92 to 1.06 of the rest cage. New test `test_live_line_starts_at_the_hand`.
- The 7 s after a detour in D39 was the broken map being quick. Timed again: 13.0 s from rest, 12.2 s after a
  detour, 10 processes. So "right but 12 to 13 s old". Protocol and draft changed.
- `prestudy.py` had its own bug: the "updating" flag from the live block carried into the next block and the rule
  never sent or parked there (rest and gate blocks of id 1, all 90 s timeouts). State is now cleared per block.
- The id 1 rows from before the fix are deleted. No participant ever saw the broken map.

## D43 The guaranteed cage does not hold two moves from rest (2026-10-10)
- Run: `analysis/envelope/twoleg.py`, 40 starts reached by two moves (rest, a, b; a and b drawn like the D39
  starts, other seeds). From each, one rehearsal to the tip of each of the 98 lines of the guaranteed cage.
  5 min on 10 processes, scripted. `data/twoleg_medium.npz`.
- All 40 arrived. One had 2.74 deg of tilt on arrival: two moves that are each fine can spend the budget together.
- From the other 39: 24 starts have at least one tip the rehearsal refuses. 4.3% of pairs, 70 of 98 lines. Most
  starts lose 1 to 9 tips, five lose 16 to 24. Lines already cut to 0.4 fail as well as lines left at full length.
- Same test with every line pulled in to 0.8 (`twoleg.py 0.8`): 11 of 39 starts, 0.9% of pairs, 23 lines.
- So the cage is "held from 40 one-move starts" and nothing more. Shrinking it further does not fix it.
  The study only uses one-move starts on the scored leg, so the condition stands as designed. The name stays as a
  label. Protocol and draft say so.
- Scripted pre-study, ids 1 to 4, after the D42 fix, 28 scored goals per map (`data/scripted/`, not participants):
  rest 2 wrong goes and 4 reachable goals given up; guaranteed 0 and 9; live 0 and 4; gate 0 wrong goes, 0 given
  up, 15 refused sends. Seconds per goal 32.6, 34.7, 34.9, 22.1. The maps can differ on these goals.
- One check per tip, so it says a line fails, not by how much.

## D44 A map without the physics is right but only twice as fast (2026-10-10)
- Tried to get the live map under a second: `livemap.fast`, the same control law, base moved by momentum
  conservation, no physics step. Checked against `T.ray` from rest and 4 starts, 98 lines each
  (`analysis/envelope/fastmap.py`, `data/fastmap_medium.npz`).
- At a 0.02 s step: median error 7 to 8 mm, 95% under 0.06 m, 0 to 2 lines promise 0.05 m too much. About 60 s
  for 98 lines on one process, so roughly half the cost of the simulated map. Timed with another job running.
- Bigger steps are faster and wrong: at 0.05 s, 53 to 65 of 98 lines stop 0.05 m or more short. The time goes in
  the control law per tick, not in the physics, and the law does not survive a coarse step.
- Not wired into the tool. 2x does not change what the operator sees. A map in under a second needs something
  other than replaying the controller.

## D45 The textbook planar arm shows the same thing (2026-10-10)
- `analysis/envelope/planar.py`: two-link arm on a free base, exact momentum, no MuJoCo. Base 200 kg, links 10 kg
  and 1 m. Made-up numbers, not our arm.
- Known result reproduced (Papadopoulos and Dubowsky): with no tilt limit, straight moves that stay clear of the
  radii with a dynamic singularity never failed (0 of 131), moves through those radii failed 24 of 142.
- Ours: 2 deg budget from rest, 36 lines, 40 one-move starts. The rest map at full size holds for 60% of
  start-tip pairs and from none of the 40 starts on every line. Pulled in to 0.8: 99%, 33 of 40 starts.
  Pulled in to 0.5: 100%.
- First-order guess from the start state (tilt spent + tilt per metre x distance): agrees 57% at full size and
  says yes when it is no on 18% of pairs. At 0.8: 87%, no false yes. So it is fine well inside and useless at the edge.
- Two joints cannot hold the base level at all, so tilt builds from the first centimetre. Our 7-joint arm holds
  it level until it runs out of joints. This is the harder case for a map, not a copy of ours.
- Not checked: the singular radii against their closed form, only that moves clear of them never fail.

## D46 A looked-up map is 3 to 4 times closer than a fixed one and still promises too much (2026-10-11)
- `analysis/envelope/learn.py`: 200 states (100 one move from rest, 100 two moves), the 98 reaches from each
  with `livemap.fast` (`data/learn_medium.npz`). Input 7 joint angles + base tilt, output the 98 reaches.
  Numpy only: ridge, and the average of the 5 nearest states. Compared with the mean of the training maps,
  which is what one fixed map amounts to.
- States left out, 5 folds: mean map median error 0.120 m, ridge 0.043, 5 nearest 0.034. 95% of lines within
  0.33, 0.21, 0.18 m. Promises 0.05 m too much on 43%, 20%, 18% of lines.
- Two-move states are a bit worse than one-move (median 0.037 against 0.030, 5 nearest).
- Against the simulated map (rest + 4 starts, 490 lines): 5 nearest median 0.026 m, 95% under 0.10 m, too much
  on 4.9% of lines, short on 19%.
- So it is instant and much closer than a fixed map, and it is not safe to act on: about one line in six says
  0.05 m more than is there. Not wired in. More states would help, by how much is not measured.
- Labels come from `livemap.fast`, 8 mm median off the simulation (D44), not from the simulation.
- The mean map is not the rest cage the tool draws. That comparison is not done.

## D47 The moving edge belongs to the reactionless controllers, not to the arm (2026-10-11)
- `analysis/envelope/second.py`: the same test under four controllers on the same arm. Map from rest (98 lines),
  10 starts one move out, then a straight move from each start to every point of the rest map pulled in to
  0.95, 0.8, 0.5. `T.ray` now also says what ended a line (`why=`).
- 5 Imp.+RNS and 4 RNS (hold the base level while they can): reach from rest 0.24 to 0.75 m, median 0.45.
  Pulled in to 0.8 the map holds for 90% and 91% of pairs, every tip from 0 of 10 starts. At 0.5: 98%, 5 of 10.
- 1 DLS and 3 Impedance (do not look at the base): reach 0.24 to 0.64 m, median 0.39 and 0.37. Pulled in to 0.8
  the map holds for 100% of pairs from 10 of 10 starts. At 0.95: 86% and 89%.
- So a plain controller with a 0.8 margin has a fixed map on this arm. Holding the base level buys about 0.07 m
  of median reach and costs the fixed map. The claim "no fixed map for a free-floating arm" is dead. What stands:
  no fixed map for a reactionless controller under a tilt budget.
- Why: the failed moves under 4 and 5 end on the tilt budget or the bus with 2 joints at a limit (median). Under
  1 and 3 there are no joints at a limit. The level-base controllers park joints on their limits and what is left
  depends on which ones. Believed, from the counts. Not shown joint by joint.
- The short lines (0.24 to 0.30 m) are stalls with no joint at a limit, and they are there under all four
  controllers (11 to 26 of 98 lines, median 0.28 to 0.30 m). That is the arm, not the controller. Which
  singularity or reach limit it is: not looked at.
- Fits the planar arm (D45): two joints cannot hold the base, and there 0.8 held for 99%.
- Limits: 10 starts per controller, one move out, starts taken on each controller's own map. 20 to 55 failed
  moves under 4 and 5 ended on nothing (ran out of time short of the goal), not counted as a cause.
- Planar band checked against the masses: a + b - c = 0.41, a + b + c = 2.36 m, same as the grid. Assert in
  `planar.py`.

## D48 Over a 12-move session only the impedance controller keeps its fixed map (2026-10-11)
- `analysis/envelope/session.py`: 10 sessions of 12 moves per controller, goals inside its own 0.8 map
  (0.2 to 0.8 along a random line), each move rehearsed and made only on a yes. After moves 6 and 12 a straight
  move to every point of the 0.8 map.
- Moves inside the map that the rehearsal refused: 5 Imp.+RNS 4 of 120, 4 RNS 8 of 120, 1 DLS 0, 3 Impedance 0.
  All of them from move 6 on.
- 0.8 map after move 12, pairs held and sessions where every tip held: Imp.+RNS 91%, 0 of 10. RNS 80%, 0 of 10.
  DLS 93%, 4 of 10. Impedance 100%, 9 of 10.
- So DLS loses its fixed map too once the session is long (it was 10 of 10 one move out, D47). Impedance keeps
  it. The difference between those two is the pull back to the rest posture.
- Tilt left after a move goes up with the move number under all four. Median after move 1 and move 12:
  Imp.+RNS 0.10 and 0.51 deg, RNS 0.09 and 0.58, DLS 0.71 and 1.25, Impedance 0.62 and 1.05. 12 moves is as far
  as this goes. Whether it levels off is not measured.
- Limits: 10 sessions per controller. The hand never makes a refused move, so no recovery from over the budget.
  Goals are mid-map, the edge was only tried at moves 6 and 12.

## D49 Over 40 moves no controller keeps the fixed map at its edge, and two of them get stuck (2026-10-11)
- `session.py 40`: the D48 test with 40 moves, 10 sessions per controller (`data/session40_medium.npz`).
- 0.8 map after move 40, pairs held and sessions where every tip held: Imp.+RNS 88%, 1 of 10. RNS 83%, 1 of 10.
  DLS 65%, 3 of 10. Impedance 85%, 4 of 10 (it was 99%, 8 of 10 at move 20).
- Mid-map moves the rehearsal refused, of 400: Imp.+RNS 15, RNS 46, DLS 40, Impedance 3.
- Stuck sessions (every one of the last 10 moves refused, tilt sitting at the budget): DLS 2 of 10, RNS 1 of 10,
  Imp.+RNS 0, Impedance 0. The two that never get stuck are the two with the pull back to the rest posture.
  Part of this is the test: a refused move is not made, so nothing brings the base back.
- Tilt left after a move does not keep climbing. Median over the first 10 and the last 10 moves: Imp.+RNS 0.31
  and 0.45 deg, RNS 0.33 and 0.58, DLS 0.95 and 1.22, Impedance 0.85 and 0.97.
- So D47's "a plain controller with a 0.8 margin has a fixed map" holds for one move and not for a session.
  Impedance is the closest to a fixed map and still loses the edge by move 40.
- Read for the paper, from four searches, highlights only, papers not read through:
  Papadopoulos, Teleoperation of Free-floating Space Manipulator Systems, SPIE 1992: a planning aid that checks
  whether a target can be reached from the start and offers another path. That is the rehearsal in outline.
  Cocuzza et al., Acta Astronautica 2010: reaction control with joint limits. Kinematic drift over repeated moves:
  a 2002 classification paper and Repetitive Motion Planning of Free-Floating Space Manipulators, 2013.
  No study with people that shows a reach map for a floating base turned up.

## D50 Read the 1992 planner, rewrote what the journal draft claims (2026-10-11)
Read Papadopoulos, "Teleoperation of Free-floating Space Manipulator Systems", SPIE 1992, sections 1 to 5 (the
example section was cut off in the copy I got).
What it already has: a planner for the operator. Operator names target and path, planner checks the path against
dynamic singularities, previews the simulated motion, executes after approval, and proposes another path if it
fails (detour through the path-independent workspace plus small closed hand motions that turn the base).
So "rehearse then execute" is not ours, and neither is "the reach depends on the path".
What it does not have: any limit on base attitude (it turns the base on purpose), joint limits, collision (named,
left out), redundancy or reactionless control (example is planar, 2 joints), a map shown to the operator, any
measurement of how wrong a map is, any user.
What the draft now claims: the reach under a tilt budget with joint limits and bus contact; the measured
comparison of maps against the check, across four controllers and over 40-move sessions; the edge failing first
under the reactionless controllers; the study. Abstract, introduction and related work rewritten, sections added
for D46 to D49 and the planar check. Title changed.
Still not read in full: Cocuzza 2010, the two drift papers, Sato 2002, the DLR wall. Marked red in the draft.

## D51 The other related papers, and what they take from us (2026-10-11)
Read from abstracts and excerpts, not in full: Sato et al. ICRA 1993, Cocuzza et al. Acta Astronautica 2011,
Chen et al. IJARS 2013, Coelho et al. ICRA 2020 (the "DLR wall").
- Sato 1993 ("Sato 2002" in our notes was the upload date). A tilt limit while teleoperating a satellite arm,
  kept by slowing the arm plus feedforward to the attitude control. The operator is deliberately not shown it.
  So "a tilt budget in teleoperation" is not ours. Ours is the free-floating case, where nothing recovers the
  attitude, slowing does not help, and the limit becomes an edge of the reach that somebody has to see.
- Chen 2013. Joints and base attitude do not come back after a closed hand path, known since the 1980s for
  redundant arms; they plan the reset. So the stuck sessions of D49 are known drift. Ours is what it does to a map.
- Cocuzza 2011. A reactionless controller with joint limits in it. A controller, no workspace, no operator.
  Not tried here: would its handling of joint limits keep our edge still? Open.
- Coelho 2020. Haptic wall that moves with the state, aerial arm, base actuated, simulation only, no users.
The "2002 drift classification" paper did not turn up again, dropped from the draft.
What is left as ours after all of this: the tilt-bounded reach of a free-floating arm as a thing to draw, the
measured trust in maps of it (4 controllers, 1 to 40 moves), and the study with people. No single idea in it is
new. The measurement and the study are.

## D52 Joint limits inside the reactionless controller: no change (2026-10-11)
Question left by D51: our reactionless controllers lose the map with joints on their limits. Would handling the
limits inside the solve keep the edge still?
Scheme 6 = scheme 5 + weighted least norm (Chan and Dubey 1995): a joint heading for its limit costs more the
closer it is (BAND 0.5 rad, first value tried). Velocity level. In the spirit of Cocuzza 2011, not his solver.
Same tests as D47 and D49, same seeds, 10 starts and 10 sessions (`second.py 6`, `session.py 40 6`).
- Rest map: reach 0.24 to 0.75 m, median 0.46 (scheme 5: 0.45). Ended by tilt 65, stall 24, bus 9.
- One move out: 0.95 map 84%, 0.8 map 91%, 0.5 map 97% of pairs. Every tip from 0, 0, 4 of 10 starts.
  Scheme 5 at 0.8: 90%, 0 of 10. Failed moves still have a median of 2 joints at a limit.
- 40 moves: 23 of 400 refused (scheme 5: 15). After move 20: 93%, 1 of 10. After move 40: 90%, 0 of 10
  (scheme 5: 88%, 1 of 10).
So no. Why: hand 3 + level base 3 = 6 of 7 joints, one left to spare the others with. In the unit check a joint
0.1 rad from its limit slowed by 3%.
What this does not rule out: a tuned weight, the acceleration-level solver, or a controller that gives up some
tilt to save a joint (that spends the budget the map is about, so it is a different trade, not tried).
For the paper: the answer to "what to show" does not change. One more controller, same result.

## D53 The looked-up map made careful: under 1% over-promise costs about half the reach (2026-10-11)
D46's lookup (5 nearest, averaged) promised 0.05 m too much on 4.9% of pairs. Too much for a map somebody trusts.
Careful version (`learn.py safe`): per line the shortest reach among the k nearest stored states, times a pull-in
factor, and no reach when the base is already over 2 deg (one of the 200 states is, it was 98 of the misses).
200 states x 98 lines, 5 folds, labels from livemap.fast as before.
- 5 nearest, no pull-in: over by 0.05 on 3.5%, keeps 0.78 of the total reach.
- 5 nearest x 0.7: 0.8%, keeps 0.55.   10 nearest x 0.8: 0.8%, keeps 0.54.   20 nearest x 0.7: 0.3%, keeps 0.40.
- Against the simulated map (rest + 4 starts): 5 nearest x 0.8 and tighter never over by 0.05, keep 0.51 to 0.59.
- The misses that stay sit in 22 of the 200 states, not spread evenly.
Blunt: at 200 states this does not beat a fixed map. The map that held from every start (D41) keeps 0.67 of the
edge on the median direction and needs no lookup. Not like for like (different states, median against total),
so not a verdict yet.
Not clean: the factor was picked on the folds it is scored on. Only 5 states have simulated labels.
Also fixed the draft: abstract and contribution 1 now say five controllers (D52).
Next that would settle it: 1000 states (about 2.7 h of machine time), then the fixed map and the lookup scored
on the same left-out states.

## D54 Looked-up map against a fixed map, same states, same allowance. D53's lean was wrong (2026-10-11)
D53 said the careful lookup was no better than a fixed map. That compared it with D41's 0.67, which came from
one-move starts only. Scored on the same states it is the other way round.
800 more states (`learn.py gen more`, data/learn2_medium.npz), 1000 in all, half one move from rest, half two.
Fixed map with the same allowance = per line a low percentile of the stored reaches. `learn.py safe`.
5 folds, 1000 states, over by 0.05 m / reach kept:
- looked up: 10 nearest 1.0%, 0.77. 10 nearest x 0.9: 0.6%, 0.69. 5 nearest x 0.8: 0.7%, 0.68.
- fixed: 2nd percentile 0.6%, 0.25. 5th 1.6%, 0.38. Shortest over every state: 0.00 (some states sit at the budget).
- 200 -> 1000 states moved the lookup from 0.55 to about 0.69 at the same over-promise.
Clean score (600 stored, settings picked on 200 for under 1%, scored on 200 never used):
- looked up (10 nearest x 0.9): over by 0.05 on 1.5%, keeps 0.69.
- fixed (2nd percentile): 1.3%, keeps 0.29.
So: the lookup shows more than twice the reach of a fixed map at about the same risk. Neither held the 1% it
was picked for on new states. Both numbers stand on livemap.fast labels, only 5 states have a simulated map.
Next: simulate about 30 states in full and score against those. Then it can go into the study as a condition.

## D55 The lookup against the full simulation, and what the audit of D53 and D54 found (2026-10-12)

D54 asked for about 30 fully simulated states before the lookup goes near a participant. Did all 200 scoring
states (`learn.py sim`, stored in `data/learnsim_medium.npz`), and had D53 and D54 audited by one agent and
checked by a second. `learn.py safe` prints the scores below.

Two states are out. Rows 470 and 976 start with the arm inside the 2 cm clearance around the bus (-0.088 m and
-0.008 m): `state()` drives there with no clearance check and the tool would refuse that move. Row 470 alone
carried 90 of the fixed map's 248 over-promised pairs. 199 scoring states left.

- Fast labels against the simulation, 199 states: error median 0.009 m, 95% within 0.056, worst 0.544. Over by
  0.05 on 1.2% of lines, short by 0.05 on 5.9%.
- Lookup (10 nearest, pulled in to 0.9) against the simulation: over at all 3.0%, over by 0.05 on 1.4% of pairs
  (95% over states 0.8 to 2.1), in 43 of 199 states, worst 0.44 m, reach kept 0.67.
- One fixed map (2nd percentile): over at all 1.7%, over by 0.05 on 0.7% (0.1 to 1.3), in 9 of 199 states, worst
  0.16 m, kept 0.29.
- Fixed map shrunk by the tilt already spent (5th percentile, 0.9). New, and the fair baseline, because the
  lookup's neighbours carry the tilt for free: over at all 2.2%, over by 0.05 on 0.6% (0.3 to 0.9), in 25 of 199
  states, worst 0.22 m, kept 0.39.

So the lookup keeps 2.3 times the reach of the fixed map and 1.7 times that of the fair one. D54's "at about the
same risk" does not hold: the lookup is wrong in 43 states against 9 and 25, and its worst miss is 0.44 m against
0.16 and 0.22.

What else the audit found, all confirmed by the second agent. These are on the fast labels, not the simulation:
- "Neither kept the 1%" was one split's luck. Over 100 random 600/200/200 splits the lookup averages 0.85% and is
  above 1% in about 30, the fixed map 0.75% and above 1% in 26 to 30. Seed 0 sits near the 93rd and 85th
  percentile.
- The lookup holds only for states drawn like the stored ones. By quarter of distance to the nearest stored
  state, lookup against fixed, % over by 0.05: 0.10 / 0.00, 0.16 / 0.04, 0.78 / 0.10, 5.04 / 4.92. Stored one-move
  and scored two-move: 3.36% against 2.82%. 20 hold-outs by region: lookup median 6.33% (worst 16.49%), fixed
  1.62% (worst 7.91%), lookup worse in 19 of 20.
- Tried: beyond 1.0 sd from the nearest stored state take the smaller of lookup and fixed. 0.86% and 0.64 kept,
  region hold-outs median 2.80%. Halves it, does not close it.
- A lookup on joints alone, without the tilt: 1.51% and 0.685. The tilt features are what keep it under.
- D46's "0.05 m too much on 4.9% of lines" came from 5 simulated states. On 101 to 125 states it is 14.0 to
  15.8%, short by 0.05 on about 20%, median error 0.033 to 0.035 m. One line in seven, not one in twenty.
- D54's "a fixed map that holds from every stored state has no reach left, because some states are already at the
  budget" is half the reason. The zeros come from row 669 (over 2 deg) and row 976 (in the bus). Without the
  all-zero rows the per-line minimum keeps 0.089 and is over on 0.14%.
- "200 stored states instead of 1000" was instead of 600. Kept reach by stored states: 50 0.35, 100 0.46, 200
  0.56, 300 0.65, 400 0.67, 600 0.69.
- Per state the lookup shows a median 0.70 of the true reach, the fixed map 0.29. Lines shown at 0.8 or more of
  the true reach without over-promising: 20.2% against 3.3%. Lines shown at under half: 12.3% against 84.0%.
- `max()` on an empty grid crashed when no setting passed. Fixed.

Not redone: the 5-fold tables of D53 still include rows 470 and 976.

For the paper: "Made careful" rewritten with these numbers and the scope limit. The lookup is good near its stored
states and no better than a fixed map away from them. That goes in the protocol too if it becomes a condition.

## D56 The short lines are the arm at full stretch (2026-10-12)

D47 left 11 to 26 lines per controller that stop short "with no joint at a limit" and called it a singularity we
had not worked out. One agent worked out the mechanism, a second tried to break it.

Held:
- All 26 stall lines of scheme 5 end with the hand 0.962 to 0.970 m from the shoulder. Full stretch is 0.970 m.
  Elbow and wrist within 12 deg of straight, base at 0.24 to 1.98 deg.
- Bus x10, x1000, 3000 kg: still stalls at 0.970 m. Hand site moved out 0.10 m: stalls at 1.063 to 1.070 m. A
  start 0.21 m further in gives 0.40 to 0.41 m on lines 43, 45, 58, was 0.24 to 0.25 m. Schemes 1, 3, 4, 6 end at
  the same distance.
- So it is the reach limit of the arm, not a dynamic singularity of the free base. Rest is close to full stretch
  in those directions.

Did not hold:
- "No setting moves it" is false. With the joint rate cap at 8 or 20 rad/s lines stall with the elbow bent 9 to
  32 deg, 0.5 to 5 cm short. On the 30 kg bus 4 of 6 lines stall 1.3 to 7.6 cm short.
- The hand is under half speed from 0.83 to 0.97 m (median 0.915): the scheme 5 stack asks 5 to 15 rad/s there
  and the 2 rad/s cap scales it down. The last 2 to 14 cm is a crawl the controller makes. `dpinv` eps 1e-4
  removes the crawl and changes the map.
- "No joint at a limit" is a median. 10 of the 26 rest stall lines have joint 5 on its limit (13 of 26 under
  scheme 4, none under schemes 1, 3, 6). From two other states 30 of 34 stalls were full stretch and 4 were
  joint-limit stalls.
- 7 lines that reached full stretch are labelled "tilt".
- Stall-line reach is 0.24 to about 0.53 m, median 0.29. Not "0.24 to 0.30".

Next: label a stall with hand-to-shoulder at 0.96 m or more as a reach limit and a joint-limit stall as its own
kind. Not done, `T.ray` still says "stall" for both.
For the paper: the "which singularity" sentence is replaced.

## D57 Scheme 6: the result stands, the reason in D52 was wrong (2026-10-12)

D52 said the weighted solve changes nothing because hand 3 + base 3 leaves one joint of seven to spare the others
with, and that a joint 0.1 rad from its limit slowed by 3%. Audited, then checked by a second agent. All five
findings confirmed.

- The spare direction is the tool roll. At rest the null vector of `[2J; W]` is 99.98% joint 7. It moves neither
  hand nor base and cannot unload joints 1 to 6. Stopping the joint nearest its limit through it needs a median
  3412 rad/s over the 98 directions (least 56) against a cap of 2. Along moves the median is 1032 to 2164 rad/s
  and 525 of 535 states are over the cap.
- The 3% was one entry of a random matrix. The unit test used a random J and W: joint 2 of seed 0 slows 3.22%,
  the other joints of the same matrix 0.02 to 59.75%, seeds 1 to 3 give 0.0003 to 69.9%. It also ran with `tw` at
  100 where teleop uses 2, and would pass at any slow-down above zero.
- On the real arm at rest the slow-down per joint is [24.27, 0.18, 49.12, 0.16, 36.87, 1.81, 80.0]% with the
  shipped damping and [0.012, 0, 0.036, 0, 0.02, 0, 79.997]% with the damping off. So what the weights do to
  joints 1 to 6 comes through directions the damped inverse has already dropped.
- `dpinv` eps = 1e-6 sets rest reach and end reasons, and the paper did not say so. Scheme 5 at 1e-8: line 28
  0.736 tilt to 0.399 stall, line 56 0.696 tilt to 0.343 stall, line 70 0.590 bus to 0.320 stall. End reason
  changed on 5 of 8 lines. Not monotone: lines 55 and 56 lose about half at 1e-4 too.
- The null result is tied to that eps. At 1e-8 line 28 is 0.399 m under scheme 5 and 0.732 m under scheme 6, line
  56 0.343 against 0.701. At 1e-6 they agree to 1 mm. Belief, not measured: the weights shrink columns and the
  damping bites sooner.
- "It changed nothing" is "it changed two lines of 98": line 71 0.364 to 0.458 m, line 27 0.380 to 0.455 m.
  Median difference 0.4 mm. Lines ending with 0..4 joints at a limit: scheme 5 [42, 41, 7, 6, 2], scheme 6 [53,
  31, 6, 6, 2].
- Scheme 6 is refused more: 23 against 15 of 400, worse in 3 sessions, better in none, same goals.
- "Untuned" was the wrong caveat. Strong weights, a limit push and a soft stop move rest lines by millimetres (15
  mm at most) and fix 0 of 22 failing tips. What was never tried was giving up tilt, and on paper there is room:
  joint 4 moved 0.5 rad costs 0.09 deg. So D52's "So no" reads "not while the base is held level". D59 tries it.

Changed: `tests/test_scheme6.py` is rebuilt on the model at rest. The comment in `schemes.py` says what the spare
DOF is.
For the paper: the paragraph is rewritten, eps is stated in System and in Limits.

## D58 The draft audited claim by claim, and a second literature pass (2026-10-12)

One agent listed every claim in `paper/journal/draft.tex` against the records and the data, a second tried to
refute the list: 18 of 21 confirmed, 2 refuted, 1 unclear. Fixed in the draft:

- "The rehearsal is always right / never wrong / right every time" is gone. D28 had it wrong on 1 of 45, D33 on 2
  of 46, and in the stored sessions 4 of 826 accepted moves ended over 2.00 deg (1 of 354 RNS and 2 of 360 DLS in
  `session40_medium.npz`, 1 of 112 in `session_medium.npz`, worst 2.013). It is the reference by construction.
- "1 to 4 of 10 sessions" is 0 to 4 (scheme 6 has 0).
- Live map timing is D42's 13.0 s and 12.2 s, not D39's. "12 processes are no faster" dropped, no record has it.
- "The budget is not what runs out. The arm's posture is." was not supported. Stuck sessions sit at a median 2.01
  deg and all 30 refusals are for tilt. Now stated as what we saw, mechanism open.
- 12-move sessions were named and never reported. D48's numbers are in.
- "Pre-registered" is "a protocol written before any data". Sentences about what an operator can tell are marked
  as predictions.
- The scripted pre-study (D43) had run and the draft said it would. Reported as scripted.
- "air-bearing", "acceleration-level" and the line on predictive displays had no source here. Removed.
- Stated now: bus 300 kg (1 m cube), arm 12.5 kg in seven links, 0.97 m at full stretch, 2 rad/s cap, eps 1e-6,
  and that the planar check uses made-up masses (200 kg base, 10 kg and 1 m links).
- Logged here because no record had them: start tilt medians 0.40 deg (reactionless) and 1.07 deg (plain), 0.399
  and 1.0726 in `second_medium.npz`.
- Searches were about 20 plus 58, not "a dozen". Papadopoulos 1992 was read to section 5, not its example.
- D42 said "D40" where it meant D39. Fixed there.
- D16's rest-pose score (0.037 / 0.043 / 0.048) is the old pose. The current pose scores 0.078 / 0.077 / 0.077
  and `singularity_map.py` still prints the label "score 0.043". Not fixed.

Literature, space robotics, about 58 queries, two checkers, abstracts and landing pages only:
- No direct hit on any of the four claims. The idea under claim 1 is not new.
- Nearest: Jia, Yuan, Chen, IEEE Access 7, 2019: reachable sets for a free-floating arm, one of them under a limit
  on base attitude deflection, 3 DOF, no display, no decay. Umetani and Yoshida, Trans. SICE E-1, 2001: five
  workspaces, one attitude-constrained, one guaranteed. Cocuzza's zero reaction workspace (IAC 2010, 2013): ours
  reads as that with the zero relaxed to a budget.
- So claim 1 is worded as: drawn for the operator, 7 joints with limits, compared over controllers and a session.
- Stored maps queried online exist for fixed bases (Porges 2015, Cannizzaro 2022). Singularity maps for space
  robots: Calzolari, Lampariello, Giordano, RSS 2020.
- Pietras et al., IAC 2010: support for a ground operator against breaking attitude limits, on a simulator meant
  for operator tests. The abstract reports none. Until someone reads it, "no study with people" is "we found
  none".
- Human factors: 25 references checked, none a threat. ReachVox (Hauck et al., ISMAR 2025) is the nearest design:
  20 people, reachability drawn, fixed base.
- Bibliography: volume, pages and DOI filled in for 15 entries. Chan and Dubey 1995 added from memory, marked.
- Not checked: D22 gives sice.jp/e-trans/papers/E1-16.pdf to Nenchev and Yoshida. One checker says that URL is
  the Umetani and Yoshida workspace paper.
- Weak: Japanese and Chinese work, 1990s proceedings, no forward citations. Full texts unread: Jia 2019, Pietras
  2010, Cocuzza 2010 and 2013, Sato 1993, Nenchev and Yoshida 1999.

Next: get those six papers. Jia 2019 decides how claims 1 and 3 can be worded.

## D59 Scheme 7: base tilt traded for joint room. A different trade, not a better one (2026-10-12)

D57 left one thing untried: let the base go a little to keep joints off their limits. Built as scheme 7 by one
agent, reviewed by a second. `soft()` in `schemes.py`, `tests/test_scheme7.py`.

The law, 3-D only: hand rows as scheme 5. Joints weighed by room as scheme 6, but only by the share of tilt
budget left. The three base rows are a cost and not a constraint while a joint is short of room, hard again when
there is room everywhere or the budget is spent, and hard whenever the plan held for 1/KA s would pass the
budget. `TILT` = 1.5 deg and `SOFT` = 1e-3 are first values, not tuned. Three earlier forms were tried on rest
lines 0, 33, 66 before this one.

- From rest, 98 lines, scheme 5 against 7: reach median 0.449 against 0.442 m, mean 0.474 against 0.456. Ended on
  tilt / bus / stall 63 / 9 / 26 against 65 / 8 / 25. Lines ending with no joint at a limit 42 against 61.
- Per line, 7 minus 5: further by over 5 mm on 18 lines (median gain 9 mm), shorter on 22, same on 58. Shorter by
  10 to 41 cm on 9 lines (35, 48, 52, 55, 56, 59, 67, 68, 72).
- One move out, 4 starts, 98 goals each at 0.8 of scheme 5's map: scheme 5 reaches 373 of 392, scheme 7 372.
- That is two effects that cancel. On the 89 other lines 7 reaches 349 of 356 and 5 reaches 340 (+9). On the 9
  lines where its rest reach fell it reaches 23 of 36 against 33 (-10).
- Each on its own 0.8 map: 373 and 384. The shorter map is not why: scheme 5 on scheme 7's map still gets 373.
- The price: smaller rest map, more tilt at the same distance out on 48 of 98 lines and less on none, pull back
  toward level cut in 57% of ticks past 1.5 deg.
- The reviewer's checks: schemes 1 to 6 give the same bits in 187,200 calls, old code against new. No soft plan
  passes the budget in 32,679 ticks. Hand error is smaller under 7 than under 5.

Thin: four starts, and start 3 carries 6 of the +9. The measuring scripts are not in the repo. `second.py 7` then
`session.py 40 7` would be the full run. Not done.

For the paper: a short paragraph after the joint-limit one. Not a column in any table.

## D60 The looked-up map is in the tool as a fifth condition (2026-10-12)

`--maps` now runs five blocks: rest, held, current state, no map, looked up. `livemap.lookup()` takes the 10
nearest stored states, pulls the reach in to 0.9 and gives the cage the moment the hand stops. No pool, no wait,
no yellow marker. Reach is zero on every line once the tilt is at the budget.

Block order is a Williams design for five: 10 orders on the participant id, every map 4 times in every place and
4 times right after every other in 20 sessions. `tests/test_order.py` checks that from the tool itself.
`tests/test_maps.py::test_lookup` checks the cage against `learn.safe` on the same state, the zero at the budget
and one headless drive.

What this does not settle:

- 4 goal sets for 5 conditions. Rest and looked up get the same 15 goals in every participant, shuffled again.
  A fifth set means drawing the pool again, which changes the sets the scripted logs ran on.
- No hypothesis is written for the looked-up map.
- Block orders per id changed, so the scripted logs of D43 ran in the old orders. They have four blocks and
  still read. No scripted run has a looked-up block.
- D55 stands: the looked-up map keeps 0.67 of the reach against 0.29, and is over by 5 cm in 43 of 199 states
  against 9. It is only that good one or two moves from rest.
- Nobody has looked at it on a screen. How the smaller cage reads next to the dot, and whether the jump from the
  rest cage is confusing, is not known.
- Five blocks do not fit the hour (D61). Four or five is for the team to decide after the pilot.

## D61 The study design after a methods review, and what 20 participants can show (2026-10-12)

The design in D41 had holes. A referee pass found them and the Design section of `paper/journal/draft.tex` is
rewritten. `analysis/study/PROTOCOL_maps.md` logs the changes with today's date. No participant has been run.

What changed:

- Primary measure: test goals with a command the fence had to bring back, not seconds per goal. In the scripted
  run seconds follow the 30 s give-ups more than the map.
- One aim for the participant: lowest total time, with the costs told. "No wrong go first, then speed" had no
  price, so a participant could score well by never sending near the edge.
- A give-up costs 30 s. Missed edge goals are a planned contrast, so holding back shows.
- Goal sets drawn again: 10 pairs a block, 2 traps, 4 edge, 2 inner, 2 outer, edge goals at the depth of the
  traps. As the tool draws goals now a block holds a trap about one time in five and the maps can hardly differ.
- Set index gets one more step every four participants. Before, a set came at a given place 10 or 0 times in 20
  sessions. Now 4 to 6.
- Practice under the participant's own first display, not under the rest map for everyone.
- One confirmatory test (H1), five secondary tests under Holm. The unit is the participant.

Sample size, from `analysis/study/power.py` (4000 simulated studies a cell, seed 0). a is the chance a person
sends on a trap under the rest map, b the chance they send before the current-state map is in. Both are guesses.

| case | a | b | n = 12 | n = 20 | n = 30 |
|---|---|---|---|---|---|
| kind | 0.9 | 0.2 | 0.99 | 1.00 | 1.00 |
| kind | 0.9 | 0.5 | 0.64 | 0.93 | 0.99 |
| kind | 0.7 | 0.35 | 0.40 | 0.72 | 0.93 |
| less kind | 0.9 | 0.2 | 0.69 | 0.93 | 0.99 |
| less kind | 0.9 | 0.5 | 0.25 | 0.47 | 0.68 |
| less kind | 0.7 | 0.35 | 0.17 | 0.32 | 0.48 |
| goals as the tool draws them now | 1.0 | 0.0 | 0.01 | 0.11 | 0.42 |

So 20 is enough only if most people send on a trap under the rest map and most wait for the current-state map.
If half do not wait it is a coin flip. And by the scripted times waiting loses 2 to 7 s a test goal under a
time-only score, so H1 may well fail. The draft says so.

Not done, and needed before the first participant: the script that draws and sorts the new sets (whether the pool
can supply them is untested), the prompt text with the costs, the log columns for map age, `feedback.py` on the
new primary measure with the exact test, practice pairs per block, the scripted operator on the new sets. The
list is a todo in the draft. The hour: 60 min at 40 s a pair, which is a guess. Five conditions come to 64 min.

## D62 reproduce.py (2026-10-12)

`python reproduce.py` from `v1/` re-scores the stored data and checks what the scripts print against the numbers
in this file: 9 commands, one process each, 39 checks, about 2 min on a free machine, PASS or FAIL per line. It
writes nothing into the repo. Last run: 39 PASS, 0 FAIL, 90 s.
`--full` lists the commands that rebuild the data with their runtimes, `--full --run` runs them.

It does not cover anything that needs the 10-process pool, the sweep table, the scripted pre-study counts or
part B of `planar.py`. One line is known to differ from its D-number (D16 rest pose, old text) and says so.

## D63 Four conditions in the maps study (2026-10-12)

The main study has four conditions: rest map, the map that held, the map from the current state, no map. This is
a choice. Nothing new was measured for it.

Five would have cost: 64 min a session and not 60, 8 scored pairs a block and not 10, seven tests under Holm and
not five (D61). The looked-up map (D54, D55) stays in `teleop.py` as a display, `free_drive` still draws it. No
participant sees it.

In the tool: `T.MAPS` has four names and `python src/teleop.py --maps N` runs `maps_study`. The block order is
`[(pid + x) % 4 for x in (0, 1, 3, 2)]`, four orders. The goal set of a display is `(c + pid + pid // 4) % 4`.
`tests/test_order.py` counts ids 1 to 20: every display 5 times in every place and 5 times right after every
other display. A set in a place, and a display on a set, 4 to 6 times each.

## D64 Goal sets from pairs.py (2026-10-12)

`python analysis/study/pairs.py` draws the pairs of the maps study and writes `data/pairs_medium.csv`. The
rehearsals are kept in `data/pairs_cand_medium.npz`. Simulation, no people.

A pair: 2080 are drawn from seed 0, 2000 with the test goal at 0.5 to 1.0 of the rest edge and 80 at 1.2 to 1.4.
The lead goal is 0.6 of the way out in the map that held. The arm goes from rest to the lead goal, settles 1 s,
then goes straight to the test goal, as `teleop.check` does it.

| class | where the test goal is | the second move |
|---|---|---|
| trap | inside the rest map, 4 cm or more outside the map that held | ends 4 cm or more short |
| edge | same | ends within 5 mm |
| inner | at 0.9 or less of the map that held | ends within 5 mm |
| outer | at 1.2 to 1.4 of the rest edge | ends 4 cm or more short |

4 cm because the tool counts a goal as reached within 2 cm, so a goal missed by 1 cm would be no trap. A pair is
kept only if it is in the same class in 7 rehearsals: the lead move ending on the lead goal, and 2 cm off it in
six directions.

| | trap | edge | inner | outer |
|---|---|---|---|---|
| in the class, lead move on the lead goal | 79 | 929 | 480 | 79 |
| still in it in all 7 | 55 | 188 | 50 | 45 |

For edge, inner and outer only as many as needed were tried, so their second row is a floor. Traps were all
tried: 79 of 2000 pairs are traps by these rules and 55 get through the 2 cm check. That is few. Two scored traps
a block times four sets, plus four for practice, uses 12 of the 55.

Picked: 52 pairs. 4 sets of 10 scored pairs (2 traps, 4 edge, 2 inner, 2 outer), 2 more per set that start its
block and are not scored (1 edge, 1 trap), 4 for training (2 inner, 1 edge, 1 outer).

- Scored traps come from two bands of depth, four traps next to each other in each: 0.68 to 0.72 and 0.94 to 0.97
  of the rest edge. One of each band to a set. Mean trap depth by set: 0.825, 0.837, 0.832, 0.830.
- A first pick gave sets with scored traps at (0.51, 0.92), (0.64, 0.94), (0.72, 0.97), (0.78, 1.00), means 0.72
  to 0.89. The draft promises the same depths in every set, so it was thrown away. Nothing is reported from it.
- The traps that start the blocks are from a middle band, 0.86 to 0.88.
- Every set has two edge goals at the depth of each of its traps, the nearer within 0.001. So depth does not tell
  a trap from an edge goal.
- Inner goals at 0.52 to 0.90 of the map that held, outer at 1.20 to 1.40 of the rest edge.
- On the 12 traps the hand ends 4.7 to 21 cm short. What ends the move in the rehearsal: tilt 7, bus 4, and 1
  where neither limit trips and the hand is still 5.4 cm short when the time is up (a practice trap of set 2). In
  the real move the fence brings the arm back on all three kinds, the last by its time limit of 30 s.

The map from the current state was not used to sort. On the picked pairs it was probed after the lead move:

- It stops the dot short of all 12 traps. One only just: a scored trap of set 2 lies 1% outside it.
- It also stops short of 3 of the 21 edge goals, and the arm reaches all three. Two are scored (set 0, 8% outside;
  set 1, 25% outside), one is a practice goal of set 2 (1%).

So a participant who waits for that map should meet no trap, and loses one reachable edge goal in set 0 and one
in set 1. That is the map being wrong on the safe side between its 98 lines. It goes into the missed goes of that
condition. We do not pick around it.

Not covered: a lead move that took another path than the rehearsal's. So scoring uses the rehearsal from the
participant's own state when the goal opens (`true_go`), not the class in the file.

## D65 Tool changes for the four-condition design (2026-10-12)

The list was a todo in the draft (D61). Where it stands:

| # | what | state |
|---|---|---|
| 1 | a script draws and sorts the pairs, the tool reads its file | done (D64). `maps_study` reads `data/pairs_medium.csv` |
| 2 | the prompt says what a give-up and a command brought back cost | done, same words in training and in every block |
| 3 | log whether the current-state map was in, and how long a map took | done per goal, not per command. `early`: sends made while the cage on screen was not from where the hand is. `map_s`: the longest wait for a cage in that goal, wall clock |
| 4 | a time-out | counts as a give-up: +30 s, `parked` true, and `timeout` true so it can be told apart |
| 5 | gamepad | not done, never run. The study is keyboard only |
| 6 | `feedback.py --maps` | done: H1 first, the planned tests only, the exact test of `power.py`, test goals with at least one wrong go. `--maps --check` passes on made-up rows |
| 7 | set index, one more step every four participants | done (D63) |
| 8 | practice | done: 4 training pairs under the first block's display, once more if no inner goal was reached or the outer one was not given up. Every block starts with 1 edge and 1 trap pair, not scored |
| 9 | scripted operator on the new sets | D66 |

Two more things found on the way:

- A live probe still running when the arm is reset held the pool, so the next cage came late and its logged time
  was wrong. The pool is started again in that case. Timed on this machine, 10 processes, two tries: the next
  cage comes 2.3 and 3.0 s later than it would have (a cage takes 11 to 15 s here).
- No probe is asked for while the hand sits on the open goal.

Rows of `maps_<id>.csv`: `use` is train, block or scored, `block` 0 is the training. Only `use` scored and `leg`
2 are analysed. The scripted files of the first design are in `data/scripted/first_design`. No script reads them.

Still open: the scripted operator of the first design gave up all 4 reachable goals just outside the rest map
under the current-state map without sending. Believed cause: its stuck timer ran while the new cage was still
being drawn. That was changed. Not confirmed, the new sets have no goal just outside the rest map.

## D66 Scripted operator on the final goal sets (2026-10-12)

`python analysis/study/prestudy.py`, ids 1 2 3 16, about 5 min an id. A rule plays the operator, not people.
Nothing here is participant data. Files: `data/scripted/maps_{1,2,3,16}.csv`, 416 rows, 160 scored test goals,
40 per display, every display on every set once.

| display | reached | wrong go | missed | refused | gave up | s per goal |
|---|---|---|---|---|---|---|
| rest map | 24 | 8, all on traps | 0 | 0 | 16 | 27.3 |
| map that held | 8 | 0 | 16 | 0 | 32 | 33.2 |
| current-state map | 22 | 0 | 2 | 0 | 18 | 32.6 |
| no map | 24 | 0 | 0 | 16 | 16 | 18.3 |

s per goal has the 30 s of a give-up in it. Missed: the goal could be reached and the rule gave it up without a go.

What it says:

- The sets do what they were built for. When the goal opened, 64 of 64 edge and 32 of 32 inner goals could be
  reached, 0 of 32 outer and 0 of 32 traps (`true_go`). The first move of every scored pair arrived, 160 of 160.
- Rest map: the rule went on all 8 traps and was brought back on all 8. No other display had a wrong go.
- Map that held: no wrong go, but it gave up all 16 edge goals. Only the 8 inner goals were reached.
- Current-state map: no wrong go, 14 of 16 edge goals reached. The 2 it gave up are the 2 known from D64 (pairs 67
  and 268). Under the rest map and with no map both were reached.
- No map: 16 goes refused, 8 on traps and 8 on outer goals, nothing brought back.
- Worst tilt on a test goal 2.02 deg (rest map), 2.04 deg over all rows.

Time:

- A scored pair, both moves, without the 30 s: median 11.6 simulated s, 4.4 to 47.9, and 93.0 for one pair that
  ran into the time-out. By display, median: no map 9.0, map that held 10.4, rest map 10.8, current-state map 18.8.
- The current-state map made the rule wait 9.8 to 15.3 s for a cage, median 12.0, on 52 goals, 10 processes
  (`map_s`). That wait is in its time. No send was made on an old cage (`early` 0).
- One time-out: id 2, trap pair 155, current-state map. The rule sat 90 s without a go and without giving up, so
  the row has `timeout` and `parked` true and 120 s. The cage came in 12.7 s, so it was not the map being late.
  Believed: the dot kept creeping along the cage, so the rule's 2 s stuck test never fired. Not checked. Other
  jobs ran on the machine during that block (a LaTeX build and two small scripts).

What it does not say: how people do. The rule does not plan, hesitate or learn, and it never goes on a goal its
map says no to. So the split between the displays is the one the sets were built to allow, the times are a floor,
and the run is a check that the tool logs what the analysis needs. `tlx` and `trust` are empty in these rows.

## D67 Pilot sessions get their own folder and a clock (2026-10-12)

`python src/teleop.py --maps <id> --dry` runs the session a participant gets and saves it to
`data/pilot/maps_<id>.csv`. `feedback.py` reads only `data/participants`, so a pilot can not end up in the analysis.

- `--dry` works for every study mode, it moves the folder all of them save to. `--pilot` was taken by the delay
  pilot.
- The tool stops if `maps_<id>.csv` is there already. Before, a second run with the same id wrote over the first
  one without a word.
- Two new columns, both on the wall clock. `run_s`: how long the drive the row came from took, window and map
  drawing included, 12 pairs in a block. `clock_s`: from the start of the tool to the end of that block's ratings.
  The tool prints both after each block. Everything else in a row is still simulated time.
- Not on that clock: consent and the sheet before the tool is started (5 min in the plan) and the close after the
  last rating (3 min). Add them by hand or use a stopwatch.
- What the pilots are for, same rule as in the journal draft: two pilots, not data. If either runs over 65 min,
  every block is cut to 8 scored pairs, 2 traps, 3 edge, 1 inner and 2 outer.

Checked: `tests/test_maps.py::test_dry_run_is_kept_out_of_the_participants`, the file a session would write is under
`data/pilot` with `--dry` and under `data/participants` without. Suite 24 of 24.
Not checked: nobody has driven a session by hand yet. The stored scripted files do not have the two columns.
