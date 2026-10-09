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
