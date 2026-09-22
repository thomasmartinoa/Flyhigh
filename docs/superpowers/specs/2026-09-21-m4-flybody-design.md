# M4 — flybody: design

Date: 2026-09-21. Status: implemented the same day; decisions marked *(decision)* are the
author's; **as built** notes record what changed and why (details in `docs/05-flybody.md`).

## Goal

Replace the flying brick with Janelia's anatomically detailed MuJoCo fruit fly
([flybody](https://github.com/TuragaLab/flybody), Vaxenburg et al. 2025) *flying on its own
wings*: the brain's `MotorCommand` steers flybody's trained low-level flight controller, which
beats the wings; MuJoCo's fluid model turns the wing beats into lift. The eyes sit on the fly's
head. The M2 brain, bridge and readout are unchanged; the M3 room becomes a fly-scale drum.

```
fly-scale drum ─► cubemap render (head) ─► FlyAgent.tick ─► MotorCommand ─► steering ─► flybody policy ─► wing beats ─► MuJoCo (fluid)
```

## What flybody gives us (checked, 2026-09-21)

- The model is in **real fly units, cgs**: gravity −981 cm/s², body 0.29 cm, mass 0.98 mg, air
  density 1.28e-3 g/cm³. Physics timestep 0.05 ms, control 0.2 ms (in flight tasks).
- A **wing-beat pattern generator** (WBPG, from measured kinematics; base frequency modulated
  by one action) and a trained **flight policy** (`trained-fly-policies/flight`, a TF
  SavedModel; loads under TF 2.20 once tfp's legacy type-spec names are aliased) with inputs
  `joints_pos/vel (25)`, `accelerometer`, `gyro`, `velocimeter`, `world_zaxis` and the
  **steering** `ref_displacement (6, 3)`, `ref_root_quat (6, 4)` — the next six reference root
  poses in the fly's egocentric frame — and a 12-dim action (11 non-wing actuators + wing-beat
  frequency). On its own flight-imitation task it tracks a recorded trajectory with mean
  displacement error 0.016 cm at reward 0.90, at ~0.5× real time on this CPU.
- `eye_left`/`eye_right` cameras on the head (unused: our cubemap rig goes on the head instead).

## Steering *(decision)*

**As built:** a velocity-only command (displacement = k·dt·v, the gap always zero) let the fly
sag to the floor within a second; the steering is a *reference point* moving at the commanded
velocity, kept within a body length (and the heading within 30°) of the fly. `ω_max` = 180°/s
(720 turned readout noise into a wobble); escape 20 cm/s up / 10 back (40 tumbles the policy).


flybody's "controller reuse" replaces the reference observables with a steering command from
a high-level network. We do the same with a hand-written one: for command velocities
`v = (v_max·forward, 0, vz_max·lift)` (body frame) and yaw rate `ω = −ω_max·yaw`,

```
ref_displacement[k] = k·dt·v                              k = 0..5, dt = 0.2 ms
ref_root_quat[k]    = dquat_local(q_fly, yaw(ψ_fly + k·dt·ω) ∘ q_flight)
```

where `q_flight` is the canonical flight attitude (the recorded hover pose's roll/pitch) so that
a fly knocked off attitude is steered back level. `escape` overrides `v` for 100 ms with
(−escape_back, 0, escape_up). Because the policy was trained on real trajectories, commanded
speeds stay inside their range: `v_max` = 30 cm/s, `vz_max` = 15 cm/s, `ω_max` = 720°/s, escape
up 40 cm/s / back 20 cm/s *(decision; measured against the dataset's speed and turn
distributions in the validation)*.

## The world at fly scale *(decision)*

**As built:** stripes of 7.5 cm period (28°); the four walls are one mocap body that spins for
the optomotor experiment (the fly's controller cancels an imposed angular velocity); each tick's
panorama averages five exposures (the head bobs ~2° per wing beat); eye renders without shadows
(flybody's 8192² shadow maps cost 33 ms per face). The optomotor check is on the drum's
rotation, not a body kick; it fails (docs/05 §3).


A 30 × 30 × 15 cm drum (the M3 room ÷ 20, i.e. the same ~50 body lengths): stripes of 3.75 cm
period on the side walls (14° at 15 cm), plain floor/ceiling/end walls, one light. The hand is
a 4 cm-radius sphere (a fingertip) on a mocap body, approaching at 50 cm/s from ahead-right and
stopping 2 cm short. The cubemap eyes are five 90° cameras on the `head` body, hidden from the
fly's own geoms by geom group as in M3; rendered at 96 px.

One fly per environment *(decision)*: flybody's tasks are single-walker `dm_control.composer`
tasks and the policy machinery (WBPG, action mapping, observables) lives in them; reusing that
is the point of M4. Two flybody flies in one world is M4b.

## Code

- `flyhigh/flybody/policy.py` — `FlightPolicy`: loads the SavedModel (with the tfp alias), maps
  a `dict` observation to the action mean. TensorFlow CPU only.
- `flyhigh/flybody/task.py` — `SteeredFlight(Flying)`: the drum arena, the head cameras, the
  WBPG step from `FlightImitationWBPG`, `set_command(MotorCommand)`, the two steering
  observables computed from the command, no ghost, no time limit, episode reset to the
  recorded hover pose.
- `flyhigh/flybody/sim.py` — `FlySimulation`: the M3 `Simulation` API (`frames`, `step`,
  `run`, `log`, `hand`, `disturb_yaw`) over a composer environment; one brain tick = 50 policy
  steps = 200 physics steps.
- `scripts/validate_fly.py` — 1 hover (policy, no brain: height held for 1 s, attitude level),
  2 steering (forward/yaw/lift/escape each produce the commanded motion within 20 %),
  3 hand → escape with the brain (fly escapes before the hand's surface is within 1 cm, over a
  plain wall), 4 optomotor (imposed 200°/s yaw for 100 ms; brain command opposes), 5 speed.
- `docs/05-flybody.md`, `notebooks/05_flybody.ipynb`.

## Out of scope

Walking (the walking policy exists; legs are retracted in flight), learned steering, the vision
policies (they replace our brain). *(Two flybody flies were in scope for M4b and are done:
`flybody/multi.py`, `scripts/validate_fly2.py`, docs/05 §4.)*
