# M3 — The Box: design

Date: 2026-09-20. Status: implemented the same day; decisions marked *(decision)* are the
author's. **As built** notes record where implementation overturned the draft and why
(details in `docs/04-the-box.md`).

## Goal

Put two fly brains into bodies in a room and let them react to the room, to a moving "hand"
and to each other, with nothing hand-placed between pixels and motor command:

```
MuJoCo room ─► cubemap render ─► PanoramicFrame ─► FlyAgent.tick ─► MotorCommand ─► Body ─► MuJoCo step
   (physics)    (5 faces/agent)   (M2 interface)    (M2, batched)    (M2)            (this milestone)
```

Deliverables: `flyhigh.world` (the MuJoCo scene and its eyes), `flyhigh.body` (command →
forces), `flyhigh.sim` (the loop, logging, video), `scripts/validate_box.py`,
`docs/04-the-box.md`, `notebooks/04_the_box.ipynb`. M2's brain, bridge and readout are not
changed by this milestone; if a reflex needs retuning inside a body, that is recorded as a
finding, not silently tuned.

## The room

**As built:** 6 × 6 × 3 m, an optomotor *drum* — vertical stripes (0.75 m period) on the side
walls only, plain grey floor, ceiling and end walls. Checkered walls and floor made the looming
detectors fire at the room itself when the flies flew forward, and at 4 m the flies' own cruise
past the stripes tripped the giant fiber in most runs.

*Draft:* A 4 × 4 × 3 m box *(decision: metres, not fly scale — the eye has 5° columns, so what matters is
angular size; a 10 cm drone at 1 m subtends 6°, a 30 cm hand at 50 cm subtends 33°)*. Walls,
floor and ceiling carry a high-contrast checker texture (period ~0.25 m ≈ 7° at 2 m) so that
self-rotation produces wide-field optic flow for the optomotor reflex. One directional light
plus ambient so no wall is black.

The **hand** is a 30 cm dark sphere on a `mocap` body driven by a script: it idles ahead-right of
fly 0 (**as built:** over a plain wall — over the striped wall the loom escaped in 1 run of 4),
then moves toward a chosen drone at 1 m/s *following it* (`Hand.approach(target)` takes a
callable), stops with its surface 20 cm short and retreats. This is the M2 looming disc made
physical.

## The bodies

Two **flying bricks** *(decision)*: a 30 cm box (**as built**; a 10 cm fly reaches the ~40°
escape size only at contact, a 30 cm one at 0.4 m) with four cosmetic rotor discs on a free joint,
mass 30 g, controlled by body-frame forces and torques (`data.xfrc_applied`) from a velocity
controller — not by rotor thrusts. Reason: M3 is about brains reacting, and a rotor-level
quadrotor needs an attitude controller that is a project of its own; a real drone (M5) brings
its own flight controller anyway. Gravity, inertia, drag and collisions are real MuJoCo.
`Body.apply(command)` is the interface a rotor model can later replace.

`MotorCommand` → targets *(decision, all in `BodyParams`)*:

| channel | target |
|---|---|
| `forward` ∈ −1..1 | body-frame forward speed `v_max · forward`, `v_max` = 2.5 m/s (**as built**; 0.5 m/s at the 0.2 bias so two flies loom at ~70°/s) |
| `yaw` ∈ −1..1 | yaw rate `ω_max · yaw`, `ω_max` = 180°/s, + = right |
| `lift` ∈ −1..1 | vertical speed `vz_max · lift`, `vz_max` = 0.5 m/s |
| `escape` | for 100 ms: override with up 2 m/s and backward 1 m/s, then resume |

Roll and pitch are held at zero by a PD torque; altitude holds at 1 m when `lift` = 0
(altitude PD, not velocity, so the bodies do not drift into the floor). The controller runs at
the physics rate (500 Hz, `timestep` 2 ms); the brain runs every 10 ms (5 physics steps).

## The eyes

Per agent, five 90° pinhole cameras attached to the body (front, left, right, up, down; the
back is outside both eyes' fields), rendered at 96 × 96 with `mujoco.Renderer`, grey-converted
and resampled into the 360 × 180 `PanoramicFrame` through a precomputed lookup table
(`CubemapPanorama`: for every panorama pixel, which face and which pixel). Pixels the five faces
do not cover (the rear 90°) are grey (0.5). The cameras sit at the body origin: MuJoCo's camera
frame looks along −z, so the front camera's `xyaxes` are set such that the panorama's azimuth 0
is the body's +x and elevation 0 its horizon. Verified by rendering a known bright sphere at a
known bearing and checking the lit panorama pixel and the lit eye column (M2's `EyeSampler`).

Rendering backend: `MUJOCO_GL=glfw` on this desktop (EGL is not usable here); the `Renderer`
is created once per process and reused.

## The loop

`Simulation(connectome, n_agents=2, seed=0)` owns model, data, bodies, hand, `FlyAgent` and a
log. `step()` = render both agents' panoramas → `agent.tick(frames)` → `body.apply(cmd)` × 5
physics steps (the hand script advances too). `run(seconds)` loops it and returns the log:
per tick, each agent's position, yaw, command, GF/HS rates, and the hand position. Optional
`VideoWriter`: a third-person camera on the room plus one agent's panorama, written with
`imageio` to mp4 at 25 fps (every 4th tick).

Agents start hovering at 1 m, 1.5 m apart, offset 0.4 m sideways, facing each other. A new
`Simulation` lets the brain look at the still room for 0.5 s first (**as built**: a brain that
woke on grey startles at a room appearing). `hide_agent` makes a fly a ghost — invisible *and*
without contacts — for control runs.

## Validation (`scripts/validate_box.py`)

**As built:** the hover and hand experiments run with the readout's forward bias off (the
flies hover; everything else is the brain); the hand check is "escapes during the approach,
before contact, and only the targeted fly" (the escape comes at 39–47°, M2's marginal size);
the optomotor disturbance is a sustained 30°/s for 500 ms (a 100 ms kick is damped by the body's
own rate loop before the brain answers) and the criterion is the sign of the mean command;
the two-fly experiment runs 2.5 s (the encounter; later the flies reach the end walls, whose
corners loom) with ghosts as the control; the hand and two-fly experiments run three times and
pass by majority (CUDA nondeterminism makes identical runs differ like trials on a fly).

1. **Hover.** 3 s, hand idle, brains connected: both bodies stay within 0.3 m of their start
   height, inside the room, and roll/pitch stay < 5°. (The brain's forward bias 0.2 means they
   drift forward at 0.2 m/s; the test measures height and attitude, not position.)
2. **Hand → escape.** The hand approaches agent 0 from its right (where M2's disc was). Agent 0
   escapes (position jumps ≥ 0.3 m up/back within 200 ms of the first `escape`) before the hand
   is 40 cm away; agent 1, which the hand does not approach, does not escape.
3. **Optomotor in the body.** With agent 0 given an external yaw kick of 90°/s for 100 ms, the
   brain's `yaw` command over the next 300 ms has the sign that opposes the kick (the scene
   rotates the other way; the fly turns with the scene, i.e. back). Reported as the mean command
   and its sign; required: sign correct in ≥ 80 % of ticks.
4. **Two flies.** Agents fly toward each other (forward bias, facing); at least one escapes
   before they collide, and neither escapes in the same run with the other agent removed from
   the scene (so the escape is to the *other fly*, not the walls).
5. **Speed.** ticks/s for 1 and 2 agents including rendering; target ≥ 8 ticks/s.

Unit tests (no data): cubemap lookup geometry (a sphere at bearing (az, el) lights the right
panorama pixel), body controller (converges to target speeds, holds altitude, escape override
timing), hand script kinematics, simulation loop with a fake agent.

## Out of scope

Rotor-level flight dynamics; learned control; object fixation (LC10/11); sound; M4's flybody.

## Learning track

`docs/04-the-box.md`: what changes when a brain has a body (closed loop, self-motion vs world
motion, why the optomotor reflex is a stabiliser); the eye rig; the body layer as a contract;
the four experiments with numbers; limits. `notebooks/04_the_box.ipynb`: the room from above
and through a fly's eye, a video of the hand experiment, traces of escape and yaw.
