# 04 — The box: a brain in a body, in a room, with company

M2 fed the brain synthetic pictures and read commands off the back. M3 closes the loop through
physics: the pictures are rendered from where the body is, the commands move the body, and
what the body does changes the next picture. Two brains share the room with a moving "hand".

```
MuJoCo room ─► cubemap render ─► PanoramicFrame ─► FlyAgent.tick ─► MotorCommand ─► Body ─► MuJoCo step
   (physics)    (5 faces/agent)   (M2 interface)    (M2, batched)    (M2)         (forces)   (5 × 2 ms)
```

Nothing between the pixels and the command changed: the same brain, bridge, gains and readout
as M2. Everything in this milestone is the world around it.

## 1. What changes when a brain has a body

Two things a stimulus generator never shows you.

**Self-motion looks like world motion.** When the body yaws, the whole scene sweeps across
both eyes exactly as M2's rotating grating did — and the brain answers as before: it turns
*with* the scene. Since the scene moves opposite to the body's turn, that is a turn *against*
the disturbance. The optomotor reflex, which in M2 was "the fly follows the drum", is in a body
a **stabiliser** of heading: a negative-feedback loop closed through the world. (The sign was
never tuned; it falls out of the T4a → HS anatomy fixed in M2.)

**Flying forward paints an expanding picture.** Everything ahead grows; a textured floor or a
wall in front is, to a looming detector, a looming object. The first room had checkered walls
and floor and both flies escaped from it within 350 ms of take-off, with the other fly hidden
or not. Real flies solve this in the lobula plate (LPi cells suppress LPLC2 for wide-field
motion); this brain has that circuit, but its giant fiber is marginal (docs/03 §5), so the room
had to help.

## 2. The room is an optomotor drum

The fly physiologist's apparatus: vertical stripes on the side walls (rotation flow for HS,
0.75 m period ≈ 14° at 3 m — coarse enough for 5° columns), and plain grey floor, ceiling and
end walls so that translation paints nothing that expands. 6 × 6 × 3 m: at 4 m the flies' own
0.5 m/s cruise past the stripes still tripped the giant fiber in two runs out of three. The
only dark objects are the hand (a 15 cm-radius sphere on a mocap body) and the other flies
(30 cm boxes with cosmetic rotors: a 10 cm fly would reach the ~40° escape size only at
contact, a 30 cm one at 0.4 m).

## 3. Eyes and body

**Eyes** (`flyhigh.world.eyes`): five 90° pinhole cameras per agent — front, left, right, up,
down; the rear is outside both eyes' fields — rendered at 96 × 96 through `mujoco.Renderer`
and resampled into M2's 360 × 180 `PanoramicFrame` through a lookup derived from the cameras'
quaternions. A bright sphere at six bearings lights the expected panorama pixel *and* the
expected eye column (the M2 sampler), which is the whole chain checked. Each agent's own geoms
are in a geom group its cameras skip: with a near plane small enough to see a hand 5 cm away,
the fly otherwise sees its own rotors.

**Body** (`flyhigh.body.brick`): a *flying brick* — a free-floating box steered by body-frame
forces and torques from a velocity controller. `MotorCommand` names velocities: `forward` →
2.5 m/s × forward (so the readout's 0.2 bias cruises at 0.5 m/s), `yaw` → 180°/s × yaw (+ =
right, i.e. clockwise from above), `lift` → 0.5 m/s × lift, with altitude held when lift is
zero and roll/pitch held level; `escape` overrides everything for 100 ms with 2 m/s up and
1 m/s back. The brain never flies the aircraft, which is what a real drone's flight controller
(M5) is for; rotor-level dynamics can replace the class behind `apply()`.

**Hand** (`flyhigh.body.hand`): idles in a corner ahead-right of fly 0, approaches at 1 m/s
following its target, stops with its surface 20 cm from the fly, retreats.

**Loop** (`flyhigh.sim`): one tick = render both agents → `FlyAgent.tick` → apply commands
for 5 physics steps of 2 ms (the hand script advances too) → log position, attitude, command,
GF/HS rates, hand distance. A brain that woke up on grey startles at a room appearing (the M2
flash response), so a new `Simulation` first lets the brain look at the still room for 0.5 s.

## 4. The experiments

`uv run python scripts/validate_box.py`, 2026-09-20, RTX 3070 Ti Laptop. The giant fiber is
marginal and CUDA's `index_add` is not deterministic, so identical runs differ like trials on
a fly; the two stochastic experiments run three times and are judged by majority.

```
[speed]     1 agent(s): 14.5 ticks/s with rendering
[speed]     2 agent(s): 10.5 ticks/s with rendering
[hover]     agent 0: z 1.00..1.09  |roll| 0.0° |pitch| 0.0°  escapes 0  inside room True
[hover]     agent 1: z 1.00..1.05  |roll| 0.0° |pitch| 0.0°  escapes 0  inside room True
[hand]      run 0: agent 0 escapes 7, first at 0.375 m (hand 47°), rise 0.249 m | agent 1 escapes 0 | closest 0.35 m -> ok
[hand]      run 1: agent 0 escapes 2, first at 0.435 m (hand 40°), rise 0.193 m | agent 1 escapes 0 | closest 0.35 m -> x (rise < 0.2)
[hand]      run 2: agent 0 escapes 6, first at 0.445 m (hand 39°), rise 0.222 m | agent 1 escapes 0 | closest 0.35 m -> ok
[optomotor] imposed +30°/s: brain yaw command mean -0.21 (opposing sign in 23% of ticks)
[optomotor] imposed -30°/s: brain yaw command mean +0.18 (opposing sign in 20% of ticks)
[two flies] visible run 0..2: escapes [7,2] [8,6] [3,3], first at 0.49 / 0.52 / 0.50 m separation (contact 0.30)
[two flies] hidden  run 0..2: escapes [0,0] [0,0] [0,0]
6/6 checks pass
```

- **Hover.** With the forward bias off, both bodies hold 1 m within 9 cm (the small drift is
  the VS-driven `lift` channel), stay level and never escape.
- **Hand.** The hand comes at fly 0 from ahead-right over a plain wall; fly 0 escapes when the
  hand subtends 39–47° (M2's escape size was ~40°) and jumps 0.2–0.25 m; fly 1, which the hand
  does not visit, rarely escapes (0 of 3 validation runs, 0 of 7 in development, 3 escape
  ticks in one notebook run). Over the *striped* wall
  the same approach escaped in 1 run of 4: a dark ball over dark stripes loses half its edge.
- **Optomotor.** An imposed 30°/s rotation (re-applied every physics step for 500 ms) makes
  the brain command a turn the other way, both ways. It is weaker than in the 4 m room
  (−0.95/+0.92 there, 100 % of ticks) because the stripes fill less of the eye; at 60–90°/s
  flyvis's T4 fall off (temporal frequency).
- **Two flies.** Facing each other, cruising at 0.5 m/s each, at least one escapes when they
  are 0.49–0.52 m apart (the other fly subtends ~35°), never touching (closest 0.40 m, contact
  at 0.30). As ghosts — invisible and without contacts, everything else identical — nobody
  escapes in 2.5 s. The escape is to the other fly.
- **Speed.** 14.5 / 10.5 ticks/s for one / two agents with rendering (the brain alone runs
  ~38); the cubemap renders cost about as much as the brain.

## 5. Honest limits

- Everything that was marginal in M2 is marginal here: escapes come at 40–50°, one run in
  three misses a criterion by a hair, and the room had to be arranged (drum, size, plain
  walls, hover for the hand) so that the flies' own flight does not fire their giant fibers.
  A fly that cruises at a checkered floor escapes from the floor.
- The optomotor response is a *command*, not yet a closed heading loop worth measuring: the
  body's own rate loop damps a kick in ~70 ms, faster than the brain's ~100 ms latency, so
  the brain's contribution to heading stability is not visible in the trajectory.
- The body is a brick with a velocity controller; nothing about rotor physics is real.
- `lift` follows VS and makes the flies drift upward slowly; `forward` is a constant bias.
- Run-to-run variance is GPU nondeterminism, not modelled noise.
