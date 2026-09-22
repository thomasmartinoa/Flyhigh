# 05 — flybody: the brain in an anatomical fly, flying on its own wings

M3's body was a brick with a velocity controller. M4 swaps in Janelia's
[flybody](https://github.com/TuragaLab/flybody) (Vaxenburg et al. 2025): a MuJoCo fruit fly at
real scale — 2.9 mm, 0.98 mg, in centimetres and with air of the right density — whose wings
beat at 218 Hz from a measured pattern and whose flight is held together by a trained
controller. The brain's `MotorCommand` now steers *that* controller; nothing between the
pixels and the command changed since M2.

```
fly-scale drum ─► cubemap render (head, 5 exposures) ─► FlyAgent.tick ─► MotorCommand ─► carrot ─► flybody policy ─► wing beats ─► MuJoCo fluid
```

## 1. What flybody provides, and what we use of it

flybody ships the body, a **wing-beat pattern generator** (WBPG: recorded wing kinematics
replayed at a frequency the controller modulates), tasks built on `dm_control.composer`, and
**trained policies** as TensorFlow SavedModels. The flight policy is a trajectory tracker: its
inputs are proprioception (25 joint positions and velocities, accelerometer, gyro, velocimeter,
gravity direction) plus a **steering command** — the next six reference root poses, 0.2 ms
apart, in the fly's own frame (`ref_displacement (6, 3)`, `ref_root_quat (6, 4)`) — and its
output is 12 numbers: eleven non-wing actuators and the wing-beat frequency. On its own
flight-imitation task it tracks a recorded saccade with mean error 0.016 cm at reward 0.90;
on synthetic references it hovers (error 0.018 cm), flies straight at 20 cm/s and turns at
3 rad/s (0.010 cm). It loads under TF 2.20 once tfp's legacy type-spec names are aliased
(`flybody/policy.py`); it runs on the CPU while the brain runs on the GPU.

flybody's own "controller reuse" feeds the steering from a learned high-level network. Ours is
hand-written (`flybody/task.py`, `SteeredFlight`): a **carrot** — a reference point that
moves at the commanded velocity (`forward` → 30 cm/s, `lift` → 15 cm/s at full scale) and a
reference heading that turns at the commanded rate (`yaw` → 180°/s), both kept within reach of
the fly (one body length, 30°). The six poses are the carrot's next six positions and the
canonical flight attitude (body pitched 47.5° nose-up, as flies hover) at the reference
heading. `escape` moves the carrot up at 20 cm/s and back at 10 for 100 ms: a 2 cm hop in
100 ms (seven body lengths). The first version steered by velocity alone — the command as the
displacement, the gap always reported as zero — and the fly sagged to the floor within a
second: the policy was trained to *close a gap*, and needs to see one.

## 2. The fly's eyes, the fly's jitter

The M3 cubemap rig (five 90° cameras) sits on flybody's `head` body. The head mesh's frame is
arbitrary (its x-axis points to the fly's right, y forward-up), so after compilation the
cameras are re-oriented from the measured head↔level-heading rotation, and the panorama
lookup gets the same rotation; a hand placed at six bearings lands within 2° in the panorama.
The fly's own geoms (mesh, collision, fluid groups) are hidden from its eyes, as in M3.

Two things the brick never showed:

- **flybody renders for film** — a 4K offscreen buffer and 8192² shadow maps for four lights —
  and every eye face cost 33 ms even at 32 px. Shadows and reflections are off for eye renders
  (a 5° eye cannot see them): 4 ms per tick.
- **The head bobs ~2° with every wing beat.** One snapshot per 10 ms tick samples a 218 Hz
  oscillation and aliases it into motion; the hovering fly saw a jittering world and escaped
  from it (7 escapes in a second, tumbling). Photoreceptors integrate over about a tick, so the
  panorama is now the average of five exposures spread over the tick. Hover: 0 escapes.
- **Yaw authority.** At a saccade-like 720°/s, the readout's ±0.02 noise became a ±15°/s wobble
  the policy executed within milliseconds, and the fly's own wobble swamped any optomotor
  signal. 180°/s, as in M3.

The drum is the M3 room at fly scale: 30 × 30 × 15 cm, stripes of 7.5 cm period (28° at the
wall) on the side walls, plain floor, ceiling and end walls; the four walls are one mocap body
that can be **spun around the fly** — the physiologist's optomotor drum — because the fly's
own controller cancels an imposed angular velocity within one 0.2 ms control step, so M3's
disturbance cannot be used. The hand is a 4 cm sphere (a fingertip) at 50 cm/s.

## 3. The experiments

`uv run python scripts/validate_fly.py`, 2026-09-21, RTX 3070 Ti Laptop (brain) + CPU (policy):

```
[steer]     hover (policy alone): dz +0.00 cm, |roll| max 1°
[steer]     forward 0.5 -> 15.0 cm/s (target 15.0); yaw 0.25 -> -39°/s (target -45); lift 0.5 -> 7.5 cm/s (target 7.5); escape hop +2.01 cm
[hover]     brain in the fly: z 7.0..8.8 cm, |roll| max 6°, |pitch| max 8°, escapes 0, GF max 25 Hz
[hand]      run 0: escapes 40, first at 7.40 cm (hand 65°), rise within 200 ms 4.31 cm, closest 6.0 cm (contact 4.3) -> ok
[hand]      run 1: escapes 54, first at 7.39 cm (hand 65°), rise within 200 ms 4.29 cm -> ok
[hand]      run 2: escapes 38, first at 7.90 cm (hand 61°), rise within 200 ms 4.24 cm -> ok
[optomotor] drum +60°/s: yaw command mean -0.11 (with the drum in 0% of ticks), HS L/R 4/1 Hz, escapes 14
[optomotor] drum -60°/s: yaw command mean -0.15 (with the drum in 22% of ticks), HS L/R 13/2 Hz, escapes 29
    -> FAIL: optomotor: yaw command follows the drum, both ways
[speed]     4.0 ticks/s (brain + 50 policy steps + 200 physics steps + 5 renders per tick)
4/5 checks pass
```

- **Steering.** The policy does what the command says: 15.0 cm/s forward, 7.5 cm/s climb, a
  2 cm hop; yaw 39°/s against 45 commanded (the ramp). Hover holds to 0.0 cm.
- **Hover with the brain.** Height within 2 cm over 1.5 s (the VS-driven `lift` drifts it up),
  attitude within 8°, no escape.
- **The hand.** Three of three: the fly hops 4 cm when the fingertip subtends 61–65° — later
  than M3's 40–47°, because at fly scale the same M2 detectors see a fingertip that is 25
  radii per second instead of 7, and the escape comes when it is 7.4–7.9 cm away, 2 cm before
  the hand stops. Then it keeps hopping (40–54 escape ticks) until it hits the ceiling.
- **Optomotor: fails.** With escapes disabled the spinning drum does drive HS (up to 60 Hz)
  with the right laterality, and the yaw command follows the drum for +30°/s — but at other
  speeds DNa and giant-fiber activity from the same stimulus contaminate the command, and with
  the readout as it is the rotating drum fires the giant fiber (14–29 escape ticks) before HS
  build up; once the fly hops the measurement is over. A ramped spin-up does not help.
  Every drum that carried more HS signal (taller, or striped end walls: T4 ×3) fired the giant
  fiber during plain hovering. This is M2's marginal giant fiber again, now driven also by the
  fly's residual jitter.
- **Speed.** 4 ticks/s: 50 policy evaluations per tick on the CPU (1.1 ms each), 200 physics
  steps, five renders, and the brain.

## 4. Two flies (M4b)

flybody's tasks hold one walker, so two flies needed a task of our own
(`flybody/multi.py`): each fly gets flybody's flight configuration (wing gains, the fluid
model, retracted legs, and — see below — MuJoCo's mass bounds switched off), its own wing-beat
generator, its own carrot, and its own head cameras; the task maps a concatenated action vector
onto each fly's actuators itself, because flybody's `apply_action` writes the whole control
vector with walker-local indices. The flies tell each other apart from themselves by geom
group. One brain tick is one batched policy call for both flies.

Two things had to be found on the way, and both are worth remembering:

- **The flies were 45 % overweight.** flybody's own `Flying` task switches off MuJoCo's default
  lower bounds on mass and inertia (`compiler boundmass/boundinertia`); a task that forgets
  compiles a 1.43 mg fly with 10 µg wings instead of 0.98 mg and 8 µg. The trained policy still
  flies it — but it works harder, the head shakes more, and the brain sees motion that is not
  there: 4–12 escape ticks per second of *hovering*, at up to 40° of roll, in half the trials.
  With the bounds off, hovering is 0 escapes and |roll| ≤ 3°. A heavy fly is a jittery fly is a
  frightened fly.
- **A fly you start watching is already flying.** Each episode begins at a random wing-beat
  phase, and the first ~0.2 s are a transient. `FlySimulation(warmup_s=0.3)` lets the policy
  fly alone before the brain is connected (M4's `settle` only fed the brain a still frame; the
  physics was not running). `reset()` starts a fresh trial on the same model, brain and policy —
  necessary as well as cheap: `dm_control` recompiles the model on reset, so the renderer,
  cameras and mocap ids are rebound (and six brains in one process is an out-of-memory kill).

`uv run python scripts/validate_fly2.py`, 2026-09-22:

```
[hover x2]  run 0 fly 0: z 7.0..8.5 cm, |roll| max 3°, escapes 0, GF max 25 Hz -> ok
[hover x2]  run 0 fly 1: z 7.0..8.2 cm, |roll| max 2°, escapes 0, GF max 25 Hz -> ok
[hover x2]  run 1 fly 0: z 7.0..13.8 cm, |roll| max 55°, escapes 10, GF max 100 Hz -> x
[hover x2]  run 1 fly 1: z 7.0..8.1 cm, |roll| max 3°, escapes 0, GF max 25 Hz -> ok
[hover x2]  run 2 fly 0: z 7.0..11.9 cm, |roll| max 47°, escapes 6, GF max 175 Hz -> x
[hover x2]  run 2 fly 1: z 7.0..8.0 cm, |roll| max 3°, escapes 0, GF max 0 Hz -> ok
[speed]     two flies, two brains: 2.8 ticks/s
    -> FAIL: two flies hover with their brains, no escape (4/6 fly-trials clean)
[hand]      run 0: fly 0 escapes 41, first at 9.57 cm (hand 49°), rise 2.09 cm | fly 1 escapes 0 (hand stays 18.3 cm from it) -> ok
[hand]      run 1: fly 0 escapes  4, first at 9.25 cm (hand 51°), rise 1.43 cm | fly 1 escapes 0 -> ok
[hand]      run 2: fly 0 escapes  9, first at 8.69 cm (hand 55°), rise 1.43 cm | fly 1 escapes 0 -> ok
    -> ok: hand → fly 0 escapes before contact, fly 1 never (3/3 runs)
[fly-by]    visible: closest 2.0 cm (the other fly subtends ~17°); escapes 0, GF max 0 Hz, |yaw| max while < 3 cm 0.0
[fly-by]    ghost:   closest 1.9 cm (the other fly subtends ~18°); escapes 0, GF max 0 Hz, |yaw| max while < 3 cm 0.0
2/3 checks pass; failed: ['two flies hover with their brains, no escape (4/6 fly-trials clean)']
```

- **Two brains, two flies, 2.8 ticks/s** (two brains, one batched policy call for both flies and
  two cubemaps per tick). In four of six hovering fly-trials the fly holds its height within
  1.5 cm and its attitude within 3°; in the other two it emits a stray escape, and *that* is
  what breaks it: the hop rolls the fly 47–55°, the rolled view sweeps across the eye, the
  giant fiber fires again, and it takes a second to settle. It is the same self-motion →
  escape loop as M3 §5 and M4 §4 — one stray spike is enough because the escape is a 100 ms
  full-authority manoeuvre, and nothing in the readout suppresses the next one. The check is
  left as written; the rate (a third of fly-trials) is the result.
- **The hand.** It waits on fly 0's side of the drum and visits fly 0 only; fly 0 escapes when
  the fingertip subtends 49–55° and rises 1.4–2.1 cm, fly 1 never (3/3 runs; the hand stays
  18 cm away from it). Getting this to
  mean anything took moving the flies 16 cm apart: at 8 cm the 4 cm hand *engulfed* fly 1 on its
  way to fly 0 (`hand_dist` 3.4 cm against a 4 cm radius), and fly 1 escaped from being inside
  a black ball — a geometry bug that read exactly like a brain result.
- **Two flies do not react to each other, and should not.** In the fly-by, fly 1 crosses 2 cm in
  front of hovering fly 0: it subtends ~15°, no escape, no yaw, and the ghost control is
  identical. At real fly scale a 3 mm fly reaches this brain's escape size (~40°, docs/03 §5)
  only at 0.4 cm — contact. M3's flying bricks escaped each other at half a metre because they
  were 30 cm across; a fly is not. To get fly-to-fly interaction one has to model what real
  flies use at these distances: LC10/LC11 small-object fixation, not the giant fiber.

## 5. Honest limits

- The optomotor reflex, which works in M2 (stimulus) and M3 (brick), is not demonstrated in
  the anatomical fly; the reason is the giant fiber's margin, not the wings.
- The escape hop is a steered manoeuvre through the flight controller, not a fly's take-off
  or the tumbling backward flip real flies perform.
- Two flies per world (M4b); more is only bookkeeping (geom groups) but 2.1 ticks/s already.
  The two flies ignore each other for the geometric reason above, not because anything broke.
- Legs are retracted (flybody's flight configuration); walking is another policy entirely.
- flybody's policy expects flybody's own timestep and control rate; our brain tick is 50 of its
  control steps, so commands change 100 times a second — the fly reacts to a command in
  ~50 ms, faster than the brain (~100 ms).
