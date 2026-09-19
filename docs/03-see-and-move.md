# 03 — See & move: eyes and a motor output for the brain

M1 gave the connectome a spiking brain. M2 puts a picture of the world in front of it and reads
motor commands out of the back, so that a video goes in and `MotorCommand(forward, yaw, lift,
escape)` comes out. Two reflexes are the yardstick: the **optomotor turn** (the fly turns with a
rotating scene) and the **looming escape** (an expanding shadow fires the giant fiber). The first
works end-to-end; the second does not yet, and the last section says exactly why.

```
world frame ─► PanoramicFrame ─► EyeSampler ─► flyvis ─► FlyvisBridge ─► LIFBrain ─► Readout ─► MotorCommand
              (360×180 lum)    (2×721 hexals)  (64 types)  (ext_i, mV)   (M1, 165k)  (DN rates)
```

One `FlyAgent.tick(frames)` is one 10 ms camera frame: sample both eyes, advance flyvis one
step, hold its activity as currents for 100 LIF steps, count descending-neuron spikes over the
last 20 ms, apply four hand-written channels. Agents are the batch dimension all the way through.

## 1. How a fly sees

A fly's compound eye is ~750 **ommatidia** per side, each a lens plus eight photoreceptors,
looking ~5° apart and covering, between the two eyes, almost the full sphere. The retina does
not deliver an image but a hexagonal lattice of luminance samples, one per **column**, and the
optic lobe keeps that lattice: lamina, medulla, lobula and lobula plate are stacks of ~750
repeated columnar micro-circuits. `EyeGeometry` places flyvis's 721-column hexagonal patch
(extent 15, 5° spacing) on the sphere at ±65° azimuth, and `EyeSampler` reads each column as a
Gaussian acceptance cone (5° half-width) over a `PanoramicFrame` — an equirectangular 360×180
luminance map that any source can fill: synthetic stimuli now, a MuJoCo render in M3, a camera
in M5.

Two things happen on the way down that the M1 brain cannot do. In the lamina the signal splits
into an **ON** pathway (L1 → Mi1, Tm3 …: things getting brighter) and an **OFF** pathway
(L2/L3 → Tm1, Tm2, Tm4, Tm9: things getting darker). In the medulla and lobula, **T4** (ON) and
**T5** (OFF) become **direction selective**: four subtypes each, a/b/c/d = front-to-back,
back-to-front, up, down, computed by comparing a column with its neighbour after a delay. All
of this happens in graded, non-spiking cells with time constants a uniform 1.8 ms LIF delay
cannot express. Everything downstream is spatial pooling: **HS** cells sum front-to-back T4/T5
across a whole eye (so they see yaw rotation), **VS** cells sum vertical motion (pitch/roll),
and **LC4 / LPLC2** sum outward-moving edges around a point (so they see expansion — a looming
object) and synapse directly onto the giant fiber `DNp01`.

## 2. Why flyvis sits in front of the LIF

[flyvis](https://github.com/TuragaLab/flyvis) (Lappalainen et al., *Nature* 2024) is a
connectome-constrained, task-trained model of exactly those 64 optic-lobe cell types on a
721-column eye; it reproduces the measured T4/T5 tuning. We run ensemble member `flow/0000/000`
online, one 10 ms step per tick with persistent state (`FlyvisEye`; the public `simulate()` is
offline-only). The **ownership rule**: every LIF neuron of a type flyvis models loses its
incoming synapses (`LIFBrain(driven_only=…)`) and spikes only from the current the bridge
injects; everything else — LC4, LPLC2, HS/VS, the descending neurons, the VNC — keeps the real
wiring. No neuron is computed twice.

Wiring the two together means saying which male-CNS neuron sits in which flyvis column
(`ColumnAlignment`). Three things had to be discovered on the way (`scripts/build_alignment.py`):

- **Only 15 of 65 flyvis types carry column coordinates** (`hex1/hex2`) in male-CNS v1.0; T4/T5
  have none. But a columnar neuron gets most of its synapses from partners in its own column
  (a T4a takes 99 % of its Mi1 input from one Mi1), so `columns.infer_columns` gives every
  unlabelled neuron the column that wins the synapse-weighted vote of its labelled partners,
  propagated for a few rounds: 44,171 neurons of 60 types, median vote share 0.35.
- **The two lattices are written in different bases.** Between annotated cells, synapses cross to
  neighbours at (±1,0), (0,±1) and ±(1,+1) in the male CNS, but at ±(1,−1) in flyvis's axial
  hex; `mcns_to_axial` flips `hex2`. That alone lifted the annotated types' coverage from 71 %
  to 99 % because the eye outline finally matched the disc.
- **Counting matched columns cannot choose among the 12 hex symmetries** (flyvis's eye is a
  symmetric disc, all 12 tie). The *direction* of known synapses can: Mi9 sits one column to the
  preferred side of the T4 it feeds. `choose_symmetry` compares the Mi9/Mi4→T4a–d and
  Tm9/Tm4/Tm2→T5a–d offsets in flyvis (`FlyvisEye.edge_offsets`) with the connectome's
  (`columns.partner_offsets`): the reflection `('x','y')` wins on both eyes with an rms
  mismatch of 0.55 columns against 0.77 for the runner-up.

Result: 61,503 (flyvis neuron, LIF neuron) pairs, 56,218 driven LIF neurons. Coverage per
flyvis type (fraction of its 721 cells with a partner on the right eye):

| coverage | types |
|---|---|
| 0.97–0.99 | L1 L2 L3 L5 Mi1 Mi4 Mi9 T1 Tm1 Tm2 Tm4 Tm9 Tm20 C2 C3 |
| 0.85–0.90 | T4a–d T5a–d T3 Tm3 |
| 0.7–0.83 | T2 T2a L4 R7 R8 TmY5a Mi15 |
| 0.2–0.7 | TmY18 Mi2 Mi13 Tm5Y TmY3 TmY9 Tm5a–c R1–R6 TmY4 TmY10 TmY13 TmY14 Mi10 Tm16 Lawf1/2 |
| < 0.2 | Mi14 TmY15 Tm30 (the male CNS has far fewer of these than columns) |
| 0 | Am, CT1 (one giant cell per side, not columnar — left to the LIF); Mi3 Mi11 Mi12 Tm28 (no such type in v1.0) |

One last orientation was fixed by physiology rather than by reading the code: with the naive
reading of flyvis's `hex_to_pixel`, a front-to-back grating on the right eye excited flyvis's
T4b and a downward one its T4c — the opposite of Maisak et al. 2013. `EyeGeometry` turns the
lattice by 180° so that T4a = front-to-back and T4c = up; a test pins all three measured
preferences. Without it HS would have been fed the wrong subtype and the optomotor sign would
have come out reversed.

## 3. The bridge and its calibration

`FlyvisBridge`: `ext_i[agent, lif] = gain[type] · max(activity − rest, 0)`, with `rest` the
per-neuron steady state after 2 s of grey (so a grey screen injects nothing — measured
< 1e-4 mV) and the sum over a column's flyvis cells weighted 1/k when several stand for one LIF
neuron (the six R1–R6). `scripts/calibrate_bridge.py` sweeps one gain for all types on a 60°/s
grating:

| gain | T4/T5 mean | T4/T5 p90 (driven) | LC4/LPLC2 | HS | GF | active neurons |
|---|---|---|---|---|---|---|
| 5 | 0.3 Hz | 0 | 0 | 24 Hz | 0 | 3,803 |
| 10 | 2.2 | 12 | 0 | 76 | 0 | 9,864 |
| 20 | 7.7 | 32 | 0.5 | 143 | 13 | 28,240 |
| 40 | 18.1 | 60 | 4.4 | 206 | 53 | 44,867 |
| 80 | 33.5 | 96 | 16.1 | 228 | 121 | 53,040 |

Grey gives 0 Hz everywhere at every gain. The plan's target was T4/T5 at 50–100 Hz (their
measured range), i.e. gain ≈ 40. That is not the gain we ship, because the checks bound it
from above: at gain ≥ 18 a plain grating already fires the giant fiber (a false escape, which
also zeroes `yaw`), and HS saturate near 200 Hz — refractory-limited — which compresses their
left/right difference to a few per cent. Choosing by effect, as M1 chose `w_syn`:

| gain | optomotor yaw cw / ccw | false-escape ticks (cw, ccw, receding) | T4/T5 p90 |
|---|---|---|---|
| 15 | +0.85 / −0.88 | 0, 0, 0 | 20 Hz |
| 18 | +0.43 / −0.50 | 0, 1, 0 | 26 |
| 20 | +0.31 / −0.27 | 0, 5, 0 | 30 |
| 30 | +0.08 / −0.07 | 8, 11, 3 | 44 |
| 40 | +0.01 / −0.02 | 26, 21, 2 | 56 |

**`DEFAULT_GAIN = 15`.** The LIF's tangential cells and giant fiber are far more excitable than
their real counterparts; the T4/T5 themselves end up at a fifth of physiological rates.

## 4. Descending neurons: the bottleneck and the four channels

The brain reaches the body only through ~1,300 descending neurons of a few hundred types —
a bottleneck, and the natural place to read a motor command. `Readout` reads eight of them,
resolved once by type on the real connectome (all present: 1 of each per side; VS matches
VS/VST1/VST2/VSm, 34 cells), from rates over the last 20 ms:

| channel | neurons | rule |
|---|---|---|
| `escape` | DNp01 (giant fiber), DNp04 | GF > 50 Hz, or GF > 20 Hz and DNp04 > 50 Hz |
| `yaw` | DNa01/02 L,R; HSN/HSE/HSS L,R | `k_dn (R−L)/(R+L+ε) + k_hs (R−L)/(R+L+ε)`, clipped to ±1 |
| `lift` | VS | `k_vs · rate` |
| `forward` | DNp09 | `0.2 + k_fwd · rate`; 0 during escape |

The spec left the HS side→sign mapping to be verified rather than assumed. It verified: with
the eye oriented by T4 physiology (section 2), a clockwise scene (seen from above, i.e. moving
right-to-left in front of the fly) is front-to-back motion on the *right* eye, drives right-eye
T4a (55 Hz vs 20 Hz at gain 40), hence right HS more than left, hence `R − L > 0`, hence
`yaw > 0` = turn right = turn *with* the scene, which is the optomotor response. No sign flip
was needed and `k_hs = k_dn = 1` are untouched. DNa01/02 stay at ≤ 4 Hz for a grating in this
brain; the yaw signal is carried by HS.

## 5. Results

`uv run python scripts/validate_reflexes.py`, 2026-09-20, RTX 3070 Ti Laptop (8 GB):

```
[silence]   escape=False yaw=+0.000 fwd=0.200 lift=+0.000  DN spikes in 1 s=0
    -> ok: silence
[escape]    first escape tick=None  disc diameter then=None  GF rate at end=0 Hz  LC4/LPLC2 mean 0.0 Hz, max 0 Hz
    -> FAIL: escape before 40°
[receding]  escape after onset=False
    -> ok: no escape for receding disc
[optomotor] cw: mean yaw +0.889, fraction with the right sign = 1.00, escapes = 0
[optomotor] ccw: mean yaw -0.888, fraction with the right sign = 1.00, escapes = 0
    -> ok: optomotor cw > +0.3
    -> ok: optomotor ccw < -0.3
    -> ok: optomotor mirror within 20 %
[two agents] agent0 escape=False agent1 escape=False
    -> FAIL: two agents independent
[two agents] optomotor: agent0 (grating) yaw +0.846, agent1 (grey) yaw +0.000
[speed]     1 agent(s): 40.3 ticks/s
    -> ok: speed 1 agent(s) >= 10 ticks/s
[speed]     2 agent(s): 37.6 ticks/s
    -> ok: speed 2 agent(s) >= 10 ticks/s

7/9 checks pass; failed: ['escape before 40°', 'two agents independent']
```

**What works.** A rotating world produces a clean, mirror-symmetric optomotor turn of the right
sign through 721 columns × 2 eyes of flyvis, 56k driven neurons, the real T4/T5 → HS wiring and
a two-line readout, with the correct sign falling out of the anatomy. Two agents run as one
batch and do not leak into each other (the second two-agent line: the agent on grey does
nothing while its neighbour turns). The whole loop runs 4× real time.

**What does not: the looming escape.** The checks were not touched; the diagnosis is:

- flyvis sees the loom — ~550 T4/T5 cells active on the right eye by the end of the expansion,
  peak activity 1.5 against 1.8 for the grating — and the LIF T4/T5 follow.
- The signal dies at LC4/LPLC2. Over the 500 ms loom, LC4 receives +87k weighted excitatory
  spikes and −45k inhibitory; the grating gives +915k / −829k. The net is similar, but the
  loom's is spread evenly, so LC4 sits at −45.01 mV against a −45 mV threshold and only the ~5
  LC4 whose field covers the disc fire, at 10–20 Hz; LPLC2 never fires (it would need ~6× the
  drive). The giant fiber needs ~4 near-coincident LC4 spikes (50 synapses × 0.2 mV each, 7 mV
  to threshold); the loom yields 1–4 LC4+LPLC2 spikes per 10 ms in total.
- T2, LC4's largest input (46k synapses), contributes +476k for the grating and 0 for the dark
  loom: flyvis's T2 *hyperpolarises* for a dark stimulus and the bridge clamps at zero. The loom
  arrives through the OFF pathway (TmY3, Tm2, T5) instead.
- Gains cannot fix it: tripling the OFF pathway's gain moved LC4/LPLC2 from 0.2 to 0.5 Hz on the
  loom and made the grating false escape worse; dark-on-white contrast and a 250 ms loom change
  nothing.
- The root cause is the LIF's excitation/inhibition balance in the lobula. Real LC4/LPLC2 are
  loom-specific *because* wide-field inhibition wins for gratings; in this brain it cancels
  91 % of the excitation and the residual drives the giant fiber, while the loom's local drive
  is ~5× too weak. Shiu et al. 2024 showed the loom→GF circuit by activating LC4/LPLC2
  directly (M1's `validate_looming.py` reproduces that); reaching it from pixels needs an
  M1-level recalibration (an inhibition scale, re-validated on the gustatory and looming
  circuits) or a graded model of the lobula, not a bridge gain. That is the first item of M3.

**Honest limits.**
- The gain is chosen by the reflexes, not by T4/T5 physiology; T4/T5 fire at ~20 Hz, HS at
  ~50–140 Hz, both models of graded cells.
- The readout is hand-written; `k_hs`, thresholds and the 20 ms window are guesses that
  happened to work. `lift` and `forward` are untested against anything.
- Column inference places a multi-column neuron at the modal column of its partners (median
  vote share 0.35); a T4 may sit one column off. Am1/CT1 keep their LIF wiring, a deliberate
  break of the ownership rule.
- flyvis's T4d barely responds to anything in ensemble member 0; the other 49 members are
  untried.
