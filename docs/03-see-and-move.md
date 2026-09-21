# 03 — See & move: eyes and a motor output for the brain

M1 gave the connectome a spiking brain. M2 puts a picture of the world in front of it and reads
motor commands out of the back, so that a video goes in and `MotorCommand(forward, yaw, lift,
escape)` comes out. Two reflexes are the yardstick: the **optomotor turn** (the fly turns with a
rotating scene) and the **looming escape** (an expanding shadow fires the giant fiber). Both
work end-to-end — the second only after three bugs and two calibrations that section 5 tells
in order, because finding them was most of the milestone.

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
  (`columns.partner_offsets`): `('-x','-y')` wins on both eyes with an rms mismatch of 0.55
  columns against 0.77 for the runner-up. (flyvis stores `du` as target − source; reading it
  the other way round once picked the 180°-rotated symmetry — see section 5.)
- **The anatomy checks the answer.** LPLC2 is a looming detector because each of its T4/T5
  inputs prefers motion *away* from the cell's receptive-field centre. `columns.radial_index`
  computes that through an alignment: +0.68 for `('-x','-y')`, −0.68 for its 180° twin, ~0 for
  the other ten. `build_alignment.py` prints it every build.

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

## 3. The bridge, the brain, and their calibration

`FlyvisBridge`: `ext_i[agent, lif] = gain[type] · max(activity − rest, 0)`, with `rest` the
per-neuron steady state after 2 s of grey (so a grey screen injects nothing — measured
< 1e-4 mV) and the sum over a column's flyvis cells weighted 1/k when several stand for one LIF
neuron (the six R1–R6). `scripts/calibrate_bridge.py` sweeps one gain for all types on a 60°/s
grating (M1 brain preset, before the changes below):

| gain | T4/T5 mean | T4/T5 p90 (driven) | LC4/LPLC2 | HS | GF | active neurons |
|---|---|---|---|---|---|---|
| 5 | 0.3 Hz | 0 | 0 | 24 Hz | 0 | 3,803 |
| 10 | 2.2 | 12 | 0 | 76 | 0 | 9,864 |
| 20 | 7.7 | 32 | 0.5 | 143 | 13 | 28,240 |
| 40 | 18.1 | 60 | 4.4 | 206 | 53 | 44,867 |
| 80 | 33.5 | 96 | 16.1 | 228 | 121 | 53,040 |

Grey gives 0 Hz everywhere at every gain. The plan's target was T4/T5 at 50–100 Hz (their
measured range), i.e. gain ≈ 40. That is not the base gain we ship, because the optomotor
check bounds it from above: HS cells receive ~20k synapses from T4a/T5a (28 per cell, 5.6 mV per
spike), so above gain ≈ 18 any coincidence of a few spikes fires them, Shiu's `g = 0` reset then
wipes the inhibition that had accumulated, and they sit near 200 Hz in *both* directions
(null-direction HS: +448k weighted excitatory spikes, −634k inhibitory, 191 Hz). Selectivity
survives only when T4a/T5a fire sparsely. Choosing by effect, as M1 chose `w_syn`:

| gain (all types) | optomotor yaw cw / ccw | false-escape ticks (cw, ccw, receding) | T4/T5 p90 |
|---|---|---|---|
| 15 | +0.85 / −0.88 | 0, 0, 0 | 20 Hz |
| 18 | +0.43 / −0.50 | 0, 1, 0 | 26 |
| 20 | +0.31 / −0.27 | 0, 5, 0 | 30 |
| 30 | +0.08 / −0.07 | 8, 11, 3 | 44 |
| 40 | +0.01 / −0.02 | 26, 21, 2 | 56 |

The looming detectors want the opposite: LPLC2 needs its OFF-edge inputs (T5b/c/d, ~3 synapses
per cell) at 100+ Hz. The two reflexes read almost disjoint T4/T5 subtypes (HS ← T4a/T5a;
LPLC2 ← T5b/c/d, T4c/d), so the bridge ships `DEFAULT_GAIN = 15` with
`DEFAULT_GAINS = {T5b, T5c, T5d: 120}`. That still did nothing for the loom until two things
in the *brain* changed (`LIFBrain.for_male_cns`; M1's central calibration is untouched and
its two validation scripts still pass):

- **Short-term depression is off inside the optic lobe.** M1 added STD (u = 0.2, τ = 300 ms) to
  stop the male-CNS graph igniting, exempting sensory neurons. T4/T5 are `ol_intrinsic`, and
  at 130 Hz their outputs deplete to 11 %: the best-placed LPLC2 received ~840 weighted
  synapse-spikes per 10 ms at the end of a loom — enough to fire any LIF neuron flat out —
  and sat at −45.01 mV against a −45 mV threshold. Under the ownership rule the flyvis-driven
  optic lobe *is* the sensory periphery; `ol_intrinsic` and `ol_sensory` neurons are now
  exempt like the sensory ones. Their projections into the central brain (LC4, LPLC2 → GF)
  keep M1's depression: exempting those too pushed looming-by-activation over M1's 2 %
  locality bound.
- **Optic-lobe inhibition is ×8.** For a grating, inhibition onto LC4 cancelled 91 % of the
  excitation instead of exceeding it, so the giant fiber fired for wide-field motion before
  it fired for a loom. `inh_scale` with `inh_scale_pre` multiplies the inhibitory weights of
  the optic-lobe intrinsic interneurons (LPi, Li, Am1, Dm, Pm …). It has to be regional: a
  global ×2 already silences the M1 gustatory circuit (MN9 6 → 1 Hz), ×8 on the optic lobe
  leaves it at 6 Hz.

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
[escape]    first escape tick=51  disc diameter then=39.795918367346935  GF rate at end=0 Hz  LC4/LPLC2 mean 1.9 Hz, max 100 Hz
    -> ok: escape before 40°
[receding]  escape after onset=True  (escape ticks: [7, 8, 9, 10])
    -> FAIL: no escape for receding disc
[receding]  after 300 ms visible: escapes during the receding motion = 0 (startle ticks while static: [7, 8, 9])
[optomotor] cw: mean yaw +0.673, fraction with the right sign = 1.00, escapes = 0
[optomotor] ccw: mean yaw -0.720, fraction with the right sign = 1.00, escapes = 0
    -> ok: optomotor cw > +0.3
    -> ok: optomotor ccw < -0.3
    -> ok: optomotor mirror within 20 %
[two agents] agent0 escape=True agent1 escape=False
    -> ok: two agents independent
[two agents] optomotor: agent0 (grating) yaw +0.715, agent1 (grey) yaw +0.000
[speed]     1 agent(s): 38.3 ticks/s
    -> ok: speed 1 agent(s) >= 10 ticks/s
[speed]     2 agent(s): 36.6 ticks/s
    -> ok: speed 2 agent(s) >= 10 ticks/s

8/9 checks pass; failed: ['no escape for receding disc']
FAIL
```

**What works.** A rotating world produces a clean, mirror-symmetric optomotor turn of the right
sign through 721 columns × 2 eyes of flyvis, 56k driven neurons, the real T4/T5 → HS wiring and
a two-line readout, with the sign falling out of the anatomy. An expanding dark disc on the
right fires the giant fiber from about 30° on and the escape flag at 39.8°, through flyvis
T5 → LIF T5b/c/d → LPLC2 → DNp01, with nothing hand-placed along the way; a static or receding
disc, a grating and grey do not fire it once the disc has been visible for a while. Two agents
run as one batch and do not leak into each other. The loop runs 4× real time.

**How the looming escape was found, in order** — it did not fire from pixels at all for most of
the milestone, and each step below was a measured dead end before the next:

1. *Gain.* No global gain, no per-type gain (OFF pathway ×3), no contrast or speed made
   LC4/LPLC2 fire for a loom, while every gain ≥ 18 fired the giant fiber for a grating.
2. *Inhibition.* Scaling all inhibitory weights ×2–5 removed the grating false escapes exactly
   as the excitation/inhibition ledger predicted (+915k / −829k weighted spikes into LC4 for a
   grating) but left the loom at 0.2 Hz — and silenced M1's gustatory circuit.
3. *The lattice was upside down.* With HS kept selective (T4a/T5a low) and T5b/c/d driven at
   gain 300, LPLC2 still did nothing. Its T4/T5 inputs, placed in visual space through the
   alignment, pointed *towards* the receptive-field centre (radial index −0.68): the eye was
   wired as a contraction detector. Cause: flyvis's edge offsets are target − source and had
   been read as source − target, so `choose_symmetry` had picked the 180°-rotated symmetry.
   With the sign fixed the offsets and LPLC2's anatomy agree on `('-x','-y')` (+0.68).
4. *Depression.* Wired correctly, the best LPLC2 received ~840 weighted synapse-spikes per
   10 ms and still sat at threshold: M1's short-term depression cut T4/T5 outputs to 11 % at
   130 Hz. Exempting the optic lobe's intrinsic neurons gave the first loom escape (49°).
5. *Regional inhibition.* Un-depressed T4/T5 also drove the grating harder; ×8 on the optic
   lobe's inhibitory interneurons (not globally — see section 3) gated it. Final tuning of
   the T5b/c/d gain put the escape at 38–40°.

**What does not pass: the receding-disc check.** The 60° disc *appearing* on grey fires the
giant fiber 70–100 ms later (escape ticks 7–10) — an appearance startle, not a response to
the receding motion: once the disc has been visible for 300 ms, its shrinking produces zero
escapes, and a static disc produces the same startle ticks. The spec's check ignores the first
50 ms after onset; this pipeline's flash latency (flyvis's ~40 ms photoreceptor-to-T5 delay
plus the LIF chain) is 70 ms. The check is left as written rather than widened.

**Where the escape's margin comes from (investigated 2026-09-21, after M3 and M4 kept
hitting it).** The first escape sits at ~40° whatever is done downstream of LPLC2: scaling
the LC4/LPLC2 → GF synapses ×3 or ×6 (a stand-in for the gap junctions the model lacks)
raises the GF's spike count from 6 to 22 and 36 over the loom but not the onset; limiting the
×8 inhibition to the LPi cells frees LC4 for gratings, not for looms (its excitation is what
is missing); a one-spike escape rule sustains the escape (5–7 ticks instead of 2) but does
not advance it and turns hovering flies' stray GF spikes into escapes; ×3 on every optic-lobe
output with the bridge gains ÷3 changes nothing. The GF's ledger during the loom is +1256 to
−58 mV·spikes — it is not inhibited; it is fed by LPLC2 spikes each worth ~0.8 mV at its peak
(a 5 mV jump in a 5 ms synapse driving a 20 ms membrane), and LPLC2 itself starts firing only
when the disc is ~35° across, because each T5 spike is worth ~0.1 mV to it and dozens must
coincide — which happens once the rim is long. Real flies take off at 20–40° for looms of
this speed; the spec's "before 40°" sits on that onset, which is why every neighbouring
setting lands at 38–54°. `LIFBrain(edge_scale=…)` and `for_male_cns(optic_inh_types=…)` are
the (tested) knobs left from the investigation.

**Honest limits.**
- The looming escape is marginal in the sense above: onset at the biological edge of the
  criterion, two GF spikes in 20 ms needed. LC4 — biologically the GF's strongest loom
  input — stays silent: its main input T2 hyperpolarises in flyvis for dark stimuli.
- Gains are per pathway, not per physiology: T4a/T5a at 15 because HS cells lose selectivity
  above ~18 (28 synapses per T4a, `g = 0` reset), T5b/c/d at 120 because LPLC2 needs it.
  T4/T5 fire at 20–70 Hz, HS at ~100 Hz — both models of graded cells.
- The optic-lobe rules in `for_male_cns` (STD exemption, inhibition ×8 for intrinsic
  interneurons) are effect-calibrated on two reflexes. M1's gustatory and looming-by-activation
  scripts still pass, but nothing else has been re-checked.
- The readout is hand-written; `k_hs`, thresholds and the 20 ms window are guesses that
  happened to work. `lift` and `forward` are untested against anything.
- Column inference places a multi-column neuron at the modal column of its partners (median
  vote share 0.35); a T4 may sit one column off. Am1/CT1 keep their LIF wiring, a deliberate
  break of the ownership rule.
- flyvis's T4d barely responds to anything in ensemble member 0; the other 49 members are
  untried.
