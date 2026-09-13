# M2 — See & Move: design

Date: 2026-09-13. Status: approved in brainstorming, awaiting written review.

## Goal

Give the M1 brain eyes and a motor output, so that a video of the world goes in and a
motor command comes out, with two visual reflexes working end-to-end:

1. **Looming escape** — an expanding dark disc → LC4 / LPLC2 → giant fiber (`DNp01`) → `escape`.
2. **Optomotor response** — the whole scene rotating → T4/T5 → HS/VS tangential cells →
   steering descending neurons → a counter-`yaw`.

Everything is driven by synthetic stimuli in M2; M3 replaces them with rendered camera frames
from the MuJoCo box, M5 with a real camera. Two agents must be supported (batched) from the start.

## Why flyvis in front of the LIF brain

The M1 LIF brain cannot compute visual motion by itself: photoreceptors and the first
optic-lobe layers are graded (non-spiking) cells, and direction selectivity in T4/T5 depends on
timing a uniform 1.8 ms delay cannot express. [flyvis](https://github.com/TuragaLab/flyvis)
(Lappalainen et al., Nature 2024) is a pretrained, connectome-constrained model of exactly
those 64 optic-lobe cell types on a 721-column hexagonal eye; it reproduces measured motion
responses. We use it as the retina-to-motion-detector stage and keep the real male-CNS wiring
for everything downstream (LC4, LPLC2, HS/VS, descending neurons, VNC).

**Ownership rule (Option A):** for every cell type flyvis models, the LIF copies of those
neurons become *driven-only*: their incoming synapses are removed and their spikes are caused
solely by an external current derived from flyvis's activity for the same type and eye column.
All other neurons keep their wiring untouched. No neuron is computed twice.

## Architecture

```
world frame(s) ─► PanoramicFrame ─► EyeSampler ─► flyvis.Network ─► FlyvisBridge ─► LIFBrain ─► Readout ─► MotorCommand
                 (az×el luminance)  (2×721 hexals) (64 types×721 cols) (ext_i, mV)   (M1)        (DN rates)
```

Tick = one camera frame at 100 Hz: flyvis advances one 10 ms step, the bridge converts the
resulting activities into `ext_i` currents held constant for the next 100 LIF steps (0.1 ms),
then the readout integrates descending-neuron spikes over the last 20 ms into a command.

### `flyhigh.senses`

- `PanoramicFrame`: float32 luminance in [0, 1] on an equirectangular grid, azimuth −180…180°
  × elevation −90…90°, default 360×180 (1° per pixel). Any source fills it: the synthetic
  stimulus generator (M2), the MuJoCo renderer (M3), a fisheye camera (M5). Cheap, camera-agnostic.
- `EyeGeometry`: for each eye, the 721 flyvis hex columns (u, v) → viewing direction
  (azimuth, elevation). Fly ommatidia are ~5° apart; each eye looks sideways-forward and covers
  roughly the ipsilateral hemisphere with ~15° binocular overlap in front. Parameters: column
  spacing (5°), eye yaw offset (±30°), pitch (0°), acceptance-cone half-width (5°).
- `EyeSampler`: pre-computes a sparse (721 × 360·180) matrix per eye whose rows are
  Gaussian-weighted acceptance cones; `sample(frame) → (2, 721)` luminances. Pure linear algebra,
  unit-testable on synthetic frames (a bright spot at a known azimuth lights the expected column).
- `FlyvisEye`: wraps a pretrained `flyvis.Network` (ensemble member 0000 by default). Holds the
  network state across ticks so it runs online (`Network.forward` with a persisted state — the
  public `simulate()` is offline-only, so this needs a thin adapter). Input `(batch, 1, 1, 721)`
  per tick; output activity `(batch, n_flyvis_neurons)`. Batch = agents × 2 eyes.
- `ColumnAlignment`: maps each flyvis neuron (type, u, v) to a male-CNS neuron index
  (`type`, `side`, `hex1`, `hex2`). Built once, cached to parquet. Method: both lattices are
  hexagonal with one column per ommatidium; translate flyvis's (u, v) into the male-CNS
  (hex1, hex2) frame by aligning lattice centres and axes, then nearest-neighbour for columns
  outside the 721-hexal disc. Verification: the fraction of flyvis neurons with a same-type
  partner within one column must be > 90 %; T4a–d / T5a–d subtype names must match one-to-one.
  Types present in flyvis but named differently in the male CNS are mapped through a small
  explicit table (`senses/type_map.py`).

### `flyhigh.senses.bridge`

- `FlyvisBridge`: `activity (batch, n_flyvis) → ext_i (n_agents, N)` for the LIF brain.
  `ext_i[agent, lif_idx] = gain[type] · clamp(activity − rest[type], 0)`, with `rest` the
  flyvis steady-state activity on grey (so a grey screen injects nothing) and `gain` per type,
  calibrated so a strong stimulus drives T4/T5 to ~50–100 Hz (measured in the validation
  script, stored as constants with the sweep table in `docs/03-see-and-move.md`).
- `LIFBrain.for_male_cns(..., driven_only=<indices>)`: new keyword that zeroes incoming weights
  of the given neurons (in both propagation kernels). Existing preset unchanged otherwise.

### `flyhigh.motor`

```python
@dataclass(frozen=True)
class MotorCommand:
    forward: float    # −1..1, body frame
    yaw: float        # −1..1, + = turn right
    lift: float       # −1..1
    escape: bool      # take-off reflex; the body layer lets it override the rest for ~100 ms
```

`Readout(connectome)` resolves the neuron sets once and exposes `command(rates) → MotorCommand`
where `rates` is `(n_agents, N)` Hz over the last 20 ms window (from `SpikeRecorder`-style
counts kept by the agent). Channels, each a small pure function with its own tests:

| channel | neurons | rule |
|---|---|---|
| `escape` | `DNp01` (GF), `DNp04` | GF rate > 50 Hz, or GF > 20 Hz and DNp04 > 50 Hz |
| `yaw` | `DNa01`, `DNa02` (L/R); `HSN/HSE/HSS` (L/R) | `k_dn·(R−L)/(R+L+ε)` from DNa + `k_hs·(L−R)/(L+R+ε)` from HS (sign: counter-rotate the perceived motion), clipped to ±1 |
| `lift` | `VS` (L+R) vs rest | `k_vs·(VS − vs_rest)` clipped |
| `forward` | `DNp09` | `0.2 + k_fwd·rate`, clipped; 0 while `escape` |

Gains and thresholds live in one `ReadoutParams` dataclass. The rules are deliberately simple;
a learned linear readout can replace `command()` later behind the same interface.

### `flyhigh.agent`

`FlyAgent(connectome, n_agents)`: owns sampler, flyvis eye, bridge, brain, readout;
`tick(frames: list[PanoramicFrame]) → list[MotorCommand]`. Also exposes the last DN rates and
spike counts for plotting. No body, no physics — that is M3.

### `flyhigh.senses.stimuli`

Synthetic `PanoramicFrame` sequences for tests and validation: `grey()`, `looming_disc(azimuth,
elevation, start_deg, end_deg, duration_ms)`, `rotating_grating(wavelength_deg, deg_per_s,
direction)`, `moving_spot(...)`.

## Data flow per tick (2 agents)

1. frames (2 × PanoramicFrame) → `EyeSampler.sample` → luminance `(4, 721)` [agent × eye]
2. `FlyvisEye.step` → activity `(4, n_flyvis)`
3. `FlyvisBridge` → `ext_i (2, N)`
4. `LIFBrain.step(ext_i)` × 100, accumulating DN spike counts in a 20 ms ring
5. `Readout.command(rates)` → 2 × `MotorCommand`

## Validation (`scripts/validate_reflexes.py`, all must pass)

1. **Silence:** 1 s of grey → no descending-neuron spikes, `|yaw|, |forward−0.2|, |lift| < 0.05`, `escape=False`.
2. **Escape:** looming disc from azimuth +60° (right eye), 5° → 60° over 500 ms → LPLC2/LC4 rates
   rise; GF spikes; `escape=True` at some tick before the disc reaches 40°; no escape for a
   receding disc (60° → 5°).
3. **Optomotor:** full-field grating rotating clockwise at 60°/s → `yaw` < −0.3 sustained for
   ≥ 200 ms; anticlockwise → `yaw` > +0.3; the two responses mirror to within 20 %.
4. **Two agents independent:** agent 0 sees the looming disc, agent 1 sees grey → only agent 0 escapes.
5. **Speed:** report ticks/s for 1 and 2 agents (target ≥ 10 ticks/s at full GPU clocks).

Unit tests (synthetic data, no downloads): sampler geometry, alignment table shape/coverage on a
fake lattice, bridge indexing, each readout channel, `driven_only` weight removal.

## Environment change

flyvis supports Python ≤ 3.12 → the project moves to **Python 3.12** (`requires-python
">=3.12,<3.13"`, `.python-version` 3.12). torch 2.14 and mujoco have 3.12 wheels. `flyvis` is a
required dependency; `flyvis download-pretrained` is documented in the README quick start and
called by `flyhigh.data.download` with a `--flyvis` flag. Day-one check: import flyvis, load the
pretrained ensemble, simulate 100 ms of grey on the GPU.

## Out of scope for M2

Bodies, physics, rendering (M3); object tracking / LC10–LC11 fixation; learned readouts;
the VNC's own motor circuits (we read descending neurons, not leg/wing motor neurons).

## Learning track

`docs/03-see-and-move.md` (how a fly sees: ommatidia, the ON/OFF pathways, T4/T5, why a spiking
model needs flyvis; the descending-neuron bottleneck) and `notebooks/03_see_and_move.ipynb`
(show the sampled eye image, flyvis T4 activity maps for a rotating grating, DN rasters, the
command traces for the three stimuli).
