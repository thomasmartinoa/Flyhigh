# Flyhigh

A fly brain, simulated from the wiring diagram, that will fly things.

The [male *Drosophila* CNS connectome](https://www.janelia.org/project-team/flyem/male-cns-connectome)
(Janelia FlyEM + Google + Cambridge, Cell 2026) maps every neuron and synapse of a fruit fly's
brain, eyes and nerve cord: **165,122 neurons, 124 million synapses**. This project turns that
wiring into a running spiking brain on a single GPU, then gives it eyes and a body:

| milestone | what | status |
|---|---|---|
| **M1 Brain** | load the connectome, run all 165k neurons as leaky integrate-and-fire units on the GPU, reproduce published circuit results | ✅ done |
| **M2 See & Move** | camera → flyvis optic lobe → LIF brain; descending neurons → motor commands (optomotor turn, looming escape) | ✅ done — 8 of 9 spec checks; the ninth is an appearance startle ([details](docs/03-see-and-move.md#5-results)) |
| **M3 The Box** | MuJoCo room with two fly-brained flying bodies and a moving "hand"; they react to each other | ✅ done — 6/6 experiments ([details](docs/04-the-box.md#4-the-experiments)) |
| **M4 flybody** | the brain steers Janelia's anatomical MuJoCo fly, flying on its own wings | ✅ done — 4/5 experiments; the optomotor drum fires the marginal giant fiber ([details](docs/05-flybody.md#3-the-experiments)) |
| **M4b two flies** | two anatomical flies, two brains, one drum | ✅ done — [2/3](docs/05-flybody.md#4-two-flies-m4b): the hand experiment works per fly, hovering shows stray escapes in a third of trials; at fly scale the flies ignore each other (17° at 2 cm) |
| M5 Real drone | the same brain flying a small drone in a room, reacting to you like a fly | next |

## Quick start

```bash
uv sync --all-groups                         # Python 3.12 venv with torch (CUDA), polars, flyvis
uv run python -m flyhigh.data.download       # 1.1 GB of connectome tables → data/raw/
uv run flyvis download-pretrained            # pretrained optic-lobe ensemble → data/flyvis/
# M4: flybody's trained flight policy + wing pattern (figshare 25309105: trained-fly-policies.zip,
#     datasets_flight-imitation.zip) unzipped under data/flybody/
uv run pytest                                # unit tests (synthetic fixtures, ~35 s)
uv run python scripts/validate_gustatory.py  # M1: sugar neurons → feeding motor neuron
uv run python scripts/validate_looming.py    # M1: looming detectors → giant fiber escape
uv run python scripts/build_alignment.py     # M2: flyvis columns ↔ male-CNS columns → data/cache/
uv run python scripts/validate_reflexes.py   # M2: silence, escape, optomotor, two agents, speed
uv run python scripts/validate_box.py        # M3: hover, hand → escape, optomotor in a body, two flies, speed
uv run python scripts/demo_box.py            # M3: annotated videos of the brain reacting → data/runs/
uv run python scripts/play_box.py            # M3: interactive box — drag things at the fly, live
uv run python scripts/validate_fly.py        # M4: flybody steering, hover, hand → escape, optomotor, speed
uv run python scripts/validate_fly2.py       # M4b: two flybody flies, two brains, one drum
uv run jupyter lab notebooks/                # 01 connectome, 02 brain, 03 see & move, 04 the box, 05 flybody
```

```python
from flyhigh.data.connectome import Connectome
from flyhigh.brain.lif import LIFBrain
from flyhigh.brain.stimulus import PoissonActivation
from flyhigh.brain.recorder import SpikeRecorder

c = Connectome.load("data/raw")                      # 165,122 neurons, 25.6 M signed edges
brain = LIFBrain.for_male_cns(c, n_agents=2)         # two flies, one GPU
looming = [PoissonActivation(c.ids_by_type(r"^LC4$|^LPLC2$"), rate_hz=100)]
rec = brain.run(5_000, stimuli=looming, recorder=SpikeRecorder())   # 500 ms
rec.rates(500)[:, c.ids_by_type(r"^DNp01$")]         # giant fiber ≈ 200 Hz in both flies
```

```python
from flyhigh.agent import FlyAgent
from flyhigh.senses.frame import rotating_grating

agent = FlyAgent(c, n_agents=2)                      # eyes + flyvis + bridge + brain + readout
for frame in rotating_grating(30, 60, 500):          # the world turns clockwise at 60°/s ...
    left, right = agent.tick([frame, frame])         # ... 10 ms per tick, 40 ticks/s
left.yaw                                             # ≈ +0.9: the flies turn with it
```

```python
from flyhigh.sim import Simulation

sim = Simulation(FlyAgent(c, n_agents=2), n_agents=2)   # a MuJoCo room, two flying bodies, a hand
sim.hand.approach(lambda: sim.bodies[0].pos)         # the hand goes for fly 0 ...
log = sim.run(3.0)                                   # ... which escapes; fly 1 does not
```

## Learn

- [`docs/01-connectome.md`](docs/01-connectome.md) — what the dataset is and isn't, the tables, the famous neurons
- [`docs/02-lif-brain.md`](docs/02-lif-brain.md) — the neuron model, how we validated it against the published FlyWire model, why the male CNS needed its own calibration, and the honest limits
- [`docs/05-flybody.md`](docs/05-flybody.md) — flybody's flight controller and how a `MotorCommand` steers it (the carrot), eyes on a bobbing head, what worked in the anatomical fly and what did not
- [`docs/04-the-box.md`](docs/04-the-box.md) — what changes when a brain has a body, why the room is an optomotor drum, the eye rig and the flying brick, the four experiments and their limits
- [`docs/03-see-and-move.md`](docs/03-see-and-move.md) — how a fly sees, why flyvis feeds the LIF, aligning two hex lattices (and how LPLC2's anatomy caught a 180° error), the optic-lobe calibration of the brain, the descending-neuron readout, and the five-step hunt for the looming escape
- `notebooks/` — the same as runnable, plotted walkthroughs

## Layout

```
src/flyhigh/data/      download.py, connectome.py   (tables → Connectome: neurons + signed edges)
src/flyhigh/brain/     lif.py, stimulus.py, recorder.py   (batched GPU LIF, Poisson activation, spikes, driven_only)
src/flyhigh/senses/    frame.py, eye.py, flyvis_eye.py, columns.py, alignment.py, bridge.py   (world → eyes → flyvis → currents)
src/flyhigh/motor/     command.py, readout.py   (descending-neuron rates → MotorCommand)
src/flyhigh/agent.py   FlyAgent.tick(frames) → commands   (the closed loop, batched over agents)
src/flyhigh/world/     scene.py, eyes.py   (the MuJoCo drum, hand and bodies; cubemap eyes → PanoramicFrame)
src/flyhigh/body/      brick.py, hand.py   (MotorCommand → forces; the scripted hand)
src/flyhigh/sim.py     Simulation.run(seconds) → log; VideoWriter   (render → brain → body → physics)
src/flyhigh/flybody/   policy.py, arena.py, task.py, multi.py, sim.py   (flybody's flies: TF flight policy, fly-scale drum, steering, multi-fly task, loop)
scripts/               validate_*.py, build_alignment.py, calibrate_bridge.py, benchmark.py, build_notebooks.py
tests/                 pytest, synthetic fixtures; real-data tests skipped without data/raw
docs/, notebooks/      learning track
Research_paper/        companion papers (PDFs git-ignored)
```

## Standing on

Shiu et al. 2024 (LIF whole-brain model) · Vaxenburg et al. 2025 (flybody) ·
Lappalainen et al. 2024 (flyvis) · the FlyEM male-CNS team. Data is CC-BY.
