# Flyhigh

A fly brain, simulated from the wiring diagram, that will fly things.

The [male *Drosophila* CNS connectome](https://www.janelia.org/project-team/flyem/male-cns-connectome)
(Janelia FlyEM + Google + Cambridge, Cell 2026) maps every neuron and synapse of a fruit fly's
brain, eyes and nerve cord: **165,122 neurons, 124 million synapses**. This project turns that
wiring into a running spiking brain on a single GPU, then gives it eyes and a body:

| milestone | what | status |
|---|---|---|
| **M1 Brain** | load the connectome, run all 165k neurons as leaky integrate-and-fire units on the GPU, reproduce published circuit results | ✅ done |
| M2 See & Move | camera → photoreceptors; descending neurons → motor commands (looming escape, optomotor) | next |
| M3 The Box | MuJoCo room with two fly-brained quadrotors and a moving "hand"; they react to each other | |
| M4 flybody | swap in Janelia's anatomically detailed MuJoCo fly | |
| M5 Real drone | the same brain flying a small drone in a room, reacting to you like a fly | |

## Quick start

```bash
uv sync --all-groups                         # Python 3.13 venv with torch (CUDA), polars, mujoco later
uv run python -m flyhigh.data.download       # 1.1 GB of connectome tables → data/raw/
uv run pytest                                # unit tests (synthetic fixtures, ~20 s)
uv run python scripts/validate_gustatory.py  # sugar neurons → feeding motor neuron
uv run python scripts/validate_looming.py    # looming detectors → giant fiber escape
uv run jupyter lab notebooks/                # 01_meet_the_connectome, 02_run_a_fly_brain
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

## Learn

- [`docs/01-connectome.md`](docs/01-connectome.md) — what the dataset is and isn't, the tables, the famous neurons
- [`docs/02-lif-brain.md`](docs/02-lif-brain.md) — the neuron model, how we validated it against the published FlyWire model, why the male CNS needed its own calibration, and the honest limits
- `notebooks/` — the same as runnable, plotted walkthroughs

## Layout

```
src/flyhigh/data/      download.py, connectome.py   (tables → Connectome: neurons + signed edges)
src/flyhigh/brain/     lif.py, stimulus.py, recorder.py   (batched GPU LIF, Poisson activation, spikes)
scripts/               validate_*.py, benchmark.py, build_notebooks.py
tests/                 pytest, synthetic fixtures; real-data tests skipped without data/raw
docs/, notebooks/      learning track
Research_paper/        companion papers (PDFs git-ignored)
```

## Standing on

Shiu et al. 2024 (LIF whole-brain model) · Vaxenburg et al. 2025 (flybody) ·
Lappalainen et al. 2024 (flyvis) · the FlyEM male-CNS team. Data is CC-BY.
