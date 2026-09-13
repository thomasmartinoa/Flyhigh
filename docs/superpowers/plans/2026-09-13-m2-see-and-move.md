# M2 — See & Move Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A video of the world goes in, a motor command comes out, and two real fly reflexes (looming escape, optomotor turn) work end-to-end through the connectome brain.

**Architecture:** A camera-agnostic `PanoramicFrame` is sampled onto two 721-column hexagonal eyes; the pretrained **flyvis** optic-lobe model turns each eye's video into graded activity for 64 cell types; a **bridge** injects that activity as currents into the matching neurons of the M1 LIF brain (those neurons become *driven-only*); the untouched downstream wiring (LC4, LPLC2, HS/VS, descending neurons) does the rest; a hand-mapped **readout** of descending-neuron rates produces a `MotorCommand`. `FlyAgent.tick(frame)` runs one 10 ms camera frame = 1 flyvis step + 100 LIF steps.

**Tech Stack:** Python 3.12 (flyvis requires ≤ 3.12), `uv`, PyTorch 2.14 CUDA, polars, numpy/scipy, `flyvis` 1.x + its pretrained ensemble, pytest, existing `flyhigh.data` / `flyhigh.brain` from M1.

**Spec:** `docs/superpowers/specs/2026-09-13-m2-see-and-move-design.md`

## Global Constraints

- Python `>=3.12,<3.13` (flyvis limit); all deps managed by `uv`; run everything with `uv run`.
- Tests: TDD, `uv run pytest`; markers `data` (needs `data/raw`), `flyvis` (needs pretrained models). Both skip cleanly when absent.
- Times in ms, voltages in mV, luminance in [0, 1] with grey = 0.5, angles in degrees (azimuth −180…180, + = right; elevation −90…90, + = up).
- Camera tick = 10 ms (100 Hz): flyvis `dt = 0.01`, LIF 100 steps of 0.1 ms.
- Every commit ends with the attribution lines given by the session (see git log for the format).
- Match the M1 code style: small modules, dataclasses for parameters, docstrings that say *why*.

## A note for the learner

Each task starts with **What you're learning** — the fly-biology or engineering idea behind it. Read it before the code. The tests are the specification: if you understand why a test asserts what it asserts, you understand the design.

---

## File map

| file | responsibility |
|---|---|
| `pyproject.toml`, `.python-version`, `.env` | Python 3.12, `flyvis` dependency, `FLYVIS_ROOT_DIR=data/flyvis` |
| `scripts/check_flyvis.py` | day-one smoke test: load pretrained model, run 100 ms of grey on the GPU |
| `src/flyhigh/senses/frame.py` | `PanoramicFrame` + synthetic stimuli (`grey`, `looming_disc`, `rotating_grating`, `moving_spot`) |
| `src/flyhigh/senses/eye.py` | `EyeGeometry` (721 hex columns → viewing directions), `EyeSampler` (frame → per-column luminance) |
| `src/flyhigh/senses/flyvis_eye.py` | `FlyvisEye`: pretrained network, steady state, online `step()` |
| `src/flyhigh/senses/type_map.py` | flyvis cell-type name → male-CNS `type` |
| `src/flyhigh/senses/alignment.py` | `ColumnAlignment`: flyvis (type, u, v, eye) → LIF neuron index; lattice symmetry search |
| `src/flyhigh/senses/bridge.py` | `FlyvisBridge`: flyvis activity → `ext_i` for the LIF |
| `src/flyhigh/brain/lif.py` (modify) | `driven_only=` keyword: zero incoming synapses of given neurons |
| `src/flyhigh/motor/command.py` | `MotorCommand` |
| `src/flyhigh/motor/readout.py` | `ReadoutParams`, `Readout` with one pure function per channel |
| `src/flyhigh/agent.py` | `FlyAgent.tick(frames) → commands` |
| `scripts/build_alignment.py` | build + report the real alignment table (coverage %, unmatched types) |
| `scripts/calibrate_bridge.py` | gain sweep: rotating grating → T4/T5 LIF rates |
| `scripts/validate_reflexes.py` | the five spec validations, exit 0 iff pass |
| `docs/03-see-and-move.md`, `notebooks/03_see_and_move.ipynb` (via `scripts/build_notebooks.py`) | learning track |
| `tests/test_frame.py`, `test_eye.py`, `test_flyvis_eye.py`, `test_alignment.py`, `test_bridge.py`, `test_readout.py`, `test_agent.py`, `tests/test_lif.py` (add) | tests |

---

### Task 1: Python 3.12 + flyvis installed and smoke-tested

**What you're learning:** flyvis is a *trained* model: its synapse strengths and neuron time constants were fitted so the network reproduces measured responses. It ships as a directory of checkpoints ("flow/0000/000" = first model of the ensemble trained on optic flow). We only ever run it forward.

**Files:**
- Modify: `pyproject.toml` (requires-python, dependency), `.python-version`, `.gitignore`
- Create: `.env`, `scripts/check_flyvis.py`

- [ ] **Step 1: Pin Python 3.12 and add flyvis**

```bash
cd /mnt/ssd_sata/Flyhigh
uv python pin 3.12
```
Edit `pyproject.toml`: `requires-python = ">=3.12,<3.13"`, add `"flyvis>=1.1"` to `dependencies`, add `"flyvis: needs the pretrained flyvis models under data/flyvis"` to `[tool.pytest.ini_options] markers`.
Create `.env` with one line: `FLYVIS_ROOT_DIR=data/flyvis` and add `data/flyvis/` to `.gitignore`.

- [ ] **Step 2: Re-create the environment**

Run: `rm -rf .venv && uv sync --all-groups`
Expected: resolves without conflict; `uv run python -c "import flyvis, torch; print(flyvis.__version__, torch.cuda.is_available())"` prints a version and `True`.
If `flyvis` pulls a CPU-only torch, add to `pyproject.toml`:
```toml
[tool.uv.sources]
torch = { index = "pytorch-cu130" }
[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true
```

- [ ] **Step 3: Download the pretrained models**

Run: `uv run flyvis download-pretrained`
Expected: `data/flyvis/results/flow/0000/000` exists (an ensemble of 50; we use member 000).

- [ ] **Step 4: Write the smoke test script**

```python
# scripts/check_flyvis.py
"""Day-one check: load the pretrained flyvis model and run 100 ms of grey on the GPU."""
import time

import flyvis
import torch

view = flyvis.NetworkView(flyvis.results_dir / "flow/0000/000")
net = view.init_network().to(flyvis.device).eval()
nodes = net.connectome.nodes
print(f"flyvis device={flyvis.device}  neurons={len(nodes.type[:])}  types={len(net.connectome.unique_cell_types[:])}")
print("input cell types:", [t.decode() for t in net.connectome.input_cell_types[:]])
print("hexals per eye:", net.stimulus.n_input_elements)
t = time.perf_counter()
state = net.steady_state(t_pre=1.0, dt=0.01, batch_size=1)
torch.cuda.synchronize()
print(f"steady state after 1 s grey: {time.perf_counter() - t:.2f} s, mean activity {state.nodes.activity.mean():.4f}")
```

- [ ] **Step 5: Run it**

Run: `uv run python scripts/check_flyvis.py`
Expected: device `cuda`, ~45k neurons, 64+ types (R1–R8 among inputs), `hexals per eye: 721`, steady state in a few seconds. Record the neuron count in the commit message.

- [ ] **Step 6: Run the M1 tests on 3.12 and commit**

Run: `uv run pytest -q`
Expected: 37 passed.
```bash
git add pyproject.toml .python-version .env .gitignore uv.lock scripts/check_flyvis.py
git commit -m "M2: move to Python 3.12, add flyvis + pretrained models, smoke test"
```

---

### Task 2: `PanoramicFrame` and synthetic stimuli

**What you're learning:** a fly sees almost the whole sphere around it. Instead of tying the code to any camera, we describe the world as luminance over azimuth × elevation (an equirectangular map, like a world map). Cameras, renderers and test patterns all just fill this map. The three classic lab stimuli — a looming disc, a rotating grating, a moving spot — are how neuroscientists probe the very circuits we are reading out.

**Files:**
- Create: `src/flyhigh/senses/__init__.py`, `src/flyhigh/senses/frame.py`
- Test: `tests/test_frame.py`

**Interfaces:**
- Produces: `PanoramicFrame(lum: np.ndarray[float32, (H, W)])` with `.az_deg (W,)`, `.el_deg (H,)`, `.shape`; constructors `PanoramicFrame.grey(value=0.5, h=180, w=360)`; stimulus generators returning `list[PanoramicFrame]` (one per 10 ms tick): `looming_disc(az, el, start_deg, end_deg, duration_ms, lum=0.0, bg=0.5)`, `rotating_grating(wavelength_deg, deg_per_s, duration_ms, direction=+1)`, `moving_spot(az0, el, deg_per_s, duration_ms, radius_deg=5, lum=0.0, bg=0.5)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_frame.py
import numpy as np
import pytest

from flyhigh.senses.frame import PanoramicFrame, looming_disc, moving_spot, rotating_grating


def test_grey_frame_geometry():
    f = PanoramicFrame.grey()
    assert f.lum.shape == (180, 360) and f.lum.dtype == np.float32
    assert f.lum.min() == f.lum.max() == pytest.approx(0.5)
    assert f.az_deg[0] == pytest.approx(-179.5) and f.az_deg[-1] == pytest.approx(179.5)
    assert f.el_deg[0] == pytest.approx(89.5) and f.el_deg[-1] == pytest.approx(-89.5)  # row 0 = top


def test_looming_disc_grows_from_start_to_end_angle():
    frames = looming_disc(az=60, el=0, start_deg=5, end_deg=60, duration_ms=500)
    assert len(frames) == 50
    dark0 = (frames[0].lum < 0.25).sum()
    dark1 = (frames[-1].lum < 0.25).sum()
    assert 0 < dark0 < dark1
    # the disc is centred at azimuth 60: the darkest column is there
    col = frames[-1].lum.mean(axis=0).argmin()
    assert frames[-1].az_deg[col] == pytest.approx(60, abs=1.5)
    # background untouched on the far side
    assert frames[-1].lum[:, 0] == pytest.approx(0.5)


def test_rotating_grating_shifts_by_speed():
    frames = rotating_grating(wavelength_deg=30, deg_per_s=60, duration_ms=100, direction=+1)
    assert len(frames) == 10
    row0 = frames[0].lum[90]
    row5 = frames[5].lum[90]  # 50 ms later the pattern moved 3 degrees = 3 pixels to the right
    np.testing.assert_allclose(np.roll(row0, 3), row5, atol=1e-6)
    assert frames[0].lum.min() >= 0 and frames[0].lum.max() <= 1


def test_moving_spot_moves_right():
    frames = moving_spot(az0=-30, el=0, deg_per_s=100, duration_ms=300)
    c0 = frames[0].lum.mean(axis=0).argmin(); c1 = frames[-1].lum.mean(axis=0).argmin()
    assert frames[0].az_deg[c0] == pytest.approx(-30, abs=1.5)
    assert frames[-1].az_deg[c1] == pytest.approx(-1, abs=2)
```

- [ ] **Step 2: Run to verify RED**

Run: `uv run pytest tests/test_frame.py -q`
Expected: collection error, `No module named 'flyhigh.senses'`.

- [ ] **Step 3: Implement**

```python
# src/flyhigh/senses/__init__.py
"""Seeing: panoramic frames → hexagonal eyes → flyvis → currents into the LIF brain."""
```

```python
# src/flyhigh/senses/frame.py
"""A camera-agnostic picture of the world: luminance over azimuth × elevation.

Row 0 is the top of the sky (elevation +90), column 0 is azimuth −180 (behind the fly),
azimuth increases to the right, +90 is the fly's right side, 0 is straight ahead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TICK_MS = 10.0


@dataclass(frozen=True)
class PanoramicFrame:
    lum: np.ndarray  # (H, W) float32 in [0, 1]

    @property
    def shape(self) -> tuple[int, int]:
        return self.lum.shape

    @property
    def az_deg(self) -> np.ndarray:
        w = self.lum.shape[1]
        return -180.0 + (np.arange(w) + 0.5) * 360.0 / w

    @property
    def el_deg(self) -> np.ndarray:
        h = self.lum.shape[0]
        return 90.0 - (np.arange(h) + 0.5) * 180.0 / h

    @classmethod
    def grey(cls, value: float = 0.5, h: int = 180, w: int = 360) -> PanoramicFrame:
        return cls(np.full((h, w), value, dtype=np.float32))

    def angular_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """(az, el) in degrees for every pixel, each (H, W)."""
        return np.meshgrid(self.az_deg, self.el_deg)


def _angular_distance(az, el, az0, el0):
    """Great-circle distance in degrees between every pixel and a direction."""
    a, e, a0, e0 = np.radians(az), np.radians(el), np.radians(az0), np.radians(el0)
    cos_d = np.sin(e) * np.sin(e0) + np.cos(e) * np.cos(e0) * np.cos(a - a0)
    return np.degrees(np.arccos(np.clip(cos_d, -1.0, 1.0)))


def _disc(base: PanoramicFrame, az, el, radius_deg, lum) -> PanoramicFrame:
    azg, elg = base.angular_grid()
    out = base.lum.copy()
    out[_angular_distance(azg, elg, az, el) <= radius_deg] = lum
    return PanoramicFrame(out)


def looming_disc(az, el, start_deg, end_deg, duration_ms, lum=0.0, bg=0.5, h=180, w=360):
    """A dark disc whose angular *diameter* grows linearly from start_deg to end_deg."""
    n = int(round(duration_ms / TICK_MS))
    base = PanoramicFrame.grey(bg, h, w)
    return [_disc(base, az, el, 0.5 * (start_deg + (end_deg - start_deg) * i / max(n - 1, 1)), lum) for i in range(n)]


def rotating_grating(wavelength_deg, deg_per_s, duration_ms, direction=+1, h=180, w=360, contrast=1.0):
    """Vertical stripes covering the whole panorama, drifting `direction` (+1 = rightwards)."""
    n = int(round(duration_ms / TICK_MS))
    az = PanoramicFrame.grey(h=h, w=w).az_deg
    frames = []
    for i in range(n):
        phase = direction * deg_per_s * (i * TICK_MS / 1000.0)
        row = 0.5 + 0.5 * contrast * np.sin(2 * np.pi * (az - phase) / wavelength_deg)
        frames.append(PanoramicFrame(np.tile(row.astype(np.float32), (h, 1))))
    return frames


def moving_spot(az0, el, deg_per_s, duration_ms, radius_deg=5.0, lum=0.0, bg=0.5, h=180, w=360):
    n = int(round(duration_ms / TICK_MS))
    base = PanoramicFrame.grey(bg, h, w)
    return [_disc(base, az0 + deg_per_s * i * TICK_MS / 1000.0, el, radius_deg, lum) for i in range(n)]
```

- [ ] **Step 4: Run to verify GREEN**

Run: `uv run pytest tests/test_frame.py -q`
Expected: 4 passed. (If the grating roll test is off by one pixel, the phase sign is reversed — fix the sign, not the test.)

- [ ] **Step 5: Commit**

```bash
git add src/flyhigh/senses tests/test_frame.py
git commit -m "senses: PanoramicFrame and synthetic looming/grating/spot stimuli"
```

---

### Task 3: `EyeGeometry` and `EyeSampler`

**What you're learning:** a fly eye is ~800 little lenses (ommatidia) on a hexagonal grid, each looking in one direction with a ~5° acceptance cone. flyvis models one eye as a regular hexagon of radius 15 → 721 columns in axial coordinates (u, v). We give every column a viewing direction and pre-compute a sparse matrix that turns a panoramic frame into 721 luminances per eye in one multiply.

**Files:**
- Create: `src/flyhigh/senses/eye.py`
- Test: `tests/test_eye.py`

**Interfaces:**
- Consumes: `PanoramicFrame` (Task 2); `flyvis.utils.hex_utils.get_hex_coords(15)`, `hex_to_pixel(u, v)`.
- Produces: `EyeGeometry(side: "L"|"R", extent=15, spacing_deg=5.0, center_az_deg=65.0, center_el_deg=0.0, cone_deg=5.0)` with `.u (721,)`, `.v (721,)`, `.az_deg (721,)`, `.el_deg (721,)`; `EyeSampler(frame_shape, eyes: list[EyeGeometry])` with `.sample(frame) -> np.ndarray (n_eyes, 721)`. Eye order is always `["L", "R"]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_eye.py
import numpy as np
import pytest

from flyhigh.senses.eye import EyeGeometry, EyeSampler
from flyhigh.senses.frame import PanoramicFrame


def test_right_eye_looks_right_and_covers_about_75_degrees():
    eye = EyeGeometry("R")
    assert eye.u.shape == (721,) and eye.az_deg.shape == (721,)
    centre = (eye.u == 0) & (eye.v == 0)
    assert eye.az_deg[centre].item() == pytest.approx(65.0) and eye.el_deg[centre].item() == pytest.approx(0.0)
    assert eye.az_deg.min() > -15 and eye.az_deg.max() < 145      # 15 columns * 5 deg each side
    assert abs(eye.el_deg).max() < 80


def test_left_eye_is_mirror_of_right():
    l, r = EyeGeometry("L"), EyeGeometry("R")
    np.testing.assert_allclose(np.sort(l.az_deg), np.sort(-r.az_deg), atol=1e-6)


def test_bright_spot_lights_the_column_looking_at_it():
    eye = EyeGeometry("R")
    sampler = EyeSampler((180, 360), [eye])
    frame = PanoramicFrame.grey(0.0)
    target = 100  # some column
    az, el = eye.az_deg[target], eye.el_deg[target]
    azg, elg = frame.angular_grid()
    lum = frame.lum.copy(); lum[(abs(azg - az) < 3) & (abs(elg - el) < 3)] = 1.0
    out = sampler.sample(PanoramicFrame(lum))
    assert out.shape == (1, 721)
    assert out[0].argmax() == target and out[0, target] > 0.5


def test_grey_frame_samples_to_grey_everywhere():
    sampler = EyeSampler((180, 360), [EyeGeometry("L"), EyeGeometry("R")])
    out = sampler.sample(PanoramicFrame.grey(0.5))
    assert out.shape == (2, 721)
    np.testing.assert_allclose(out, 0.5, atol=1e-6)
```

- [ ] **Step 2: Run to verify RED**

Run: `uv run pytest tests/test_eye.py -q`
Expected: `No module named 'flyhigh.senses.eye'`.

- [ ] **Step 3: Implement**

```python
# src/flyhigh/senses/eye.py
"""One compound eye = 721 hexagonal columns (flyvis extent 15), each looking in a direction.

flyvis axial hex coords (u, v) → planar offsets via its own `hex_to_pixel` convention, scaled so
neighbouring columns are `spacing_deg` apart; the plane is placed on the sphere at the eye's
centre direction. Each column integrates light over a Gaussian acceptance cone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from flyvis.utils.hex_utils import get_hex_coords, hex_to_pixel

from flyhigh.senses.frame import PanoramicFrame


@dataclass(frozen=True)
class EyeGeometry:
    side: str  # "L" or "R"
    extent: int = 15
    spacing_deg: float = 5.0
    center_az_deg: float = 65.0  # right eye; the left eye is mirrored
    center_el_deg: float = 0.0
    cone_deg: float = 5.0  # acceptance-cone half-width (Gaussian sigma ≈ cone/2)
    u: np.ndarray = field(init=False, repr=False)
    v: np.ndarray = field(init=False, repr=False)
    az_deg: np.ndarray = field(init=False, repr=False)
    el_deg: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        u, v = get_hex_coords(self.extent)
        x, y = hex_to_pixel(u, v)  # neighbours are sqrt(3) apart in this convention
        scale = self.spacing_deg / np.sqrt(3.0)
        sign = 1.0 if self.side == "R" else -1.0
        az = sign * (self.center_az_deg + x * scale)
        el = self.center_el_deg - y * scale  # hex_to_pixel's y grows downward
        object.__setattr__(self, "u", u)
        object.__setattr__(self, "v", v)
        object.__setattr__(self, "az_deg", az.astype(np.float32))
        object.__setattr__(self, "el_deg", el.astype(np.float32))

    @property
    def n_columns(self) -> int:
        return len(self.u)


class EyeSampler:
    """Sparse (n_eyes*721, H*W) matrix of normalised Gaussian cones; sample() is one matmul."""

    def __init__(self, frame_shape: tuple[int, int], eyes: list[EyeGeometry]):
        h, w = frame_shape
        self.eyes = eyes
        frame = PanoramicFrame.grey(h=h, w=w)
        azg, elg = frame.angular_grid()
        rows = []
        for eye in eyes:
            for az0, el0 in zip(eye.az_deg, eye.el_deg):
                d = _angular_distance(azg, elg, az0, el0)
                wgt = np.exp(-0.5 * (d / (eye.cone_deg / 2.0)) ** 2)
                wgt[d > 3 * eye.cone_deg] = 0.0
                rows.append(sp.csr_matrix((wgt / wgt.sum()).ravel().astype(np.float32)))
        self.M = sp.vstack(rows).tocsr()

    def sample(self, frame: PanoramicFrame) -> np.ndarray:
        out = self.M @ frame.lum.ravel()
        return out.reshape(len(self.eyes), -1)


def _angular_distance(az, el, az0, el0):
    a, e, a0, e0 = np.radians(az), np.radians(el), np.radians(az0), np.radians(el0)
    cos_d = np.sin(e) * np.sin(e0) + np.cos(e) * np.cos(e0) * np.cos(a - a0)
    return np.degrees(np.arccos(np.clip(cos_d, -1.0, 1.0)))
```
(Move `_angular_distance` into `frame.py` as a public `angular_distance` and import it here instead of duplicating — do that now, update `frame.py` accordingly.)

- [ ] **Step 4: Run to verify GREEN**

Run: `uv run pytest tests/test_eye.py tests/test_frame.py -q`
Expected: 8 passed. Building the sampler takes ~2 s (721 × 2 Gaussians over 64,800 pixels); fine.

- [ ] **Step 5: Commit**

```bash
git add src/flyhigh/senses tests/test_eye.py
git commit -m "senses: EyeGeometry (flyvis hex lattice → viewing directions) and sparse EyeSampler"
```

---

### Task 4: `FlyvisEye` — the pretrained optic lobe, stepped online

**What you're learning:** flyvis's public `simulate()` wants the whole movie up front. A living fly (and our closed loop) gets one frame at a time, so we keep the network's internal state between calls and push one frame per tick. Neurons in flyvis are graded: their "activity" is a rectified voltage, not spikes.

**Files:**
- Create: `src/flyhigh/senses/flyvis_eye.py`
- Test: `tests/test_flyvis_eye.py` (marker `flyvis`)

**Interfaces:**
- Produces: `FlyvisEye(model="flow/0000/000", dt=0.01, device=None)` with `.n_neurons`, `.types: np.ndarray[str] (n_neurons,)`, `.u, .v: np.ndarray[int] (n_neurons,)`, `.reset(batch_size) -> None` (1 s of grey → steady state; stores `.rest (n_neurons,)` = steady activity), `.step(lum: torch.Tensor (batch, 721)) -> torch.Tensor (batch, n_neurons)` activity after one dt.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_flyvis_eye.py
import numpy as np
import pytest
import torch

flyvis = pytest.importorskip("flyvis")
pytestmark = pytest.mark.flyvis

from flyhigh.senses.flyvis_eye import FlyvisEye  # noqa: E402


@pytest.fixture(scope="module")
def eye():
    if not (flyvis.results_dir / "flow/0000/000").exists():
        pytest.skip("pretrained flyvis models not downloaded")
    e = FlyvisEye()
    e.reset(batch_size=2)
    return e


def test_metadata(eye):
    assert eye.n_neurons > 40_000
    assert set(["T4a", "T4b", "T4c", "T4d", "T5a", "L1", "Mi1"]) <= set(np.unique(eye.types))
    assert eye.u.shape == eye.v.shape == (eye.n_neurons,)
    assert eye.rest.shape == (eye.n_neurons,)


def test_grey_input_stays_at_rest(eye):
    grey = torch.full((2, 721), 0.5)
    for _ in range(20):
        act = eye.step(grey)
    assert act.shape == (2, eye.n_neurons)
    assert torch.allclose(act[0].cpu(), torch.as_tensor(eye.rest), atol=1e-3)


def test_moving_edge_drives_t4_more_than_grey(eye):
    eye.reset(batch_size=1)
    t4 = torch.as_tensor(np.isin(eye.types, ["T4a", "T4b", "T4c", "T4d"]))
    base = eye.step(torch.full((1, 721), 0.5))[0, t4].sum().item()
    # a bright half-field sweeping across the eye: columns with v < k light up one by one
    v = torch.as_tensor(eye_v_of_columns())
    peak = 0.0
    for k in range(-15, 16):
        lum = torch.where(v < k, 1.0, 0.0).float()[None]
        peak = max(peak, eye.step(lum)[0, t4].sum().item())
    assert peak > 2 * base + 1e-3


def eye_v_of_columns():
    from flyvis.utils.hex_utils import get_hex_coords
    return get_hex_coords(15)[1]
```

- [ ] **Step 2: Run to verify RED**

Run: `uv run pytest tests/test_flyvis_eye.py -q`
Expected: `No module named 'flyhigh.senses.flyvis_eye'`.

- [ ] **Step 3: Implement**

```python
# src/flyhigh/senses/flyvis_eye.py
"""The pretrained flyvis optic lobe, run one 10 ms frame at a time with persistent state."""

from __future__ import annotations

import numpy as np
import torch


class FlyvisEye:
    def __init__(self, model: str = "flow/0000/000", dt: float = 0.01, device=None):
        import flyvis

        self.dt = dt
        self.device = torch.device(device or flyvis.device)
        view = flyvis.NetworkView(flyvis.results_dir / model)
        self.net = view.init_network().to(self.device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        nodes = self.net.connectome.nodes
        self.types = np.array([t.decode() for t in nodes.type[:]])
        self.u = np.asarray(nodes.u[:], dtype=np.int64)
        self.v = np.asarray(nodes.v[:], dtype=np.int64)
        self.n_neurons = len(self.types)
        self._params = None
        self._state = None
        self.rest = None

    def reset(self, batch_size: int) -> None:
        """1 s of grey (0.5) → steady state; remember it as the per-neuron resting activity."""
        with torch.no_grad():
            self._state = self.net.steady_state(t_pre=1.0, dt=self.dt, batch_size=batch_size)
        self.rest = self._state.nodes.activity[0].detach().cpu().numpy()
        self.batch_size = batch_size

    def step(self, lum: torch.Tensor) -> torch.Tensor:
        """lum: (batch, 721) in [0, 1]. Returns activity (batch, n_neurons) after one dt."""
        assert self._state is not None, "call reset(batch_size) first"
        x = lum.to(self.device).float()[:, None, None, :]  # (batch, frames=1, 1, hexals)
        with torch.no_grad():
            self.net.stimulus.zero(x.shape[0], 1)
            self.net.stimulus.add_input(x)
            self._state = self.net.forward(self.net.stimulus(), self.dt, state=self._state, as_states=True)[-1]
        return self._state.nodes.activity
```

- [ ] **Step 4: Run to verify GREEN**

Run: `uv run pytest tests/test_flyvis_eye.py -q -m flyvis`
Expected: 3 passed (first run builds a joblib cache; ~30 s). If `forward` complains about the stimulus buffer device, move the buffer with `self.net.stimulus.buffer = self.net.stimulus.buffer.to(self.device)` after `zero()`.

- [ ] **Step 5: Commit**

```bash
git add src/flyhigh/senses/flyvis_eye.py tests/test_flyvis_eye.py
git commit -m "senses: FlyvisEye — pretrained optic lobe stepped online with persistent state"
```

---

### Task 5: Cell-type map and `ColumnAlignment`

**What you're learning:** two labs, two naming conventions, two hexagonal coordinate frames. flyvis has separate `R1`…`R6`; the male CNS lumps them as `R1-R6`. flyvis's `(u, v)` and the male CNS's `(hex1, hex2)` are both axial hex coordinates, but we don't know the rotation/reflection between them — so we try all 12 lattice symmetries and keep the one where the most flyvis columns land on real male-CNS columns. That is a small piece of genuine research, done with code.

**Files:**
- Create: `src/flyhigh/senses/type_map.py`, `src/flyhigh/senses/alignment.py`, `scripts/build_alignment.py`
- Test: `tests/test_alignment.py`

**Interfaces:**
- Consumes: `Connectome.neurons` columns `index, type, side, hex1, hex2` (M1); `FlyvisEye.types/u/v` (Task 4) — but the aligner takes plain arrays so it is testable without flyvis.
- Produces: `FLYVIS_TO_MCNS: dict[str, str]`; `mcns_type(flyvis_type) -> str | None`; `ColumnAlignment.build(fv_types, fv_u, fv_v, neurons: pl.DataFrame) -> ColumnAlignment` with `.table: pl.DataFrame[fv_index, eye("L"/"R"), lif_index]` (one row per matched flyvis neuron and eye), `.coverage: dict[str, float]` (fraction matched per type), `.transform: tuple[int, ...]` (chosen symmetry), `.save(path)`, `.load(path)`, `.lif_indices(eye) -> np.ndarray[int64] (n_matched,)`, `.fv_indices(eye) -> np.ndarray[int64]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_alignment.py
import numpy as np
import polars as pl

from flyhigh.senses.alignment import ColumnAlignment, HEX_SYMMETRIES, apply_symmetry
from flyhigh.senses.type_map import mcns_type


def test_type_map_merges_r1_to_r6_and_keeps_exact_names():
    assert mcns_type("R1") == "R1-R6" and mcns_type("R6") == "R1-R6"
    assert mcns_type("T4a") == "T4a" and mcns_type("Mi1") == "Mi1"
    assert mcns_type("CT1(Lo1)") == "CT1"


def test_twelve_symmetries_are_distinct_and_invertible():
    u, v = np.array([1, 2, 0]), np.array([0, 1, 3])
    images = {tuple(np.concatenate(apply_symmetry(u, v, s))) for s in HEX_SYMMETRIES}
    assert len(images) == 12


def fake_lattice(rotate: int):
    """A male-CNS-like neuron table: two types on a 7x7 hex patch, rotated by a known symmetry
    and shifted, on both sides. Returns (neurons, fv_types, fv_u, fv_v)."""
    uu, vv = np.meshgrid(np.arange(-3, 4), np.arange(-3, 4)); uu, vv = uu.ravel(), vv.ravel()
    fv_u = np.concatenate([uu, uu]); fv_v = np.concatenate([vv, vv])
    fv_types = np.array(["T4a"] * len(uu) + ["Mi1"] * len(uu))
    ru, rv = apply_symmetry(uu, vv, HEX_SYMMETRIES[rotate])
    rows = []
    idx = 0
    for side in ("L", "R"):
        for t in ("T4a", "Mi1"):
            for a, b in zip(ru + 10, rv + 20):  # shift = unknown lattice centre
                rows.append((idx, t, side, float(a), float(b))); idx += 1
    neurons = pl.DataFrame(rows, schema=["index", "type", "side", "hex1", "hex2"], orient="row")
    return neurons, fv_types, fv_u, fv_v


def test_alignment_recovers_rotation_and_matches_every_column():
    neurons, fv_types, fv_u, fv_v = fake_lattice(rotate=4)
    al = ColumnAlignment.build(fv_types, fv_u, fv_v, neurons)
    assert al.coverage["T4a"] == 1.0 and al.coverage["Mi1"] == 1.0
    assert al.table.height == 2 * len(fv_types)  # both eyes
    # a flyvis T4a at (u,v) maps to the LIF T4a at the rotated+shifted (hex1,hex2) on the right side
    r = al.table.filter(pl.col("eye") == "R").join(neurons, left_on="lif_index", right_on="index")
    assert set(r["side"]) == {"R"} and set(r["type"]) == {"T4a", "Mi1"}


def test_alignment_round_trips_through_parquet(tmp_path):
    neurons, fv_types, fv_u, fv_v = fake_lattice(rotate=1)
    al = ColumnAlignment.build(fv_types, fv_u, fv_v, neurons)
    al.save(tmp_path / "al.parquet")
    al2 = ColumnAlignment.load(tmp_path / "al.parquet")
    assert al2.table.equals(al.table)
    np.testing.assert_array_equal(al2.lif_indices("L"), al.lif_indices("L"))
```

- [ ] **Step 2: Run to verify RED**

Run: `uv run pytest tests/test_alignment.py -q`
Expected: import errors for the two new modules.

- [ ] **Step 3: Implement the type map**

```python
# src/flyhigh/senses/type_map.py
"""flyvis cell-type names → male-CNS `type` names. Exact matches need no entry."""

FLYVIS_TO_MCNS = {
    **{f"R{i}": "R1-R6" for i in range(1, 7)},
    "CT1(Lo1)": "CT1",
    "CT1(M10)": "CT1",
    "Am": "Am1",
}


def mcns_type(flyvis_type: str) -> str:
    return FLYVIS_TO_MCNS.get(flyvis_type, flyvis_type)
```
(Run `scripts/build_alignment.py` in Step 6 to discover any other mismatched names and add them here — the script prints unmatched flyvis types.)

- [ ] **Step 4: Implement the aligner**

```python
# src/flyhigh/senses/alignment.py
"""Map every flyvis neuron (type, u, v) to a male-CNS neuron (type, side, hex1, hex2).

Both lattices are axial hex coordinates. The unknown rotation/reflection between them is one
of the 12 symmetries of a hex lattice; the unknown translation is the lattice centre. We pick
the symmetry+shift that matches the most columns, separately per eye, and keep one table.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from flyhigh.senses.type_map import mcns_type

# The 6 rotations of axial coords (cube-coordinate cycling) and their mirror images.
# Each entry maps (u, v) -> (u', v') via the cube coordinates (x=u, z=v, y=-u-v).
HEX_SYMMETRIES = [
    ("x", "z"), ("-y", "-x"), ("z", "y"), ("-x", "-z"), ("y", "x"), ("-z", "-y"),  # rotations
    ("z", "x"), ("-x", "-y"), ("y", "z"), ("-z", "-x"), ("x", "y"), ("-y", "-z"),  # reflections
]


def apply_symmetry(u, v, sym):
    cube = {"x": u, "z": v, "y": -u - v}
    def pick(name):
        return -cube[name[1]] if name.startswith("-") else cube[name]
    return pick(sym[0]), pick(sym[1])


@dataclass
class ColumnAlignment:
    table: pl.DataFrame  # fv_index, eye, lif_index
    coverage: dict[str, float]
    transform: tuple

    @classmethod
    def build(cls, fv_types, fv_u, fv_v, neurons: pl.DataFrame) -> ColumnAlignment:
        fv_types = np.asarray(fv_types); fv_u = np.asarray(fv_u); fv_v = np.asarray(fv_v)
        target_type = np.array([mcns_type(t) for t in fv_types])
        cols = neurons.filter(pl.col("hex1").is_not_null()).select("index", "type", "side", "hex1", "hex2")
        best = None
        for sym in HEX_SYMMETRIES:
            su, sv = apply_symmetry(fv_u, fv_v, sym)
            rows, n_matched = [], 0
            for eye in ("L", "R"):
                side = cols.filter(pl.col("side") == eye)
                # shift: align lattice centres (flyvis centre is (0,0); male-CNS centre = mean of L1 columns)
                l1 = side.filter(pl.col("type") == "L1")
                if l1.height == 0:
                    l1 = side
                du, dv = round(l1["hex1"].mean()), round(l1["hex2"].mean())
                lut = {(t, int(a), int(b)): i for t, a, b, i in zip(side["type"], side["hex1"], side["hex2"], side["index"])}
                for j in range(len(fv_types)):
                    hit = lut.get((target_type[j], int(su[j]) + du, int(sv[j]) + dv))
                    if hit is not None:
                        rows.append((j, eye, hit)); n_matched += 1
            if best is None or n_matched > best[0]:
                best = (n_matched, sym, rows)
        n_matched, sym, rows = best
        table = pl.DataFrame(rows, schema={"fv_index": pl.Int64, "eye": pl.Utf8, "lif_index": pl.Int64}, orient="row")
        matched = table.filter(pl.col("eye") == "R")["fv_index"].to_numpy()
        coverage = {}
        for t in np.unique(fv_types):
            m = fv_types == t
            coverage[str(t)] = float(np.isin(np.where(m)[0], matched).mean())
        return cls(table, coverage, sym)

    def lif_indices(self, eye: str) -> np.ndarray:
        return self.table.filter(pl.col("eye") == eye)["lif_index"].to_numpy().copy()

    def fv_indices(self, eye: str) -> np.ndarray:
        return self.table.filter(pl.col("eye") == eye)["fv_index"].to_numpy().copy()

    def save(self, path) -> None:
        path = Path(path)
        self.table.write_parquet(path)
        pl.DataFrame({"type": list(self.coverage), "coverage": list(self.coverage.values()),
                      "transform": [",".join(self.transform)] * len(self.coverage)}).write_parquet(path.with_suffix(".meta.parquet"))

    @classmethod
    def load(cls, path) -> ColumnAlignment:
        path = Path(path)
        meta = pl.read_parquet(path.with_suffix(".meta.parquet"))
        return cls(pl.read_parquet(path), dict(zip(meta["type"], meta["coverage"])), tuple(meta["transform"][0].split(",")))
```

- [ ] **Step 5: Run to verify GREEN**

Run: `uv run pytest tests/test_alignment.py -q`
Expected: 4 passed. If the symmetry test finds < 12 distinct images, one tuple in `HEX_SYMMETRIES` is a duplicate — derive them by hand from cube coordinates (rotations cycle x→y→z with sign flips; reflections swap two axes).

- [ ] **Step 6: Build the real alignment and read the report**

```python
# scripts/build_alignment.py
"""Build data/cache/alignment.parquet from the real connectome and the flyvis model; print coverage."""
import polars as pl

from flyhigh.data.connectome import Connectome
from flyhigh.senses.alignment import ColumnAlignment
from flyhigh.senses.flyvis_eye import FlyvisEye

c = Connectome.load("data/raw")
eye = FlyvisEye()
al = ColumnAlignment.build(eye.types, eye.u, eye.v, c.neurons)
al.save("data/cache/alignment.parquet")
print("chosen symmetry:", al.transform, " matched rows:", al.table.height, "of", 2 * eye.n_neurons)
cov = pl.DataFrame({"type": list(al.coverage), "coverage": list(al.coverage.values())}).sort("coverage")
print(cov.head(15)); print("mean coverage:", cov["coverage"].mean())
print("types with zero coverage (add to type_map or accept):", cov.filter(pl.col("coverage") == 0)["type"].to_list())
```
Run: `uv run python scripts/build_alignment.py`
Expected: mean coverage > 0.9 for columnar types with hex coordinates; photoreceptors and a few named-differently types may be 0 → add the correct names to `FLYVIS_TO_MCNS` (find them with `c.neurons["type"].unique()` and a substring search) and re-run until only genuinely absent types remain. Record the chosen symmetry and the coverage table in `docs/03-see-and-move.md` (Task 11).

- [ ] **Step 7: Commit**

```bash
git add src/flyhigh/senses/type_map.py src/flyhigh/senses/alignment.py scripts/build_alignment.py tests/test_alignment.py
git commit -m "senses: flyvis→male-CNS cell-type map and hex-lattice column alignment"
```

---

### Task 6: `driven_only` neurons in the LIF, and `FlyvisBridge`

**What you're learning:** the ownership rule. For every cell type flyvis models, the LIF copy stops listening to its own synapses and fires only because flyvis says so. Downstream neurons can't tell the difference — they just receive spikes. The bridge converts graded activity (arbitrary units) into the LIF's graded drive `ext_i` (mV): subtract the resting activity so grey injects nothing, then scale.

**Files:**
- Modify: `src/flyhigh/brain/lif.py` (constructor + `for_male_cns`)
- Create: `src/flyhigh/senses/bridge.py`
- Test: `tests/test_lif.py` (append), `tests/test_bridge.py`

**Interfaces:**
- Consumes: `ColumnAlignment` (Task 5), `FlyvisEye.rest/types` (Task 4).
- Produces: `LIFBrain(..., driven_only=<int array>)` and `LIFBrain.for_male_cns(..., driven_only=...)`; `FlyvisBridge(alignment, fv_types, rest, n_lif, gains: dict[str, float], default_gain=20.0)` with `.ext_i(activity: torch.Tensor (n_agents*2, n_fv)) -> torch.Tensor (n_agents, n_lif)` where flyvis batch rows are ordered `[agent0-L, agent0-R, agent1-L, agent1-R, …]`; `.driven_indices -> np.ndarray` (all LIF neurons it drives, both eyes).

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_lif.py
def test_driven_only_neurons_ignore_their_synapses_but_still_spike_from_ext_i():
    c = make_connectome(2, [(0, 1, 300)])
    drive = torch.zeros(1, 2); drive[0, 0] = 30.0
    for prop in ("event", "spmv"):
        brain = LIFBrain(c, device="cpu", propagation=prop, driven_only=[1])
        rec = brain.run(3_000, ext_i=drive, recorder=SpikeRecorder())
        assert rec.counts[0, 0] > 0 and rec.counts[0, 1] == 0  # 300-synapse input is cut
        drive2 = drive.clone(); drive2[0, 1] = 30.0
        rec2 = LIFBrain(c, device="cpu", propagation=prop, driven_only=[1]).run(3_000, ext_i=drive2, recorder=SpikeRecorder())
        assert rec2.counts[0, 1] > 0
```
(add `from flyhigh.brain.recorder import SpikeRecorder` at the top of `tests/test_lif.py`.)

```python
# tests/test_bridge.py
import numpy as np
import polars as pl
import torch

from flyhigh.senses.alignment import ColumnAlignment
from flyhigh.senses.bridge import FlyvisBridge


def tiny_alignment():
    # 3 flyvis neurons: two T4a and one Mi1; LIF has 10 neurons; L eye → 0,1,2 ; R eye → 5,6,7
    table = pl.DataFrame({"fv_index": [0, 1, 2, 0, 1, 2], "eye": ["L"] * 3 + ["R"] * 3,
                          "lif_index": [0, 1, 2, 5, 6, 7]})
    return ColumnAlignment(table, {"T4a": 1.0, "Mi1": 1.0}, ("x", "z"))


def test_bridge_subtracts_rest_and_scales_per_type():
    fv_types = np.array(["T4a", "T4a", "Mi1"]); rest = np.array([0.1, 0.1, 0.5])
    b = FlyvisBridge(tiny_alignment(), fv_types, rest, n_lif=10, gains={"T4a": 10.0}, default_gain=2.0)
    act = torch.tensor([[0.6, 0.1, 0.5],   # agent0 L: T4a#0 up by 0.5, others at rest
                        [0.1, 0.1, 1.5],   # agent0 R: Mi1 up by 1.0
                        [0.0, 0.0, 0.0],   # agent1 L: below rest → clamp to 0
                        [0.1, 0.1, 0.5]])  # agent1 R: rest
    ext = b.ext_i(act)
    assert ext.shape == (2, 10)
    assert ext[0, 0].item() == 5.0 and ext[0, 1].item() == 0.0 and ext[0, 7].item() == 2.0
    assert ext[1].abs().sum().item() == 0.0


def test_bridge_driven_indices_cover_both_eyes():
    b = FlyvisBridge(tiny_alignment(), np.array(["T4a", "T4a", "Mi1"]), np.zeros(3), n_lif=10, gains={})
    np.testing.assert_array_equal(np.sort(b.driven_indices), [0, 1, 2, 5, 6, 7])
```

- [ ] **Step 2: Run to verify RED**

Run: `uv run pytest tests/test_lif.py -k driven tests/test_bridge.py -q`
Expected: `TypeError: unexpected keyword 'driven_only'` and `No module named 'flyhigh.senses.bridge'`.

- [ ] **Step 3: Implement `driven_only` in `LIFBrain`**

In `src/flyhigh/brain/lif.py`, add the keyword `driven_only=None` to `__init__` (after `std_exempt`) and, right after the propagation structures are built (end of the `if propagation == "event": … elif "spmv": …` block), add:

```python
        if driven_only is not None and len(driven_only):
            self._cut_incoming(torch.as_tensor(np.asarray(driven_only), dtype=torch.int64, device=self.device))
```
and the method:
```python
    def _cut_incoming(self, idx: torch.Tensor) -> None:
        """Make these neurons driven-only: zero every synapse onto them (their spikes then come
        solely from ext_i / ext_v). Used for the cell types flyvis models."""
        mask = torch.zeros(self.n_neurons, dtype=torch.bool, device=self.device)
        mask[idx] = True
        if self.propagation == "event":
            self._val[mask[self._col]] = 0.0
        else:
            W = self.W.to_sparse_coo().coalesce()
            post = W.indices()[0]
            keep = ~mask[post]
            self.W = torch.sparse_coo_tensor(W.indices()[:, keep], W.values()[keep], W.shape).coalesce().to_sparse_csr()
```
(`import numpy as np` at the top of `lif.py`.) Also pass `driven_only` through `for_male_cns(...)` via its existing `**kw`.

- [ ] **Step 4: Implement the bridge**

```python
# src/flyhigh/senses/bridge.py
"""flyvis activity → graded drive (mV) for the matching LIF neurons."""

from __future__ import annotations

import numpy as np
import torch

from flyhigh.senses.alignment import ColumnAlignment


class FlyvisBridge:
    def __init__(self, alignment: ColumnAlignment, fv_types, rest, n_lif: int, gains: dict[str, float], default_gain: float = 20.0):
        self.n_lif = n_lif
        fv_types = np.asarray(fv_types)
        gain_per_fv = np.array([gains.get(t, default_gain) for t in fv_types], dtype=np.float32)
        self.rest = torch.as_tensor(np.asarray(rest, dtype=np.float32))
        self._fv = {eye: torch.as_tensor(alignment.fv_indices(eye)) for eye in ("L", "R")}
        self._lif = {eye: torch.as_tensor(alignment.lif_indices(eye)) for eye in ("L", "R")}
        self._gain = {eye: torch.as_tensor(gain_per_fv[alignment.fv_indices(eye)]) for eye in ("L", "R")}
        self.driven_indices = np.concatenate([alignment.lif_indices("L"), alignment.lif_indices("R")])

    def ext_i(self, activity: torch.Tensor) -> torch.Tensor:
        """activity rows are [agent0-L, agent0-R, agent1-L, agent1-R, ...]."""
        dev = activity.device
        n_agents = activity.shape[0] // 2
        delta = torch.clamp(activity - self.rest.to(dev), min=0.0)
        out = torch.zeros(n_agents, self.n_lif, device=dev)
        for k, eye in enumerate(("L", "R")):
            rows = delta[k::2]  # (n_agents, n_fv)
            out[:, self._lif[eye].to(dev)] = rows[:, self._fv[eye].to(dev)] * self._gain[eye].to(dev)
        return out
```

- [ ] **Step 5: Run to verify GREEN**

Run: `uv run pytest tests/test_lif.py tests/test_bridge.py -q`
Expected: all pass (previous 38 + 3 new).

- [ ] **Step 6: Commit**

```bash
git add src/flyhigh/brain/lif.py src/flyhigh/senses/bridge.py tests/test_lif.py tests/test_bridge.py
git commit -m "brain: driven_only neurons; senses: FlyvisBridge activity→ext_i"
```

---

### Task 7: `MotorCommand` and `Readout`

**What you're learning:** the fly brain talks to the body through ~1,300 descending neurons (DNs), a bottleneck of a few hundred *types*. Some are famous: the giant fiber (`DNp01`) fires once and the fly jumps; `DNa02` on one side makes it turn. Reading DN firing rates and turning them into a command is the fly-to-drone translation layer, and each rule is a hypothesis we can test.

**Files:**
- Create: `src/flyhigh/motor/__init__.py`, `src/flyhigh/motor/command.py`, `src/flyhigh/motor/readout.py`
- Test: `tests/test_readout.py`

**Interfaces:**
- Consumes: `Connectome.ids_by_type`, `Connectome.neurons["side"]` (M1).
- Produces: `MotorCommand(forward: float, yaw: float, lift: float, escape: bool)`; `ReadoutParams` (all thresholds/gains); `Readout(connectome, params=ReadoutParams())` with `.command(rates: np.ndarray (N,)) -> MotorCommand`, `.commands(rates: np.ndarray (n_agents, N)) -> list[MotorCommand]`, `.watch: np.ndarray[int64]` (union of all neuron indices it reads); pure functions `escape_channel(gf_hz, dnp04_hz, p)`, `yaw_channel(dna_l, dna_r, hs_l, hs_r, p)`, `lift_channel(vs_hz, p)`, `forward_channel(dnp09_hz, escape, p)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_readout.py
import numpy as np
import polars as pl
import pytest

from flyhigh.motor.command import MotorCommand
from flyhigh.motor.readout import Readout, ReadoutParams, escape_channel, forward_channel, lift_channel, yaw_channel
from tests.test_lif import make_connectome

P = ReadoutParams()


def test_escape_requires_giant_fiber_or_gf_plus_dnp04():
    assert escape_channel(gf_hz=60, dnp04_hz=0, p=P) is True
    assert escape_channel(gf_hz=30, dnp04_hz=60, p=P) is True
    assert escape_channel(gf_hz=30, dnp04_hz=10, p=P) is False
    assert escape_channel(gf_hz=0, dnp04_hz=200, p=P) is False


def test_yaw_follows_right_minus_left_and_is_clipped():
    assert yaw_channel(dna_l=0, dna_r=50, hs_l=0, hs_r=0, p=P) > 0
    assert yaw_channel(dna_l=50, dna_r=0, hs_l=0, hs_r=0, p=P) < 0
    assert yaw_channel(dna_l=0, dna_r=0, hs_l=0, hs_r=100, p=P) > 0
    assert -1.0 <= yaw_channel(dna_l=0, dna_r=1000, hs_l=0, hs_r=1000, p=P) <= 1.0
    assert yaw_channel(0, 0, 0, 0, P) == 0.0


def test_lift_and_forward_defaults():
    assert lift_channel(vs_hz=0, p=P) == 0.0
    assert forward_channel(dnp09_hz=0, escape=False, p=P) == pytest.approx(P.forward_bias)
    assert forward_channel(dnp09_hz=100, escape=True, p=P) == 0.0


def readout_connectome():
    c = make_connectome(8, [])
    c.neurons = c.neurons.with_columns(
        pl.Series("type", ["DNp01", "DNp04", "DNa02", "DNa02", "HSN", "HSN", "VS", "DNp09"]),
        pl.Series("side", ["L", "L", "L", "R", "L", "R", "L", "L"]),
    )
    return c


def test_readout_resolves_neurons_and_builds_commands():
    ro = Readout(readout_connectome())
    assert set(ro.watch.tolist()) == set(range(8))
    rates = np.zeros((2, 8))
    rates[0, 0] = 100.0            # agent 0: giant fiber busy
    rates[1, 3] = 80.0             # agent 1: right DNa02
    cmds = ro.commands(rates)
    assert cmds[0] == MotorCommand(forward=0.0, yaw=0.0, lift=0.0, escape=True)
    assert cmds[1].escape is False and cmds[1].yaw > 0 and cmds[1].forward == pytest.approx(P.forward_bias)
```

- [ ] **Step 2: Run to verify RED**

Run: `uv run pytest tests/test_readout.py -q`
Expected: `No module named 'flyhigh.motor'`.

- [ ] **Step 3: Implement**

```python
# src/flyhigh/motor/__init__.py
"""Moving: descending-neuron firing rates → a body-agnostic MotorCommand."""
```

```python
# src/flyhigh/motor/command.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MotorCommand:
    """Normalised, body-frame command. The body layer (sim quadrotor, flybody, real drone)
    turns it into thrust and angles; `escape` may override everything for ~100 ms."""

    forward: float  # -1..1
    yaw: float  # -1..1, + = turn right
    lift: float  # -1..1
    escape: bool

    @classmethod
    def idle(cls, forward_bias: float = 0.2) -> MotorCommand:
        return cls(forward=forward_bias, yaw=0.0, lift=0.0, escape=False)
```

```python
# src/flyhigh/motor/readout.py
"""Hand-mapped channels from descending-neuron rates (Hz, 20 ms window) to MotorCommand.

Each channel is a pure function so it can be tested — and replaced — on its own.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flyhigh.data.connectome import Connectome
from flyhigh.motor.command import MotorCommand


@dataclass(frozen=True)
class ReadoutParams:
    gf_escape_hz: float = 50.0
    gf_assist_hz: float = 20.0
    dnp04_assist_hz: float = 50.0
    k_dn: float = 1.0
    k_hs: float = 1.0
    k_vs: float = 0.01
    k_fwd: float = 0.005
    forward_bias: float = 0.2
    eps_hz: float = 5.0


def escape_channel(gf_hz, dnp04_hz, p: ReadoutParams) -> bool:
    return bool(gf_hz > p.gf_escape_hz or (gf_hz > p.gf_assist_hz and dnp04_hz > p.dnp04_assist_hz))


def yaw_channel(dna_l, dna_r, hs_l, hs_r, p: ReadoutParams) -> float:
    dn = p.k_dn * (dna_r - dna_l) / (dna_r + dna_l + p.eps_hz)
    hs = p.k_hs * (hs_r - hs_l) / (hs_r + hs_l + p.eps_hz)
    return float(np.clip(dn + hs, -1.0, 1.0))


def lift_channel(vs_hz, p: ReadoutParams) -> float:
    return float(np.clip(p.k_vs * vs_hz, -1.0, 1.0))


def forward_channel(dnp09_hz, escape: bool, p: ReadoutParams) -> float:
    if escape:
        return 0.0
    return float(np.clip(p.forward_bias + p.k_fwd * dnp09_hz, -1.0, 1.0))


class Readout:
    def __init__(self, connectome: Connectome, params: ReadoutParams = ReadoutParams()):
        self.p = params
        n = connectome.neurons
        side = n["side"].to_numpy()

        def ids(pattern, s=None):
            idx = connectome.ids_by_type(pattern)
            return idx if s is None else idx[side[idx] == s]

        self.gf = ids(r"^DNp01$")
        self.dnp04 = ids(r"^DNp04$")
        self.dna_l, self.dna_r = ids(r"^DNa0[12]$", "L"), ids(r"^DNa0[12]$", "R")
        self.hs_l, self.hs_r = ids(r"^HS[NES]$", "L"), ids(r"^HS[NES]$", "R")
        self.vs = ids(r"^VS")
        self.dnp09 = ids(r"^DNp09$")
        self.watch = np.unique(np.concatenate([self.gf, self.dnp04, self.dna_l, self.dna_r, self.hs_l, self.hs_r, self.vs, self.dnp09]))

    @staticmethod
    def _mean(rates, idx):
        return float(rates[idx].mean()) if len(idx) else 0.0

    def command(self, rates: np.ndarray) -> MotorCommand:
        m = lambda idx: self._mean(rates, idx)  # noqa: E731
        escape = escape_channel(m(self.gf), m(self.dnp04), self.p)
        return MotorCommand(
            forward=forward_channel(m(self.dnp09), escape, self.p),
            yaw=0.0 if escape else yaw_channel(m(self.dna_l), m(self.dna_r), m(self.hs_l), m(self.hs_r), self.p),
            lift=0.0 if escape else lift_channel(m(self.vs), self.p),
            escape=escape,
        )

    def commands(self, rates: np.ndarray) -> list[MotorCommand]:
        return [self.command(r) for r in rates]
```

- [ ] **Step 4: Run to verify GREEN**

Run: `uv run pytest tests/test_readout.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/flyhigh/motor tests/test_readout.py
git commit -m "motor: MotorCommand and hand-mapped descending-neuron Readout"
```

---

### Task 8: `FlyAgent` — the closed loop

**What you're learning:** the whole sensorimotor loop, in one object: see (sampler → flyvis), think (bridge → LIF, 100 steps), act (readout). Two agents share the GPU as one batch. Timing matters: descending-neuron rates are computed over the last two ticks (20 ms), long enough to count spikes, short enough to react.

**Files:**
- Create: `src/flyhigh/agent.py`
- Test: `tests/test_agent.py` (unit test with fakes) + one `data`+`flyvis` integration test

**Interfaces:**
- Consumes: `EyeSampler`, `FlyvisEye`, `FlyvisBridge`, `LIFBrain`, `Readout` (Tasks 3–7).
- Produces: `FlyAgent(connectome, n_agents=1, gains=None, params=ReadoutParams(), alignment_path="data/cache/alignment.parquet", frame_shape=(180, 360), device="cuda")` with `.tick(frames: list[PanoramicFrame]) -> list[MotorCommand]`, `.dn_rates: np.ndarray (n_agents, N)` (last window, only `readout.watch` entries are meaningful), `.brain_counts_last_tick: np.ndarray (N,)` (spikes of every neuron during the last tick, all agents summed), `.last_activity: torch.Tensor` (flyvis), `.steps_per_tick = 100`; and `FlyAgent.from_parts(sampler, eye, bridge, brain, readout)` for tests.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_agent.py
import numpy as np
import polars as pl
import pytest
import torch

from flyhigh.agent import FlyAgent
from flyhigh.brain.lif import LIFBrain
from flyhigh.motor.readout import Readout
from flyhigh.senses.frame import PanoramicFrame
from tests.test_lif import make_connectome


class FakeSampler:
    def __init__(self, n_eyes=2): self.eyes = [None] * n_eyes
    def sample(self, frame): return np.full((len(self.eyes), 721), float(frame.lum.mean()), dtype=np.float32)


class FakeEye:
    """'flyvis' with 2 neurons per eye; neuron 0 activity = luminance, neuron 1 = 0.5 always."""
    n_neurons = 2; types = np.array(["T4a", "Mi1"]); rest = np.array([0.5, 0.5], dtype=np.float32)
    def reset(self, batch_size): self.batch_size = batch_size
    def step(self, lum): return torch.stack([lum[:, 0], torch.full_like(lum[:, 0], 0.5)], 1)


class FakeBridge:
    """Bright input (activity above rest) drives LIF neuron 0 (the 'GF')."""
    driven_indices = np.array([0])
    def __init__(self, n_lif): self.n_lif = n_lif
    def ext_i(self, activity):
        n_agents = activity.shape[0] // 2
        out = torch.zeros(n_agents, self.n_lif)
        out[:, 0] = torch.clamp(activity[0::2, 0] - 0.5, min=0) * 100.0
        return out


def agent_with_fakes(n_agents=2):
    c = make_connectome(3, [])
    c.neurons = c.neurons.with_columns(pl.Series("type", ["DNp01", "DNp04", "DNa02"]), pl.Series("side", ["L", "L", "R"]))
    brain = LIFBrain(c, n_agents=n_agents, device="cpu")
    return FlyAgent.from_parts(FakeSampler(), FakeEye(), FakeBridge(3), brain, Readout(c))


def test_grey_world_gives_idle_command():
    agent = agent_with_fakes()
    cmds = [agent.tick([PanoramicFrame.grey(), PanoramicFrame.grey()]) for _ in range(3)][-1]
    assert all(not c.escape and c.yaw == 0.0 for c in cmds)


def test_bright_world_for_agent_0_only_triggers_only_its_escape():
    agent = agent_with_fakes()
    for _ in range(3):
        cmds = agent.tick([PanoramicFrame.grey(1.0), PanoramicFrame.grey(0.5)])
    assert cmds[0].escape is True and cmds[1].escape is False
    assert agent.dn_rates.shape == (2, 3) and agent.dn_rates[0, 0] > 50 and agent.dn_rates[1, 0] == 0


@pytest.mark.data
@pytest.mark.flyvis
def test_real_agent_two_ticks_of_grey_are_silent():
    from pathlib import Path
    from flyhigh.data.connectome import Connectome
    if not Path("data/raw").exists() or not Path("data/cache/alignment.parquet").exists():
        pytest.skip("needs real data + alignment table")
    agent = FlyAgent(Connectome.load("data/raw"), n_agents=1)
    cmds = [agent.tick([PanoramicFrame.grey()]) for _ in range(5)][-1]
    assert cmds[0].escape is False and abs(cmds[0].yaw) < 0.05
```

- [ ] **Step 2: Run to verify RED**

Run: `uv run pytest tests/test_agent.py -q`
Expected: `No module named 'flyhigh.agent'`.

- [ ] **Step 3: Implement**

```python
# src/flyhigh/agent.py
"""The closed sensorimotor loop: frames → eyes → flyvis → LIF brain → descending neurons → command."""

from __future__ import annotations

import numpy as np
import torch

from flyhigh.brain.lif import LIFBrain
from flyhigh.data.connectome import Connectome
from flyhigh.motor.command import MotorCommand
from flyhigh.motor.readout import Readout, ReadoutParams
from flyhigh.senses.alignment import ColumnAlignment
from flyhigh.senses.bridge import FlyvisBridge
from flyhigh.senses.eye import EyeGeometry, EyeSampler
from flyhigh.senses.flyvis_eye import FlyvisEye
from flyhigh.senses.frame import PanoramicFrame, TICK_MS

DEFAULT_GAINS: dict[str, float] = {}  # filled by scripts/calibrate_bridge.py (Task 9)
DEFAULT_GAIN = 20.0


class FlyAgent:
    steps_per_tick = int(round(TICK_MS / 0.1))
    window_ticks = 2  # descending-neuron rates over the last 20 ms

    def __init__(self, connectome: Connectome, n_agents: int = 1, gains: dict[str, float] | None = None,
                 params: ReadoutParams = ReadoutParams(), alignment_path="data/cache/alignment.parquet",
                 frame_shape=(180, 360), device="cuda"):
        sampler = EyeSampler(frame_shape, [EyeGeometry("L"), EyeGeometry("R")])
        eye = FlyvisEye()
        eye.reset(batch_size=2 * n_agents)
        alignment = ColumnAlignment.load(alignment_path)
        bridge = FlyvisBridge(alignment, eye.types, eye.rest, connectome.n_neurons, gains or DEFAULT_GAINS, DEFAULT_GAIN)
        brain = LIFBrain.for_male_cns(connectome, n_agents=n_agents, device=device, driven_only=bridge.driven_indices)
        self._init(sampler, eye, bridge, brain, Readout(connectome, params))

    @classmethod
    def from_parts(cls, sampler, eye, bridge, brain, readout) -> "FlyAgent":
        self = cls.__new__(cls)
        eye.reset(batch_size=2 * brain.n_agents)
        self._init(sampler, eye, bridge, brain, readout)
        return self

    def _init(self, sampler, eye, bridge, brain, readout):
        self.sampler, self.eye, self.bridge, self.brain, self.readout = sampler, eye, bridge, brain, readout
        self.n_agents = brain.n_agents
        self.watch = torch.as_tensor(readout.watch, device=brain.device)
        self._counts = [np.zeros((self.n_agents, len(readout.watch))) for _ in range(self.window_ticks)]
        self.dn_rates = np.zeros((self.n_agents, brain.n_neurons))
        self.brain_counts_last_tick = np.zeros(brain.n_neurons)  # all agents summed; for calibration/plots
        self.last_activity = None

    def tick(self, frames: list[PanoramicFrame]) -> list[MotorCommand]:
        assert len(frames) == self.n_agents
        lum = np.concatenate([self.sampler.sample(f) for f in frames])  # (n_agents*2, 721): L, R per agent
        self.last_activity = self.eye.step(torch.as_tensor(lum))
        ext_i = self.bridge.ext_i(self.last_activity).to(self.brain.device)
        counts = torch.zeros(self.n_agents, len(self.readout.watch), device=self.brain.device)
        total = torch.zeros(self.brain.n_neurons, device=self.brain.device)
        for _ in range(self.steps_per_tick):
            spiked = self.brain.step(ext_i=ext_i)
            counts += spiked[:, self.watch]
            total += spiked.sum(0)
        self._counts = self._counts[1:] + [counts.cpu().numpy()]
        self.brain_counts_last_tick = total.cpu().numpy()
        window_s = self.window_ticks * TICK_MS / 1000.0
        self.dn_rates[:] = 0.0
        self.dn_rates[:, self.readout.watch] = sum(self._counts) / window_s
        return self.readout.commands(self.dn_rates)
```

- [ ] **Step 4: Run to verify GREEN**

Run: `uv run pytest tests/test_agent.py -q` (the real-data test runs only if data + alignment exist)
Expected: 2 passed (+1 if data present), rest skipped.

- [ ] **Step 5: Commit**

```bash
git add src/flyhigh/agent.py tests/test_agent.py
git commit -m "agent: FlyAgent closed loop (sampler → flyvis → bridge → LIF → readout), batched"
```

---

### Task 9: Bridge gain calibration

**What you're learning:** the same problem as M1's `w_syn`: flyvis activity is in arbitrary units; the LIF needs mV. We choose the gain by an *effect*: a strong rotating grating should make LIF T4/T5 neurons fire at 50–100 Hz (their measured range), and grey must leave them silent.

**Files:**
- Create: `scripts/calibrate_bridge.py`
- Modify: `src/flyhigh/agent.py` (`DEFAULT_GAINS`, `DEFAULT_GAIN`)

- [ ] **Step 1: Write the sweep script**

```python
# scripts/calibrate_bridge.py
"""Sweep the flyvis→LIF gain: rotating grating → LIF T4/T5 firing rates. Pick the gain giving 50–100 Hz."""
import numpy as np
import torch

from flyhigh.agent import FlyAgent
from flyhigh.data.connectome import Connectome
from flyhigh.senses.frame import PanoramicFrame, rotating_grating

c = Connectome.load("data/raw")
t4t5 = c.ids_by_type(r"^T[45][abcd]$")
lc = c.ids_by_type(r"^(LC4|LPLC2)$")
grating = rotating_grating(wavelength_deg=30, deg_per_s=60, duration_ms=500)
for gain in (5.0, 10.0, 20.0, 40.0):
    agent = FlyAgent(c, n_agents=1, gains={})
    agent.bridge._gain = {eye: torch.full_like(g, gain) for eye, g in agent.bridge._gain.items()}  # override for the sweep
    counts = np.zeros(c.n_neurons)
    for frames, label in (([PanoramicFrame.grey()] * 30, "grey"), (grating, "grating")):
        counts[:] = 0
        for fr in frames:
            agent.tick([fr])
            counts += agent.brain_counts_last_tick
        dur = len(frames) * 0.01
        print(f"gain={gain:5.1f} {label:8s} T4/T5 mean {counts[t4t5].mean()/dur:6.1f} Hz  LC4/LPLC2 {counts[lc].mean()/dur:6.1f} Hz  active {int((counts>0).sum())}")
```

- [ ] **Step 2: Run and choose**

Run: `uv run python scripts/calibrate_bridge.py`
Expected: grey → T4/T5 ≈ 0 Hz at every gain; grating → T4/T5 rising with gain. Pick the smallest gain with T4/T5 in 50–100 Hz. Set `DEFAULT_GAIN` in `agent.py` to it. If T4/T5 saturate (> 300 Hz) while LC4/LPLC2 stay silent, lower the gain for the T4/T5 types only via `DEFAULT_GAINS = {"T4a": …}`. Paste the printed table into `docs/03-see-and-move.md` (Task 11).

- [ ] **Step 3: Commit**

```bash
git add scripts/calibrate_bridge.py src/flyhigh/agent.py
git commit -m "senses: calibrate flyvis→LIF gain (T4/T5 at 50–100 Hz for a 60°/s grating)"
```

---

### Task 10: `scripts/validate_reflexes.py` — the five spec checks

**What you're learning:** this is the experiment a neuroscientist would run on a real fly, done on the model: does it flinch at a looming object, does it turn with the world, does it stay calm when nothing happens? Pass/fail criteria come straight from the spec.

**Files:**
- Create: `scripts/validate_reflexes.py`

- [ ] **Step 1: Write the script**

```python
# scripts/validate_reflexes.py
"""Spec validations for M2. Exit 0 iff all required checks pass."""
import sys
import time

import numpy as np

from flyhigh.agent import FlyAgent
from flyhigh.data.connectome import Connectome
from flyhigh.senses.frame import PanoramicFrame, looming_disc, rotating_grating


def run(agent, frames_per_agent):
    """frames_per_agent: list over agents of frame lists (same length). Returns list of command lists per tick."""
    out = []
    for i in range(len(frames_per_agent[0])):
        out.append(agent.tick([fa[i] for fa in frames_per_agent]))
    return out


def main():
    c = Connectome.load("data/raw")
    gf = c.ids_by_type(r"^DNp01$")
    ok = True

    # 1. silence
    agent = FlyAgent(c, n_agents=1)
    cmds = run(agent, [[PanoramicFrame.grey()] * 100])
    last = cmds[-1][0]
    dn_spikes = sum(agent.dn_rates[0, agent.readout.watch]) > 0
    print(f"[silence]   escape={last.escape} yaw={last.yaw:+.3f} fwd={last.forward:.3f} lift={last.lift:+.3f}  DN active={dn_spikes}")
    ok &= (not last.escape) and abs(last.yaw) < 0.05 and abs(last.forward - 0.2) < 0.05 and abs(last.lift) < 0.05

    # 2. escape: approaching disc on the right; receding disc must not trigger
    agent = FlyAgent(c, n_agents=1)
    frames = [PanoramicFrame.grey()] * 20 + looming_disc(az=60, el=0, start_deg=5, end_deg=60, duration_ms=500)
    cmds = run(agent, [frames])
    first = next((i for i, cm in enumerate(cmds) if cm[0].escape), None)
    size_at = None if first is None else 5 + (60 - 5) * (first - 20) / 49
    print(f"[escape]    first escape tick={first}  disc diameter then={size_at}  GF rate at end={agent.dn_rates[0, gf].mean():.0f} Hz")
    ok &= first is not None and size_at is not None and size_at < 40
    agent = FlyAgent(c, n_agents=1)
    receding = [PanoramicFrame.grey()] * 20 + looming_disc(az=60, el=0, start_deg=60, end_deg=5, duration_ms=500)
    cmds = run(agent, [receding])
    any_escape = any(cm[0].escape for cm in cmds[25:])  # ignore the onset flash
    print(f"[receding]  escape after onset={any_escape}")
    ok &= not any_escape

    # 3. optomotor, both directions
    yaws = {}
    for direction, name in ((+1, "cw"), (-1, "ccw")):
        agent = FlyAgent(c, n_agents=1)
        frames = [PanoramicFrame.grey()] * 20 + rotating_grating(30, 60, 800, direction=direction)
        cmds = run(agent, [frames])
        y = np.array([cm[0].yaw for cm in cmds[40:]])  # after 200 ms of motion
        yaws[name] = y
        print(f"[optomotor] {name}: mean yaw {y.mean():+.3f}, fraction |yaw|>0.3 with right sign = {np.mean(np.sign(y) == direction):.2f}")
    ok &= yaws["cw"].mean() > 0.3 and yaws["ccw"].mean() < -0.3
    ok &= abs(abs(yaws["cw"].mean()) - abs(yaws["ccw"].mean())) < 0.2 * max(abs(yaws["cw"].mean()), abs(yaws["ccw"].mean()))

    # 4. two agents independent
    agent = FlyAgent(c, n_agents=2)
    a0 = [PanoramicFrame.grey()] * 20 + looming_disc(az=60, el=0, start_deg=5, end_deg=60, duration_ms=500)
    a1 = [PanoramicFrame.grey()] * len(a0)
    cmds = run(agent, [a0, a1])
    e0 = any(cm[0].escape for cm in cmds); e1 = any(cm[1].escape for cm in cmds)
    print(f"[two agents] agent0 escape={e0} agent1 escape={e1}")
    ok &= e0 and not e1

    # 5. speed
    for n in (1, 2):
        agent = FlyAgent(c, n_agents=n)
        run(agent, [[PanoramicFrame.grey()] * 5] * n)
        t = time.perf_counter(); run(agent, [[PanoramicFrame.grey()] * 50] * n)
        print(f"[speed]     {n} agent(s): {50 / (time.perf_counter() - t):.1f} ticks/s")

    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and iterate on the readout/gain, never on the checks**

Run: `uv run python scripts/validate_reflexes.py`
Expected (eventually): all five lines sensible, `PASS`. Likely first-run problems and their fixes:
- *No escape:* raise `DEFAULT_GAINS` for `LPLC2`-input types is not possible (they're downstream) → check `agent.dn_rates[0, gf]` and LC4/LPLC2 rates via `agent.brain_counts_last_tick`; if LC4/LPLC2 are silent, the T4/T5 gain is too low; if they fire but GF doesn't, lower `gf_escape_hz` only after confirming GF spikes at all.
- *Optomotor sign wrong:* the HS side→sign mapping in `yaw_channel` is the assumption flagged in the spec — flip `hs_r - hs_l` to `hs_l - hs_r` if **both** directions come out mirrored, and record why in the docs.
- *Escape during the grating:* rotation should not loom; if GF fires, raise `gf_escape_hz` or require `dnp04` co-activation.
Commit each parameter change separately with the observed numbers in the message.

- [ ] **Step 3: Commit**

```bash
git add scripts/validate_reflexes.py src/flyhigh/motor/readout.py src/flyhigh/agent.py
git commit -m "M2 validation: silence, looming escape, optomotor (both directions), two agents, speed"
```

---

### Task 11: Learning track — `docs/03-see-and-move.md`, notebook 03, README

**What you're learning:** writing it down is how you find out whether you understood it.

**Files:**
- Create: `docs/03-see-and-move.md`
- Modify: `scripts/build_notebooks.py` (add `nb03`), `README.md` (M2 row → done, quick-start lines)

- [ ] **Step 1: Write the doc** with these sections (each 100–300 words, numbers from your runs):
  1. *How a fly sees* — ommatidia, ~5° columns, two eyes ≈ 360°, the lamina ON/OFF split (L1/L2), T4/T5 direction selectivity, why looming is "expansion" and why HS/VS cells see rotation.
  2. *Why flyvis in front of the LIF* — graded vs spiking, the timing problem, the ownership rule, with the coverage table and chosen lattice symmetry from `scripts/build_alignment.py`.
  3. *The bridge and its calibration* — the gain sweep table from Task 9.
  4. *Descending neurons* — the bottleneck, the four channels, the HS sign finding.
  5. *Results* — the five validation lines verbatim, and the honest limits (what didn't work, what is hand-tuned).

- [ ] **Step 2: Add notebook 03 to `scripts/build_notebooks.py`** with cells: build a `FlyAgent`; show `EyeSampler.sample` of a looming frame as a hex scatter (use `flyvis.analysis.visualization.plots.quick_hex_scatter`); flyvis T4a/T4b activity maps for cw vs ccw gratings; DN raster (GF, DNp04, DNa02 L/R, HSN L/R) during the looming disc; `yaw`/`escape` traces over time for the three stimuli; the speed cell. Same plotting conventions as notebooks 01/02 (the `STYLE` block).

Run: `uv run python scripts/build_notebooks.py && uv run jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=3600 notebooks/03_see_and_move.ipynb`
Expected: executes with zero error outputs (check with the same JSON scan used in M1).

- [ ] **Step 3: Update README** — M2 row `✅ done`, add `uv run flyvis download-pretrained`, `scripts/build_alignment.py`, `scripts/validate_reflexes.py` to the quick start, and the `flyhigh.senses / motor / agent` lines to the layout.

- [ ] **Step 4: Full verification and commit**

Run: `uv run ruff check src tests scripts && uv run pytest -q && uv run python scripts/validate_reflexes.py`
Expected: ruff clean, all tests pass (data/flyvis ones included), `PASS`.
```bash
git add docs/03-see-and-move.md scripts/build_notebooks.py notebooks/03_see_and_move.ipynb README.md
git commit -m "M2 complete: docs, notebook 03, README"
```

---

## Self-review against the spec

- Spec §Architecture: PanoramicFrame (T2), EyeGeometry/EyeSampler (T3), FlyvisEye online (T4), ColumnAlignment + type map (T5), FlyvisBridge + driven_only (T6), MotorCommand/Readout channels (T7), FlyAgent tick with 100 LIF steps and 20 ms window (T8), stimuli (T2). ✔
- Spec §Validation 1–5: all in T10; unit tests for sampler, alignment, bridge, channels, driven_only in T3/T5/T6/T7. ✔
- Spec §Environment: Python 3.12, flyvis dependency, `.env`, download in T1 (the spec's `download --flyvis` flag is dropped in favour of the documented `flyvis download-pretrained` command — README says so). ✔
- Spec §Learning track: T11. ✔
- Type consistency: `FlyvisEye.step(lum (batch,721)) -> (batch, n)`; `FlyvisBridge.ext_i(activity (2A, n)) -> (A, N)` with rows `[a0-L, a0-R, …]`, matching `FlyAgent.tick`'s `np.concatenate` of per-agent `(2, 721)` samples; `Readout.commands(rates (A, N))`; `ColumnAlignment.lif_indices/fv_indices(eye)` used by the bridge. ✔
- Known deliberate simplification: `rest` is per neuron (not per type) — spec allows either.
