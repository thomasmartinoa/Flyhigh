"""Generate the teaching notebooks in notebooks/ (run from the repo root).

    uv run python scripts/build_notebooks.py
    uv run jupyter nbconvert --execute --to notebook --inplace notebooks/0*.ipynb
"""

from pathlib import Path

import nbformat as nbf

OUT = Path("notebooks")

STYLE = '''import matplotlib as mpl, matplotlib.pyplot as plt
# dataviz conventions: fixed categorical order, single-hue sequential, thin marks, recessive grid
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
mpl.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6,
                     "axes.prop_cycle": mpl.cycler(color=C), "lines.linewidth": 2, "font.size": 10})'''


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


def write(name, cells):
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    OUT.mkdir(exist_ok=True)
    nbf.write(nb, OUT / name)
    print("wrote", OUT / name)


# --------------------------------------------------------------------------- 01
nb01 = [
    md("""
# 01 — Meet the connectome

The male *Drosophila* CNS connectome: **165,122 neurons, 124 million synapses**, brain + eyes + ventral nerve cord.
This notebook loads it with `flyhigh` and pokes at it. Read `docs/01-connectome.md` alongside.

Run `uv run python -m flyhigh.data.download` once first (1.1 GB)."""),
    code("""
import sys; sys.path.insert(0, "..")
import numpy as np, polars as pl
from flyhigh.data.connectome import Connectome
""" + STYLE + """
pl.Config.set_tbl_rows(12)
c = Connectome.load("../data/raw")   # ~8 s the first time, ~1 s from the parquet cache
c.n_neurons, c.n_edges, int(c.edges["syn_count"].sum())"""),
    md("""
## What is in the table
One row per neuron. `index` is our dense id (used in the edge list), `body_id` is Janelia's id (use it in neuPrint / Neuroglancer)."""),
    code("c.neurons.head(5)"),
    md("## Where the neurons live\nTwo thirds of all neurons are in the optic lobes. `superclass` is the dataset's coarse cell class; `region` is our grouping of it."),
    code("""
d = c.describe()
by_sc = (pl.DataFrame({"superclass": list(d["by_superclass"]), "n": list(d["by_superclass"].values())})
         .filter(pl.col("superclass").is_not_null()).sort("n", descending=True).head(14))
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.barh(by_sc["superclass"][::-1], by_sc["n"][::-1], color=C[0], height=0.6)
for y, n in enumerate(by_sc["n"][::-1]): ax.text(n + 800, y, f"{n:,}", va="center", fontsize=8, color="#52514e")
ax.set_xlabel("neurons"); ax.set_title("Neurons per superclass (top 14)"); ax.grid(axis="y", visible=False)
plt.tight_layout()"""),
    md("""
## Neurotransmitters → sign of a connection
The dataset predicts each neuron's main transmitter from its synapse images. We map ACh/DA/OA/5-HT → **excitatory (+1)** and GABA/Glu/histamine → **inhibitory (−1)**. About 62 % of neurons are cholinergic."""),
    code("""
nt = c.neurons.group_by("nt", "nt_sign").len().sort("len", descending=True)
nt.with_columns((pl.col("len") / c.n_neurons * 100).round(1).alias("%"))"""),
    md("""
## How connected is a neuron?
Synapse counts per neuron are heavy-tailed: most neurons have a few hundred input synapses, a few have tens of thousands. Log axes are the only way to see this."""),
    code("""
in_syn = c.edges.group_by("post_idx").agg(pl.col("syn_count").sum().alias("in"))
out_syn = c.edges.group_by("pre_idx").agg(pl.col("syn_count").sum().alias("out"))
deg = c.neurons.join(in_syn, left_on="index", right_on="post_idx", how="left").join(out_syn, left_on="index", right_on="pre_idx", how="left").fill_null(0)
fig, ax = plt.subplots(figsize=(6.5, 4))
bins = np.logspace(1, 5, 50)   # neurons with < 10 synapses are fragments; hide the integer spikes
for col, color, label in (("in", C[0], "input synapses"), ("out", C[1], "output synapses")):
    ax.hist(deg[col].to_numpy() + 1, bins=bins, histtype="step", linewidth=2, color=color, label=label)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("synapses per neuron (+1)"); ax.set_ylabel("neurons")
ax.legend(frameon=False); ax.set_title("Synapses per neuron — heavy tails")
print("mean input synapses per neuron:", round(deg["in"].mean()))
deg.sort("in", descending=True).select("type", "superclass", "in", "out").head(8)"""),
    md("""
## Famous neurons
Cell types are shared with FlyWire / the hemibrain, so decades of physiology map onto these ids. A few we will use:"""),
    code("""
famous = {"DNp01": "giant fiber — escape take-off", "LC4": "looming detector", "LPLC2": "looming detector",
          "HSN": "horizontal-motion tangential cell", "T4a": "motion detector (front→back)", "R1-R6": "photoreceptors",
          "DNa02": "steering descending neuron", "LB3b": "sugar taste neuron (Gr64f)", "MN9": "proboscis motor neuron",
          "APL": "the one inhibitory neuron that keeps memory sparse", "KCg-m": "Kenyon cell (memory)"}
rows = [(t, len(c.ids_by_type(rf"^{t}$")), c.neurons.filter(pl.col("type") == t)["nt"][0], what) for t, what in famous.items()]
pl.DataFrame(rows, schema=["type", "count", "nt", "what it does"], orient="row")"""),
    md("""
## Tracing a circuit: looming → giant fiber
The escape reflex: looming-sensitive visual neurons (LC4, LPLC2) synapse **directly** onto the giant fiber (DNp01), which drives the take-off jump. How strong is that connection?"""),
    code("""
gf = c.ids_by_type(r"^DNp01$")
inputs = (c.edges.filter(pl.col("post_idx").is_in(pl.Series(gf).implode()))
          .join(c.neurons.select(pl.col("index").alias("pre_idx"), "type", "nt", "side"), on="pre_idx")
          .group_by("type", "nt").agg(pl.col("syn_count").sum().alias("synapses"), pl.len().alias("cells"))
          .sort("synapses", descending=True))
inputs.head(12)"""),
    md("""
## Sex-specific circuitry
This is a *male* connectome, and the annotation marks neurons that are male-specific or differ from the female (the headline finding of the Cell paper). Where are they?"""),
    code("""
dim = (c.neurons.filter(pl.col("dimorphism").is_not_null())
       .group_by("dimorphism", "region").len().sort("len", descending=True))
piv = dim.pivot(on="region", index="dimorphism", values="len").fill_null(0)
fig, ax = plt.subplots(figsize=(6.5, 3.2))
regions = [r for r in ("brain", "vnc", "optic") if r in piv.columns]
left = np.zeros(piv.height)
for i, r in enumerate(regions):
    vals = piv[r].to_numpy(); ax.barh(piv["dimorphism"], vals, left=left, color=C[i], label=r, height=0.55, edgecolor="white", linewidth=2); left += vals
ax.legend(frameon=False, ncol=3); ax.grid(axis="y", visible=False); ax.set_xlabel("neurons"); ax.set_title("Sex-specific and dimorphic neurons by region")
plt.tight_layout()"""),
    md("""
## The eye has coordinates
The lamina/medulla columnar neurons (L1, L2, L5, Tm1, …) carry hexagonal eye-column coordinates (`hex1`, `hex2`): one L1 per ommatidium, ≈ 880 columns per eye. That is our map from camera pixels to the visual system in milestone 2."""),
    code("""
eye = c.neurons.filter((pl.col("type") == "L1") & pl.col("hex1").is_not_null())
fig, ax = plt.subplots(figsize=(7, 4))
for i, side in enumerate(("L", "R")):
    g = eye.filter(pl.col("side") == side)
    ax.scatter(g["hex1"] + (0 if side == "L" else 45), g["hex2"], s=5, color=C[i], label=f"{side} eye ({g.height} columns)")
ax.set_aspect("equal"); ax.set_xlabel("hex1 (right eye shifted +45)"); ax.set_ylabel("hex2"); ax.set_title("L1 lamina neurons = eye columns")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
plt.tight_layout()"""),
    md("""
## Try it
- `c.ids_by_type(r"^DNa")` — all the DNa steering neurons.
- `c.subgraph(region="brain")` — the central brain only (38,546 neurons).
- Change `gf` above to `c.ids_by_type(r"^MN9$")` to see who talks to the feeding motor neuron."""),
]
write("01_meet_the_connectome.ipynb", nb01)

# --------------------------------------------------------------------------- 02
nb02 = [
    md("""
# 02 — Run a fly brain

Every neuron becomes a leaky integrate-and-fire unit (Shiu et al. 2024 parameters), every synapse a signed weight,
and the whole thing steps at 0.1 ms on the GPU. Read `docs/02-lif-brain.md` for the model and how we calibrated it."""),
    code("""
import sys; sys.path.insert(0, "..")
import numpy as np, polars as pl, torch
from flyhigh.data.connectome import Connectome
from flyhigh.brain.lif import LIFBrain
from flyhigh.brain.stimulus import PoissonActivation
from flyhigh.brain.recorder import SpikeRecorder
""" + STYLE + """
c = Connectome.load("../data/raw")
MAX_HZ = 1000 / 2.3   # refractory ceiling

def run(activations, silence=None, duration_ms=1000, record=None, seed=0):
    torch.manual_seed(seed)
    brain = LIFBrain.for_male_cns(c, device="cuda")
    if silence is not None: brain.silence(silence)
    rec = brain.run(int(duration_ms / 0.1), stimuli=[PoissonActivation(i, hz) for i, hz in activations],
                    recorder=SpikeRecorder(neuron_idx=record))
    return rec, rec.rates(duration_ms)[0]

def top_types(rates, exclude=()):
    df = c.neurons.with_columns(pl.Series("rate", rates)).filter(pl.col("rate") > 5)
    df = df.filter(~pl.col("type").str.contains("|".join(f"^{t}" for t in exclude) or "^$").fill_null(False))
    return df.group_by("type", "nt").agg(pl.len().alias("cells"), pl.col("rate").mean().round(0).alias("Hz")).sort("Hz", descending=True)"""),
    md("## 1. A quiet brain\nWith no input the model is silent: there is no spontaneous activity, so everything you see later is caused by the input you give."),
    code("""
rec, r = run([], duration_ms=300)
print("spikes with no input:", rec.counts.sum().item())"""),
    md("""
## 2. Taste → feeding
Activate the 51 sugar-sensing labellar neurons (`LB3a/b/c`) at 200 Hz — the model's stand-in for putting sugar on the fly's mouth — and watch who fires."""),
    code("""
lb3 = c.ids_by_type(r"^LB3[abc]$"); mn9 = c.ids_by_type(r"^MN9$")
rec, r = run([(lb3, 200)], record=None)      # record every spike (fine for 1 s)
print(f"active neurons: {(rec.counts[0] > 0).sum().item():,} of {c.n_neurons:,}   MN9 (feeding motor neuron) L/R: {r[mn9[0]]:.0f} / {r[mn9[1]]:.0f} Hz")
top_types(r, exclude=("LB3",)).head(12)"""),
    md("""
Those are the paper's second-order taste neurons — GNG038, *Clavicle* (`ANXXX462a`), *Quasimodo* (`GNG042`), GNG175 — and the descending neuron DNg67.
A raster shows the timing: the GRNs (bottom) drive the interneurons within 5 ms, and MN9 fires sparsely."""),
    code("""
spk = rec.to_polars().join(c.neurons.select(pl.col("index").alias("neuron"), "type"), on="neuron")
groups = [("LB3 (input)", r"^LB3"), ("GNG038", r"^GNG038$"), ("Clavicle ANXXX462a", r"^ANXXX462a$"), ("Quasimodo GNG042", r"^GNG042$"), ("DNg67", r"^DNg67$"), ("MN9", r"^MN9$")]
fig, ax = plt.subplots(figsize=(9, 4.2))
y = 0; ticks = []
for i, (label, pat) in enumerate(groups):
    s = spk.filter(pl.col("type").str.contains(pat).fill_null(False))
    ids = {n: k for k, n in enumerate(sorted(s["neuron"].unique()))}
    ax.scatter(s["t_ms"], [y + ids[n] for n in s["neuron"]], s=2, color=C[i % len(C)], marker="|")
    ticks.append((y + len(ids) / 2, label)); y += max(len(ids), 1) + 3
ax.set_yticks([t for t, _ in ticks]); ax.set_yticklabels([l for _, l in ticks]); ax.set_xlim(0, 300); ax.set_xlabel("time (ms)")
ax.set_title("Sugar neurons → feeding circuit (first 300 ms)"); ax.grid(axis="y", visible=False)
plt.tight_layout()"""),
    md("""
## 3. Silencing a neuron
Connectome models are for *what-if* experiments. The gustatory paper predicts that **Clavicle** is a key relay to MN9. Silence it (its outgoing synapses set to zero) and compare."""),
    code("""
clav = c.ids_by_type(r"^ANXXX462a$")
_, r_cut = run([(lb3, 200)], silence=clav)
print(f"MN9 rate: intact {r[mn9].mean():.1f} Hz   Clavicle silenced {r_cut[mn9].mean():.1f} Hz")"""),
    md("""
At our calibration MN9 is barely above threshold, so this prediction does not reproduce cleanly (see `docs/02-lif-brain.md`). Honest modelling means reporting that.

## 4. Looming → escape
The circuit we will drive from a camera: looming-sensitive visual projection neurons **LC4 + LPLC2** synapse directly onto the **giant fiber** (`DNp01`)."""),
    code("""
lc4 = c.ids_by_type(r"^LC4$"); lplc2 = c.ids_by_type(r"^LPLC2$"); gf = c.ids_by_type(r"^DNp01$")
rec, r = run([(lc4, 100), (lplc2, 100)], duration_ms=300, record=None)
first = rec.to_polars().filter(pl.col("neuron").is_in(pl.Series(gf).implode()))["t_ms"].min()
print(f"giant fiber: first spike at {first} ms, rate L/R {r[gf[0]]:.0f}/{r[gf[1]]:.0f} Hz;  active neurons {(rec.counts[0] > 0).sum().item():,}")
top_types(r, exclude=("LC4", "LPLC2")).head(8)"""),
    md("## 5. Which descending neurons respond?\nDescending neurons are the brain's only output to the body. Rank them for the looming stimulus — this is the readout the motor module (M2) will use."),
    code("""
dn = c.neurons.filter(pl.col("superclass") == "descending_neuron").with_columns(pl.Series("rate", r[c.neurons.filter(pl.col("superclass") == "descending_neuron")["index"].to_numpy()]))
top = dn.group_by("type").agg(pl.col("rate").mean().alias("Hz")).sort("Hz", descending=True).head(12)
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.barh(top["type"][::-1], top["Hz"][::-1], color=C[0], height=0.6)
ax.set_xlabel("mean rate (Hz)"); ax.set_title("Descending neurons responding to looming"); ax.grid(axis="y", visible=False)
plt.tight_layout()"""),
    md("""
## 6. How fast is it?
One step = 0.1 ms of brain time. Two agents cost the same as one because the batch is a tensor dimension."""),
    code("""
import time
for A in (1, 2):
    b = LIFBrain.for_male_cns(c, n_agents=A, device="cuda")
    drive = (torch.rand(A, c.n_neurons, device="cuda") < 0.02).float() * 12.0
    for _ in range(100): b.step(ext_i=drive)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(500): b.step(ext_i=drive)
    torch.cuda.synchronize(); ms = (time.perf_counter() - t) / 500 * 1e3
    print(f"{A} agent(s): {ms:.2f} ms per 0.1 ms step  →  {0.1 / ms:.3f}× real time")"""),
    md("""
## Try it
- Activate `DNp01` itself and see what happens downstream in the VNC (wing and leg motor neurons `MN*`).
- Change `200` Hz to `50` Hz in section 2: taste responses are graded.
- `run([(c.ids_by_type(r"^R1-R6$"), 20)])` — drive all photoreceptors; the optic lobe is a very different beast (M2)."""),
]
write("02_run_a_fly_brain.ipynb", nb02)

# --------------------------------------------------------------------------- 03
nb03 = [
    md("""
# 03 — See & move

A picture of the world goes in, a motor command comes out: `PanoramicFrame → EyeSampler → flyvis → FlyvisBridge → LIFBrain → Readout`.
Read `docs/03-see-and-move.md` alongside — especially section 5 for what works (the optomotor turn) and what does not yet (the looming escape).

Needs `data/raw`, the pretrained flyvis models (`uv run flyvis download-pretrained`) and `data/cache/alignment.parquet` (`uv run python scripts/build_alignment.py`)."""),
    code("""
import os, sys; sys.path.insert(0, "..")
os.environ["FLYVIS_ROOT_DIR"] = os.path.abspath("../data/flyvis")   # .env is relative to the repo root
import numpy as np, polars as pl, torch
from flyhigh.data.connectome import Connectome
from flyhigh.agent import FlyAgent
from flyhigh.senses.frame import PanoramicFrame, looming_disc, rotating_grating
from flyvis.analysis.visualization.plots import hex_scatter
""" + STYLE + """
c = Connectome.load("../data/raw")
agent = FlyAgent(c, n_agents=1, alignment_path="../data/cache/alignment.parquet")
print(f"driven LIF neurons: {len(agent.bridge.driven_indices):,}   flyvis neurons per eye: {agent.eye.n_neurons:,}")

LC = c.ids_by_type(r"^(LC4|LPLC2)$")

def play(frames, watch=()):
    \"\"\"Run frames after 300 ms of grey. Returns per-tick commands, spike counts of `watch` neurons, and LC4+LPLC2 spikes.\"\"\"
    for _ in range(30): agent.tick([PanoramicFrame.grey()])
    cmds, spikes, lc = [], [], []
    for fr in frames:
        cmds.append(agent.tick([fr])[0]); spikes.append(agent.brain_counts_last_tick[list(watch)].copy()); lc.append(agent.brain_counts_last_tick[LC].sum())
    return cmds, np.array(spikes), np.array(lc)"""),
    md("""
## 1. What the eyes see
A looming disc at azimuth +60° (right of straight ahead), sampled onto the two 721-column eyes. Each hexagon is one column; the right eye looks at the disc, the left sees grey."""),
    code("""
frame = looming_disc(az=60, el=0, start_deg=5, end_deg=60, duration_ms=500)[-1]
lum = agent.sampler.sample(frame)
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), gridspec_kw={"width_ratios": [2.2, 1, 1]})
axes[0].imshow(frame.lum, cmap="gray", vmin=0, vmax=1, extent=[-180, 180, -90, 90]); axes[0].set_title("PanoramicFrame (az × el)"); axes[0].grid(False)
for ax, eye, name in zip(axes[1:], agent.sampler.eyes, ("left eye", "right eye")):
    hex_scatter(eye.u, eye.v, lum[0 if name == "left eye" else 1], fig=fig, ax=ax, cmap=plt.get_cmap("gray"), vmin=0, vmax=1, cbar=False, edgecolor="#bbb", edgewidth=0.2)
    ax.set_title(name); ax.axis("off")
plt.tight_layout()"""),
    md("""
## 2. flyvis computes direction
Run a rotating grating through the eyes and look at the T4a (front-to-back) and T4b (back-to-front) maps on the right eye. Clockwise rotation is front-to-back on the right eye, so T4a lights up; anticlockwise flips it."""),
    code("""
def flyvis_map(frames, cell_type, eye_row=1):
    for _ in range(30): agent.eye.step(torch.full((2, 721), 0.5, device=agent.eye.device))
    acts = []
    for fr in frames:
        a = agent.eye.step(torch.as_tensor(agent.sampler.sample(fr), device=agent.eye.device))
        acts.append(a.cpu().numpy()[eye_row] - agent.eye.rest)
    m = agent.eye.types == cell_type
    return np.mean(acts[20:], axis=0)[m], agent.eye.u[m], agent.eye.v[m]

fig, axes = plt.subplots(2, 2, figsize=(6, 6))
for i, (d, name) in enumerate(((+1, "clockwise"), (-1, "anticlockwise"))):
    frames = rotating_grating(30, 60, 400, direction=d)
    for j, t in enumerate(("T4a", "T4b")):
        vals, u, v = flyvis_map(frames, t)
        hex_scatter(u, v, vals, fig=fig, ax=axes[i, j], cmap=plt.get_cmap("Blues"), vmin=0, vmax=1.2, cbar=False, edgecolor=None)
        axes[i, j].set_title(f"{name}: {t}  (mean {vals.mean():+.2f})", fontsize=10); axes[i, j].axis("off")
plt.suptitle("flyvis activity − rest, right eye", y=1.0); plt.tight_layout()"""),
    md("""
## 3. Descending neurons during a looming disc
Spikes per 10 ms tick of the neurons the readout watches. The giant fiber (`DNp01`) should fire before the disc reaches 40° — in this brain it does not: LC4/LPLC2 are under-driven (see the doc, section 5)."""),
    code("""
side = c.neurons["side"].to_numpy()
def one(pattern, s): idx = c.ids_by_type(pattern); return int(idx[side[idx] == s][0])
watch = {"GF L": one(r"^DNp01$", "L"), "GF R": one(r"^DNp01$", "R"), "DNp04 R": one(r"^DNp04$", "R"),
         "DNa02 L": one(r"^DNa02$", "L"), "DNa02 R": one(r"^DNa02$", "R"), "HSN L": one(r"^HSN$", "L"), "HSN R": one(r"^HSN$", "R")}
loom = [PanoramicFrame.grey()] * 20 + looming_disc(az=60, el=0, start_deg=5, end_deg=60, duration_ms=500)
cmds, spikes, lc = play(loom, watch.values())
fig, ax = plt.subplots(figsize=(9, 3.2))
for k, (name, _) in enumerate(watch.items()):
    t = np.nonzero(spikes[:, k])[0]
    ax.scatter(t * 10, np.full_like(t, k), s=12 * spikes[t, k], marker="|", color=C[k % len(C)])
ax.set_yticks(range(len(watch))); ax.set_yticklabels(list(watch)); ax.axvline(200, color="#999", lw=1, ls="--")
ax.set_xlabel("time (ms) — disc starts expanding at 200 ms"); ax.set_title("Descending / tangential neurons during a looming disc"); ax.grid(axis="y", visible=False)
plt.tight_layout()
print(f"escape ticks: {sum(cm.escape for cm in cmds)}   LC4+LPLC2 spikes during the loom: {int(lc.sum())} (311 cells, 500 ms)")"""),
    md("""
## 4. The commands
`yaw` and `escape` over time for grey, the looming disc, and a clockwise grating. The optomotor response is the fly turning *with* the scene (+ = right)."""),
    code("""
stims = {"grey": [PanoramicFrame.grey()] * 70, "looming disc (right)": loom,
         "grating clockwise": [PanoramicFrame.grey()] * 20 + rotating_grating(30, 60, 500, +1),
         "grating anticlockwise": [PanoramicFrame.grey()] * 20 + rotating_grating(30, 60, 500, -1)}
fig, axes = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True)
for k, (name, frames) in enumerate(stims.items()):
    cmds, _, _ = play(frames)
    t = np.arange(len(cmds)) * 10
    axes[0].plot(t, [cm.yaw for cm in cmds], label=name, color=C[k]); axes[1].plot(t, [float(cm.escape) for cm in cmds], color=C[k])
axes[0].set_ylabel("yaw (+ = right)"); axes[0].set_ylim(-1.05, 1.05); axes[0].legend(loc="lower right", fontsize=8)
axes[1].set_ylabel("escape"); axes[1].set_xlabel("time (ms) — stimulus starts at 200 ms"); axes[1].set_ylim(-0.05, 1.05)
plt.tight_layout()"""),
    md("""
## 5. How fast is it?
One tick = one 10 ms frame = 1 flyvis step + 100 LIF steps + the readout."""),
    code("""
import time
frames = [PanoramicFrame.grey()] * 50
play(frames[:5])
torch.cuda.synchronize(); t = time.perf_counter(); play(frames); torch.cuda.synchronize()
tps = 50 / (time.perf_counter() - t)
print(f"1 agent: {tps:.1f} ticks/s  →  {tps / 100:.2f}× real time (100 Hz camera)")"""),
    md("""
## Try it
- Move the disc: `looming_disc(az=-60, ...)` — the left eye's HS/T4 should mirror everything.
- `FlyAgent(c, gains={"T4a": 40, "T4b": 40, ...})` — raise the T4/T5 gain and watch the giant fiber start firing for the *grating* (section 3 of the doc).
- Swap ensemble member: `FlyvisEye(model="flow/0000/007")` — flyvis has 50; T4d is weak in member 0."""),
]
write("03_see_and_move.ipynb", nb03)
