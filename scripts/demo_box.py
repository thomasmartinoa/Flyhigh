# scripts/demo_box.py
"""Watch the brain react: annotated videos of the box experiments.

    uv run python scripts/demo_box.py                 # both scenes
    uv run python scripts/demo_box.py obstacle        # an obstacle comes at a hovering body
    uv run python scripts/demo_box.py encounter       # two bodies fly at each other
    uv run python scripts/demo_box.py obstacle --raw  # without the escape refractory

Videos land in data/runs/. Each frame shows the room, the panorama the brain is being fed,
the giant fiber's rate and the motor command, so the reaction is visible rather than inferred.
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

from flyhigh.agent import FlyAgent
from flyhigh.data.connectome import Connectome
from flyhigh.motor.readout import ReadoutParams
from flyhigh.sim import Simulation
from flyhigh.viz import AnnotatedVideo

OUT = Path("data/runs")
REFRACTORY_MS = 300.0  # one escape, then the channel is deaf while the manoeuvre plays out


def report(log: pl.DataFrame, agent: int, label: str) -> None:
    a = log.filter(pl.col("agent") == agent)
    esc = a.filter(pl.col("escape"))
    first = esc["t_ms"][0] if esc.height else None
    print(f"  {label}: escapes {esc.height}"
          + (f", first at t = {first:.0f} ms" if first is not None else "")
          + (f", hand {esc['hand_dist'][0]:.2f} m away then" if first is not None and "hand_dist" in a.columns else "")
          + f" | height {a['z'].min():.2f}..{a['z'].max():.2f} m, GF peak {a['gf_hz'].max():.0f} Hz")


def obstacle(c: Connectome, params: ReadoutParams, tag: str) -> None:
    """A hovering body; an obstacle approaches at 1 m/s and stops just short."""
    sim = Simulation(FlyAgent(c, n_agents=1, params=params), n_agents=1)
    video = AnnotatedVideo(sim, OUT / f"demo_obstacle{tag}.mp4", title="obstacle approach")
    sim.run(0.6, on_tick=video)
    sim.hand.approach(lambda b=sim.bodies[0]: b.pos)
    log = sim.run(3.4, on_tick=video)
    video.close(); sim.close()
    print(f"[obstacle{tag}] {OUT / f'demo_obstacle{tag}.mp4'}")
    report(log, 0, "body")


def encounter(c: Connectome, params: ReadoutParams, tag: str) -> None:
    """Two bodies cruising at each other: whoever sees the other loom first gets out of the way."""
    sim = Simulation(FlyAgent(c, n_agents=2, params=params), n_agents=2, settle_s=0.0)
    sim.hand.park()
    sim.settle(0.5)
    video = AnnotatedVideo(sim, OUT / f"demo_encounter{tag}.mp4", title="two bodies, one room")
    log = sim.run(2.6, on_tick=video)
    video.close(); sim.close()
    d = np.linalg.norm(log.filter(pl.col("agent") == 0).select("x", "y", "z").to_numpy()
                       - log.filter(pl.col("agent") == 1).select("x", "y", "z").to_numpy(), axis=1)
    print(f"[encounter{tag}] {OUT / f'demo_encounter{tag}.mp4'}   closest approach {d.min():.2f} m")
    for i in (0, 1):
        report(log, i, f"body {i}")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    raw = "--raw" in sys.argv
    scenes = args or ["obstacle", "encounter"]
    OUT.mkdir(parents=True, exist_ok=True)
    c = Connectome.load("data/raw")
    params = ReadoutParams(escape_refractory_ms=0.0 if raw else REFRACTORY_MS)
    tag = "_raw" if raw else ""
    print(f"escape refractory: {params.escape_refractory_ms:.0f} ms")
    for scene in scenes:
        {"obstacle": obstacle, "encounter": encounter}[scene](c, params, tag)


if __name__ == "__main__":
    main()
