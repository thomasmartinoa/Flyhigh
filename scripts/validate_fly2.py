# scripts/validate_fly2.py
"""M4b: two flybody flies with two brains in one world. Exit 0 iff all pass.

At real fly scale another fly subtends one ommatidium (5°) beyond ~3 cm and reaches this brain's
looming-escape size (~40°) only at contact, so the two flies are not expected to react to each
other; the fly-by is reported, not required (see docs/05 §5).
"""
import gc
import sys

import numpy as np
import polars as pl
import torch

from flyhigh.agent import FlyAgent
from flyhigh.data.connectome import Connectome
from flyhigh.flybody.arena import DrumParams
from flyhigh.flybody.sim import FlySimulation
from flyhigh.motor.command import MotorCommand
from flyhigh.motor.readout import ReadoutParams

results: dict[str, bool] = {}
REPS = 3


def check(name, passed):
    results[name] = bool(passed)
    print(f"    -> {'ok' if passed else 'FAIL'}: {name}")


class BrainPlusScript:
    """Fly 0 has the brain; fly 1 follows a script (for the fly-by)."""

    def __init__(self, brain):
        self.brain, self.readout = brain, brain.readout
        self.cmd1 = MotorCommand.idle(0.0)

    @property
    def dn_rates(self):
        return np.concatenate([self.brain.dn_rates, np.zeros_like(self.brain.dn_rates)])

    def tick(self, frames):
        return [self.brain.tick(frames[:1])[0], self.cmd1]


def segment(sim, seconds):
    n0 = len(sim.rows)
    sim.run(seconds)
    return pl.DataFrame(sim.rows[n0:])


def main():
    c = Connectome.load("data/raw")
    params = ReadoutParams(forward_bias=0.0)

    # 1. two brains, two flies, hovering 16 cm apart, facing each other, the hand out of the room
    brains = FlyAgent(c, n_agents=2, params=params)
    # the hand waits on fly 0's side of the drum, so its path to fly 0 never passes fly 1
    sim = FlySimulation(brains, n_agents=2, drum=DrumParams(hand_home=(-12.0, -10.0, 8.0)))
    clean, trials = 0, 0
    for rep in range(REPS):
        if rep:
            sim.reset()
        sim.hand.park()
        log = segment(sim, 1.2)
        for i in (0, 1):
            a = log.filter(pl.col("agent") == i)
            flat = (a["z"] - a["z"][0]).abs().max() < 3 and a["roll_deg"].abs().max() < 30 and a["escape"].sum() == 0
            clean += flat; trials += 1
            print(f"[hover x2]  run {rep} fly {i}: z {a['z'].min():.1f}..{a['z'].max():.1f} cm, |roll| max {a['roll_deg'].abs().max():.0f}°, "
                  f"escapes {a['escape'].sum()}, GF max {a['gf_hz'].max():.0f} Hz -> {'ok' if flat else 'x'}")
    tps = sim.speed(0.5)
    print(f"[speed]     two flies, two brains: {tps:.1f} ticks/s")
    check(f"two flies hover with their brains, no escape ({clean}/{trials} fly-trials clean)", clean == trials)
    check("speed >= 2 ticks/s", tps >= 2)

    # 2. the hand goes for fly 0; fly 1, 8.5 cm away, is not visited
    wins = 0
    for rep in range(REPS):
        sim.reset()
        sim.run(0.5)  # the hand comes back to its home corner on reset
        sim.hand.approach(lambda s=sim: s.pos_of(0))
        log = segment(sim, 1.5)
        a0, a1 = log.filter(pl.col("agent") == 0), log.filter(pl.col("agent") == 1)
        e0 = a0.filter(pl.col("escape"))
        fd = e0["hand_dist"][0] if e0.height else None
        contact = sim.drum.hand_radius + 0.3
        size = None if fd is None else round(2 * np.degrees(np.arcsin(min(1.0, sim.drum.hand_radius / fd))))
        rise = None if e0.height == 0 else a0.filter(pl.col("t_ms") <= e0["t_ms"][0] + 200)["z"].max() - a0["z"][0]
        win = e0.height > 0 and contact < fd < 15 and rise is not None and rise > 0.3 and a1["escape"].sum() == 0
        wins += win
        print(f"[hand]      run {rep}: fly 0 escapes {e0.height}, first at {fd} cm (hand {size}°), rise {rise} cm "
              f"| fly 1 escapes {a1['escape'].sum()} (hand stays {a1['hand_dist'].min():.1f} cm from it) -> {'ok' if win else 'x'}")
    check(f"hand → fly 0 escapes before contact, fly 1 never ({wins}/{REPS} runs)", wins * 3 >= REPS * 2)

    # 3. (informative) a fly-by: fly 1 crosses 1.5 cm in front of hovering fly 0 at 6 cm/s, then as a ghost
    sim.close()
    del sim, brains
    gc.collect(); torch.cuda.empty_cache()
    sim = FlySimulation(BrainPlusScript(FlyAgent(c, n_agents=1, params=params)), n_agents=2,
                        starts=[(0.0, 0.0, 7.0, 0.0), (1.5, -5.0, 7.0, 90.0)])
    for ghost in (False, True):
        if ghost:
            sim.hide_fly(1)
            sim.reset()
        sim.hand.park()
        sim.run(0.3)
        sim.agent.cmd1 = MotorCommand(0.2, 0.0, 0.0, False)
        log = segment(sim, 1.2)
        a, b = log.filter(pl.col("agent") == 0), log.filter(pl.col("agent") == 1)
        d = np.linalg.norm(a.select("x", "y", "z").to_numpy() - b.select("x", "y", "z").to_numpy(), axis=1)
        near = a.filter(pl.Series(d < 3.0))
        size = round(2 * np.degrees(np.arctan(0.3 / d.min())))
        print(f"[fly-by]    {'ghost' if ghost else 'visible'}: closest {d.min():.1f} cm (the other fly subtends ~{size}°); "
              f"escapes {a['escape'].sum()}, GF max {a['gf_hz'].max():.0f} Hz, |yaw| max while < 3 cm "
              f"{near['yaw'].abs().max() if near.height else None}")
    sim.close()

    failed = [k for k, v in results.items() if not v]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks pass" + (f"; failed: {failed}" if failed else ""))
    print("PASS" if not failed else "FAIL")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
