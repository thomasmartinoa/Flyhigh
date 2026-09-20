# scripts/validate_box.py
"""The M3 spec's five experiments with real brains in the MuJoCo box. Exit 0 iff all pass.

The giant fiber is marginal in this brain (docs/03 §5) and CUDA's index_add is not
deterministic, so runs of the same experiment differ like trials of a real fly: the hand and
two-fly experiments are repeated REPS times and judged by majority, as an experiment would be.
"""
import gc
import sys

import numpy as np
import polars as pl
import torch

from flyhigh.agent import FlyAgent
from flyhigh.data.connectome import Connectome
from flyhigh.motor.readout import ReadoutParams
from flyhigh.sim import Simulation

results: dict[str, bool] = {}
REPS = 3


def check(name, passed):
    results[name] = bool(passed)
    print(f"    -> {'ok' if passed else 'FAIL'}: {name}")


def fresh(c, n_agents, params=None, **kw):
    gc.collect(); torch.cuda.empty_cache()
    return Simulation(FlyAgent(c, n_agents=n_agents, params=params), n_agents=n_agents, **kw)


def per_agent(log, i, t_from=0.0):
    return log.filter((pl.col("agent") == i) & (pl.col("t_ms") > t_from))


def separation(log):
    a = log.filter(pl.col("agent") == 0).select("x", "y", "z").to_numpy()
    b = log.filter(pl.col("agent") == 1).select("x", "y", "z").to_numpy()
    return np.linalg.norm(a - b, axis=1)


def main():
    c = Connectome.load("data/raw")
    room = None
    sim = None

    # 5. speed (first, before the laptop GPU warms up)
    for n in (1, 2):
        sim = fresh(c, n)
        tps = sim.speed(1.0)
        print(f"[speed]     {n} agent(s): {tps:.1f} ticks/s with rendering")
        check(f"speed {n} agent(s) >= 8 ticks/s", tps >= 8)
        sim.close()

    # 1. hover: 3 s with the hand idle and the readout's forward bias off (the bodies stay put;
    #    yaw, lift and escape are the brain's)
    sim = fresh(c, 2, params=ReadoutParams(forward_bias=0.0))
    room = sim.room
    log = sim.run(3.0)
    ok = True
    for i in (0, 1):
        a = per_agent(log, i)
        inside = (a["x"].abs() < room.size[0] / 2).all() and (a["y"].abs() < room.size[1] / 2).all()
        print(f"[hover]     agent {i}: z {a['z'].min():.2f}..{a['z'].max():.2f}  |roll| {a['roll_deg'].abs().max():.1f}° "
              f"|pitch| {a['pitch_deg'].abs().max():.1f}°  escapes {a['escape'].sum()}  inside room {inside}")
        ok &= (a["z"] - 1.0).abs().max() < 0.3 and a["roll_deg"].abs().max() < 5 and a["pitch_deg"].abs().max() < 5
        ok &= inside and a["escape"].sum() == 0
    check("hover: level, at height, no escape", ok)

    # 2. the hand approaches agent 0 from ahead-right at 1 m/s, following it; the flies hover
    #    (forward bias 0) so that self-motion flow and the other fly stay out of the experiment
    wins = 0
    for rep in range(REPS):
        sim.close(); sim = fresh(c, 2, params=ReadoutParams(forward_bias=0.0))
        sim.run(0.5)
        sim.hand.approach(lambda b=sim.bodies[0]: b.pos)  # follows the fly
        log = sim.run(3.0)
        a0, a1 = per_agent(log, 0, 500), per_agent(log, 1, 500)
        e0 = a0.filter(pl.col("escape"))
        first_dist = e0["hand_dist"][0] if e0.height else None
        jump = None
        if e0.height:
            jump = a0.filter(pl.col("t_ms") <= e0["t_ms"][0] + 200)["z"].max() - a0["z"][0]
        contact = sim.room.hand_radius + sim.room.body_size
        size = None if first_dist is None else round(2 * np.degrees(np.arcsin(min(1.0, sim.room.hand_radius / first_dist))))
        win = e0.height > 0 and contact < first_dist < 0.6 and jump is not None and jump >= 0.2 and a1["escape"].sum() == 0
        wins += win
        print(f"[hand]      run {rep}: agent 0 escapes {e0.height}, first at {first_dist} m (hand {size}°), rise {jump} m "
              f"| agent 1 escapes {a1['escape'].sum()} | closest {a0['hand_dist'].min():.2f} m, contact {contact:.2f} -> {'ok' if win else 'x'}")
    check(f"hand → agent 0 escapes during the approach, before contact; agent 1 never ({wins}/{REPS} runs)", wins * 3 >= REPS * 2)

    # 3. optomotor: an imposed 30°/s rotation for 500 ms, both directions; the brain steers back
    sim.close(); sim = fresh(c, 1)
    sim.run(0.5)
    ok = True
    for rate in (+30.0, -30.0):
        sim.disturb_yaw[0] = rate
        log = sim.run(0.5)
        sim.disturb_yaw.clear()
        y = per_agent(log, 0).tail(30)["yaw"].to_numpy()  # last 300 ms of the rotation
        frac = float(np.mean(np.sign(y) == -np.sign(rate)))
        print(f"[optomotor] imposed {rate:+.0f}°/s: brain yaw command mean {y.mean():+.2f} (opposing sign in {frac:.0%} of ticks)")
        ok &= np.sign(y.mean()) == -np.sign(rate) and abs(y.mean()) > 0.1
        sim.run(0.7)
    check("optomotor: the brain steers against an imposed rotation", ok)

    # 4. two flies head-on: at least one escapes before contact; none when they cannot see each other.
    #    2.5 s: they cross at ~1.5 s and are still 1.5 m from the end walls, whose corners loom later.
    wins = {False: 0, True: 0}
    for hidden in (False, True):
        for rep in range(REPS):
            sim.close(); sim = fresh(c, 2, settle_s=0.0)
            sim.hand.park()  # the only dark things in the room are the two flies
            if hidden:
                sim.hide_agent(0); sim.hide_agent(1)
            sim.settle(0.5)
            log = sim.run(2.5)
            d = separation(log)
            contact = 2 * room.body_size
            esc = [per_agent(log, i).filter(pl.col("escape")) for i in (0, 1)]
            first = min([e["t_ms"][0] for e in esc if e.height], default=None)
            sep_at_first = None if first is None else round(float(d[int(first / 10) - 1]), 2)
            win = (first is not None and sep_at_first > contact) if not hidden else first is None
            wins[hidden] += win
            print(f"[two flies] {'hidden ' if hidden else 'visible'} run {rep}: escapes {[e.height for e in esc]}, first at t={first} "
                  f"with separation {sep_at_first}, closest {d.min():.2f} m (contact {contact:.2f}) -> {'ok' if win else 'x'}")
    check(f"two flies: escape before contact when visible ({wins[False]}/{REPS}), none when ghosts ({wins[True]}/{REPS} clean)",
          wins[False] * 3 >= REPS * 2 and wins[True] * 3 >= REPS * 2)

    sim.close()

    failed = [k for k, v in results.items() if not v]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks pass" + (f"; failed: {failed}" if failed else ""))
    print("PASS" if not failed else "FAIL")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
