# scripts/validate_fly.py
"""The M4 spec's five experiments: the brain steering flybody's fly on its own wings. Exit 0 iff all pass.

As in M3, the hand experiment is repeated REPS times and judged by majority (the giant fiber is
marginal and CUDA's index_add is not deterministic).
"""
import gc
import sys

import numpy as np
import polars as pl
import torch

from flyhigh.agent import FlyAgent
from flyhigh.data.connectome import Connectome
from flyhigh.flybody.sim import FlySimulation
from flyhigh.motor.command import MotorCommand
from flyhigh.motor.readout import ReadoutParams

results: dict[str, bool] = {}
REPS = 3


def check(name, passed):
    results[name] = bool(passed)
    print(f"    -> {'ok' if passed else 'FAIL'}: {name}")


class Scripted:
    def __init__(self):
        self.cmd = MotorCommand.idle(0.0)

    def tick(self, frames):
        return [self.cmd]


def fresh(agent, **kw):
    gc.collect(); torch.cuda.empty_cache()
    return FlySimulation(agent, **kw)


def segment(sim, seconds):
    n0 = len(sim.rows)
    sim.run(seconds)
    return pl.DataFrame(sim.rows[n0:])


def main():
    c = Connectome.load("data/raw")

    # 2. steering (no brain): the policy follows the command
    sim = fresh(Scripted(), settle_s=0.0)
    p = sim.task.p
    sim.run(0.3)
    z0 = sim.pos[2]; log = segment(sim, 0.5)
    hover_ok = abs(sim.pos[2] - z0) < 0.5 and log["roll_deg"].abs().max() < 30
    print(f"[steer]     hover (policy alone): dz {sim.pos[2] - z0:+.2f} cm, |roll| max {log['roll_deg'].abs().max():.0f}°")
    p0 = sim.pos.copy(); sim.agent.cmd = MotorCommand(0.5, 0, 0, False); sim.run(0.5)
    v_fwd = (sim.pos - p0)[0] / 0.5
    sim.agent.cmd = MotorCommand.idle(0.0); sim.run(0.3)
    y0 = sim.yaw_deg; sim.agent.cmd = MotorCommand(0, 0.25, 0, False); sim.run(0.5)
    w = ((sim.yaw_deg - y0 + 180) % 360 - 180) / 0.5
    sim.agent.cmd = MotorCommand.idle(0.0); sim.run(0.3)
    z0 = sim.pos[2]; sim.agent.cmd = MotorCommand(0, 0, 0.5, False); sim.run(0.5)
    v_up = (sim.pos[2] - z0) / 0.5
    sim.agent.cmd = MotorCommand.idle(0.0); sim.run(0.3)
    z0 = sim.pos[2]; sim.agent.cmd = MotorCommand(0, 0, 0, True); sim.run(0.05); sim.agent.cmd = MotorCommand.idle(0.0); sim.run(0.25)
    hop = sim.pos[2] - z0
    print(f"[steer]     forward 0.5 -> {v_fwd:.1f} cm/s (target {0.5 * p.v_max:.1f}, ramp included); yaw 0.25 -> {w:+.0f}°/s "
          f"(target {-0.25 * p.w_max_deg:+.0f}); lift 0.5 -> {v_up:.1f} cm/s (target {0.5 * p.vz_max:.1f}); escape hop {hop:+.2f} cm")
    check("steering: hover, forward, yaw, lift within 30 %, escape hops",
          hover_ok and abs(v_fwd - 0.5 * p.v_max) < 0.3 * 0.5 * p.v_max and abs(w + 0.25 * p.w_max_deg) < 0.3 * 0.25 * p.w_max_deg
          and abs(v_up - 0.5 * p.vz_max) < 0.3 * 0.5 * p.vz_max and hop > 1.0)
    sim.close()

    # 1. hover with the brain (forward bias off): height, attitude, no escape
    sim = fresh(FlyAgent(c, n_agents=1, params=ReadoutParams(forward_bias=0.0)))
    log = segment(sim, 1.5)
    print(f"[hover]     brain in the fly: z {log['z'].min():.1f}..{log['z'].max():.1f} cm, |roll| max {log['roll_deg'].abs().max():.0f}°, "
          f"|pitch| max {log['pitch_deg'].abs().max():.0f}°, escapes {log['escape'].sum()}, GF max {log['gf_hz'].max():.0f} Hz")
    check("hover: within 3 cm of start height, attitude < 30°, no escape",
          (log["z"] - log["z"][0]).abs().max() < 3 and log["roll_deg"].abs().max() < 30 and log["escape"].sum() == 0)

    # 3. the hand (a 4 cm fingertip at 50 cm/s from ahead-right, following the fly)
    wins = 0
    for rep in range(REPS):
        sim.close(); sim = fresh(FlyAgent(c, n_agents=1, params=ReadoutParams(forward_bias=0.0)))
        sim.run(0.5)
        sim.hand.approach(lambda s=sim: s.pos)
        log = segment(sim, 1.5)
        e = log.filter(pl.col("escape"))
        fd = e["hand_dist"][0] if e.height else None
        contact = sim.drum.hand_radius + 0.3
        size = None if fd is None else round(2 * np.degrees(np.arcsin(min(1.0, sim.drum.hand_radius / fd))))
        rise = None if e.height == 0 else log.filter(pl.col("t_ms") <= e["t_ms"][0] + 200)["z"].max() - log["z"][0]
        win = e.height > 0 and contact < fd < 15 and rise is not None and rise > 0.5
        wins += win
        print(f"[hand]      run {rep}: escapes {e.height}, first at {fd} cm (hand {size}°), rise within 200 ms {rise} cm, "
              f"closest {log['hand_dist'].min():.1f} cm (contact {contact:.1f}) -> {'ok' if win else 'x'}")
    check(f"hand → escape during the approach, before contact ({wins}/{REPS} runs)", wins * 3 >= REPS * 2)

    # 4. optomotor: the drum spins around the hovering fly; the brain's yaw command follows the drum
    sim.close(); sim = fresh(FlyAgent(c, n_agents=1, params=ReadoutParams(forward_bias=0.0)))
    sim.run(0.5)
    ok = True
    for rate in (+60.0, -60.0):
        sim.spin_drum = rate; log = segment(sim, 0.6); sim.spin_drum = 0.0
        y = log.tail(40)["yaw"].to_numpy()
        hs = (log.tail(40)["hs_l_hz"].mean(), log.tail(40)["hs_r_hz"].mean())
        print(f"[optomotor] drum {rate:+.0f}°/s: yaw command mean {y.mean():+.2f} (with the drum in {np.mean(np.sign(y) == np.sign(rate)):.0%} of ticks), "
              f"HS L/R {hs[0]:.0f}/{hs[1]:.0f} Hz, escapes {log['escape'].sum()}")
        ok &= np.sign(y.mean()) == np.sign(rate) and abs(y.mean()) > 0.1
        sim.run(0.6)
    check("optomotor: yaw command follows the drum, both ways", ok)

    # 5. speed
    tps = sim.speed(0.5)
    print(f"[speed]     {tps:.1f} ticks/s (brain + 50 policy steps + 200 physics steps + 5 renders per tick)")
    check("speed >= 3 ticks/s", tps >= 3)
    sim.close()

    failed = [k for k, v in results.items() if not v]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks pass" + (f"; failed: {failed}" if failed else ""))
    print("PASS" if not failed else "FAIL")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
