# scripts/validate_reflexes.py
"""The M2 spec's five checks, run on the model like experiments on a fly. Exit 0 iff all pass.

The checks are the spec's and are not tuned to the model; the gain and readout are (Task 9).
"""
import gc
import sys
import time

import numpy as np
import torch

from flyhigh.agent import FlyAgent
from flyhigh.data.connectome import Connectome
from flyhigh.senses.frame import PanoramicFrame, looming_disc, rotating_grating

results: dict[str, bool] = {}


def check(name, passed):
    results[name] = bool(passed)
    print(f"    -> {'ok' if passed else 'FAIL'}: {name}")


def fresh(c, n_agents):
    """A new agent with the previous one's GPU memory released first (two do not fit in 8 GB)."""
    gc.collect(); torch.cuda.empty_cache()
    return FlyAgent(c, n_agents=n_agents)


def run(agent, frames_per_agent):
    """frames_per_agent: list over agents of frame lists (same length). Returns command lists per tick."""
    return [agent.tick([fa[i] for fa in frames_per_agent]) for i in range(len(frames_per_agent[0]))]


def main():
    c = Connectome.load("data/raw")
    gf = c.ids_by_type(r"^DNp01$")
    lc = c.ids_by_type(r"^(LC4|LPLC2)$")

    # 1. silence: 1 s of grey
    agent = None; agent = fresh(c, 1)
    dn_spikes = 0
    for cm in run(agent, [[PanoramicFrame.grey()] * 100]):
        dn_spikes += agent.brain_counts_last_tick[agent.readout.watch].sum()
    last = cm[0]
    print(f"[silence]   escape={last.escape} yaw={last.yaw:+.3f} fwd={last.forward:.3f} lift={last.lift:+.3f}  "
          f"DN spikes in 1 s={int(dn_spikes)}")
    check("silence", dn_spikes == 0 and not last.escape and abs(last.yaw) < 0.05
          and abs(last.forward - 0.2) < 0.05 and abs(last.lift) < 0.05)

    # 2. escape: approaching disc on the right must trigger before 40°; a receding one must not
    agent = None; agent = fresh(c, 1)
    frames = [PanoramicFrame.grey()] * 20 + looming_disc(az=60, el=0, start_deg=5, end_deg=60, duration_ms=500)
    lc_hz = np.zeros(c.n_neurons)
    cmds = []
    for cm in run(agent, [frames]):
        cmds.append(cm); lc_hz += agent.brain_counts_last_tick
    lc_hz = lc_hz[lc] / (len(frames) * 0.01)
    first = next((i for i, cm in enumerate(cmds) if cm[0].escape), None)
    size_at = None if first is None else 5 + (60 - 5) * (first - 20) / 49
    print(f"[escape]    first escape tick={first}  disc diameter then={size_at}  GF rate at end="
          f"{agent.dn_rates[0, gf].mean():.0f} Hz  LC4/LPLC2 mean {lc_hz.mean():.1f} Hz, max {lc_hz.max():.0f} Hz")
    check("escape before 40°", first is not None and size_at < 40)
    agent = None; agent = fresh(c, 1)
    receding = [PanoramicFrame.grey()] * 20 + looming_disc(az=60, el=0, start_deg=60, end_deg=5, duration_ms=500)
    cmds = run(agent, [receding])
    any_escape = any(cm[0].escape for cm in cmds[25:])  # ignore the onset flash
    print(f"[receding]  escape after onset={any_escape}  (escape ticks: {[i - 20 for i, cm in enumerate(cmds) if cm[0].escape]})")
    check("no escape for receding disc", not any_escape)
    # (informative) the same disc receding after it has been visible for 300 ms: separates the
    # appearance startle from a response to the receding motion itself
    agent = None; agent = fresh(c, 1)
    static = [looming_disc(az=60, el=0, start_deg=60, end_deg=60, duration_ms=10)[0]] * 30
    cmds = run(agent, [static + looming_disc(az=60, el=0, start_deg=60, end_deg=5, duration_ms=500)])
    print(f"[receding]  after 300 ms visible: escapes during the receding motion = {sum(cm[0].escape for cm in cmds[30:])}"
          f" (startle ticks while static: {[i for i, cm in enumerate(cmds[:30]) if cm[0].escape]})")

    # 3. optomotor, both directions: the fly turns with the scene
    yaws = {}
    for direction, name in ((+1, "cw"), (-1, "ccw")):
        agent = None; agent = fresh(c, 1)
        frames = [PanoramicFrame.grey()] * 20 + rotating_grating(30, 60, 800, direction=direction)
        cmds = run(agent, [frames])
        y = np.array([cm[0].yaw for cm in cmds[40:]])  # after 200 ms of motion
        yaws[name] = y
        print(f"[optomotor] {name}: mean yaw {y.mean():+.3f}, fraction with the right sign = "
              f"{np.mean(np.sign(y) == direction):.2f}, escapes = {sum(cm[0].escape for cm in cmds)}")
    check("optomotor cw > +0.3", yaws["cw"].mean() > 0.3)
    check("optomotor ccw < -0.3", yaws["ccw"].mean() < -0.3)
    cw, ccw = abs(yaws["cw"].mean()), abs(yaws["ccw"].mean())
    check("optomotor mirror within 20 %", abs(cw - ccw) < 0.2 * max(cw, ccw))

    # 4. two agents independent: only the one that sees the loom escapes
    agent = None; agent = fresh(c, 2)
    a0 = [PanoramicFrame.grey()] * 20 + looming_disc(az=60, el=0, start_deg=5, end_deg=60, duration_ms=500)
    a1 = [PanoramicFrame.grey()] * len(a0)
    cmds = run(agent, [a0, a1])
    e0 = any(cm[0].escape for cm in cmds); e1 = any(cm[1].escape for cm in cmds)
    print(f"[two agents] agent0 escape={e0} agent1 escape={e1}")
    check("two agents independent", e0 and not e1)
    # (informative, not in the spec) the same independence on the reflex that does work
    g0 = [PanoramicFrame.grey()] * 20 + rotating_grating(30, 60, 400, direction=+1)
    cmds = run(agent, [g0, [PanoramicFrame.grey()] * len(g0)])
    y0 = np.mean([cm[0].yaw for cm in cmds[40:]]); y1 = np.mean([cm[1].yaw for cm in cmds[40:]])
    print(f"[two agents] optomotor: agent0 (grating) yaw {y0:+.3f}, agent1 (grey) yaw {y1:+.3f}")

    # 5. speed
    for n in (1, 2):
        agent = None; agent = fresh(c, n)
        run(agent, [[PanoramicFrame.grey()] * 5] * n)
        t = time.perf_counter(); run(agent, [[PanoramicFrame.grey()] * 50] * n)
        tps = 50 / (time.perf_counter() - t)
        print(f"[speed]     {n} agent(s): {tps:.1f} ticks/s")
        check(f"speed {n} agent(s) >= 10 ticks/s", tps >= 10)

    failed = [k for k, v in results.items() if not v]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks pass" + (f"; failed: {failed}" if failed else ""))
    print("PASS" if not failed else "FAIL")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
