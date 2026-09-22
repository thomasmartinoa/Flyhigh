# scripts/play_box.py
"""Fly the box live: drag things around with the mouse and watch the brain answer.

    uv run python scripts/play_box.py                  # one body, the hand, an interactive window
    uv run python scripts/play_box.py --agents 2       # two bodies, so they can react to each other
    uv run python scripts/play_box.py --pillars        # add two static obstacles
    uv run python scripts/play_box.py --seconds 20     # close by itself after 20 s of brain time
    uv run python scripts/play_box.py --check          # no window: 20 ticks headless, for testing

In the window
    double-click a body to select it, then ctrl + right-drag  ->  move it
        the hand is a mocap body, so dragging *places* it; dragging a flying body pushes it
        (ctrl + left-drag rotates instead)
    h   send the hand at the selected body (or body 0), following it
    j   park the hand out of the room          k  bring it back to its corner
    space  pause / resume          r  re-home the hand and re-centre the bodies
    f   toggle the escape refractory (300 ms / off)
    esc / window close  quit

The brain needs ~70 ms of wall time per 10 ms tick, so the world runs at about 1/7 of real
time. That is the point: you can watch a reflex happen. To place something precisely, pause
first (space), drag it where you want it, then resume.
"""
import argparse

from flyhigh.agent import FlyAgent
from flyhigh.data.connectome import Connectome
from flyhigh.motor.readout import ReadoutParams
from flyhigh.play import Play
from flyhigh.sim import Simulation
from flyhigh.world.scene import RoomParams


def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive box: move things, watch the brain react.")
    ap.add_argument("--agents", type=int, default=1, choices=(1, 2))
    ap.add_argument("--pillars", action="store_true", help="add two static obstacles")
    ap.add_argument("--face-px", type=int, default=64, help="eye render size; smaller is faster")
    ap.add_argument("--seconds", type=float, default=None, help="stop after this much brain time")
    ap.add_argument("--no-refractory", action="store_true", help="let one escape cascade, as in the validations")
    ap.add_argument("--check", action="store_true", help="run 20 ticks and every key without a window")
    a = ap.parse_args()

    c = Connectome.load("data/raw")
    room = RoomParams(pillars=((0.6, 0.3), (2.4, -0.8))) if a.pillars else RoomParams()
    params = ReadoutParams(escape_refractory_ms=0.0 if a.no_refractory else 300.0)
    sim = Simulation(FlyAgent(c, n_agents=a.agents, params=params), n_agents=a.agents,
                     room=room, face_px=a.face_px)
    play = Play(sim)
    try:
        if a.check:
            for _ in range(20):
                sim.step()
            for keycode in Play.all_keys():
                play.key(keycode)
            print(f"[check] {sim.tick} ticks, {len(sim.rows)} rows, "
                  f"escapes {sum(r['escape'] for r in sim.rows)}, "
                  f"hand {sim.rows[-1]['hand_dist']:.2f} m, all {len(Play.all_keys())} keys ran")
        else:
            print(__doc__[__doc__.index("In the window"):])
            play.run(a.seconds)
    finally:
        sim.close()


if __name__ == "__main__":
    main()
