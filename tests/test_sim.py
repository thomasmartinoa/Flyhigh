import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from flyhigh.motor.command import MotorCommand
from flyhigh.sim import Simulation, VideoWriter


class IdleAgent:
    def __init__(self, n):
        self.n = n
        self.seen = []

    def tick(self, frames):
        self.seen.append([f.lum.mean() for f in frames])
        return [MotorCommand.idle(0.0)] * self.n


class DarkFlincher(IdleAgent):
    """Escapes when something very dark fills > 8 % of its frontal field (the hand is 0.05,
    the darkest checker ~0.17): a crude looming detector for loop tests."""

    def tick(self, frames):
        super().tick(frames)
        return [MotorCommand(0.0, 0.0, 0.0, escape=bool((f.lum[45:135, 90:270] < 0.1).mean() > 0.08))
                for f in frames]


def test_loop_ticks_physics_and_logs_every_agent():
    sim = Simulation(IdleAgent(2), n_agents=2, face_px=32)
    assert sim.physics_per_tick == 5
    log = sim.run(0.2)
    sim.close()
    assert sim.tick == 20 and log.height == 40 and set(log["agent"]) == {0, 1}
    assert (abs(log["z"] - 1.0) < 0.02).all()
    assert len(sim.agent.seen) == 20 and 0.3 < sim.agent.seen[-1][0] < 0.7


def test_hand_in_the_face_makes_a_flincher_escape_and_only_it():
    sim = Simulation(DarkFlincher(2), n_agents=2, face_px=32, start=[(-1.0, 0.0, 1.0), (1.0, 0.0, 1.0)])
    sim.hand.approach(sim.bodies[0].pos)
    log = sim.run(3.5)
    sim.close()
    a0, a1 = log.filter(log["agent"] == 0), log.filter(log["agent"] == 1)
    assert a0["escape"].any() and not a1["escape"].any()
    assert a0["z"].max() > 1.1


def test_hidden_agent_is_invisible_but_still_there():
    sim = Simulation(IdleAgent(2), n_agents=2, face_px=32, start=[(0.0, 0.0, 1.0), (0.6, 0.0, 1.0)])
    before = sim.frames()[0].lum
    sim.hide_agent(1)
    after = sim.frames()[0].lum
    changed = np.abs(after - before) > 0.1
    rows, cols = np.nonzero(changed)
    assert changed.any() and abs(cols.mean() - 180) < 10 and 70 < rows.min() < 95  # straight ahead (+ its shadow)
    assert not changed[:, :120].any() and not changed[:, 240:].any()  # and nowhere else
    assert sim.bodies[1].pos[0] == pytest.approx(0.6)
    sim.close()


def test_kick_yaw_spins_the_body_right():
    sim = Simulation(IdleAgent(1), n_agents=1, face_px=32)
    sim.kick_yaw(0, 90.0)
    _, w = sim.bodies[0].local_velocity()
    assert np.degrees(w[2]) == pytest.approx(-90.0)
    sim.step()
    assert -2 < sim.bodies[0].yaw_deg < -0.3  # the body's rate loop damps the kick within ~70 ms
    sim.close()


def test_video_writer_produces_a_file(tmp_path):
    sim = Simulation(IdleAgent(1), n_agents=1, face_px=32)
    vw = VideoWriter(sim, tmp_path / "v.mp4", every=2, width=320)
    sim.run(0.1, on_tick=vw)
    vw.close(); sim.close()
    assert (tmp_path / "v.mp4").stat().st_size > 1000
