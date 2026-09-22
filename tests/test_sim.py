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
    sim = Simulation(IdleAgent(2), n_agents=2, face_px=32, settle_s=0.3)
    assert sim.physics_per_tick == 5
    assert len(sim.agent.seen) == 30 and sim.tick == 0  # settling: brain ticks, physics does not
    log = sim.run(0.2)
    sim.close()
    assert sim.tick == 20 and log.height == 40 and set(log["agent"]) == {0, 1}
    assert (abs(log["z"] - 1.0) < 0.02).all()
    assert len(sim.agent.seen) == 50 and 0.3 < sim.agent.seen[-1][0] < 0.7


def test_hand_in_the_face_makes_a_flincher_escape_and_only_it():
    sim = Simulation(DarkFlincher(2), n_agents=2, face_px=32, start=[(-1.0, 0.0, 1.0), (1.0, 0.0, 1.0)])
    sim.hand.approach(sim.bodies[0].pos)
    log = sim.run(3.5)
    sim.close()
    a0, a1 = log.filter(log["agent"] == 0), log.filter(log["agent"] == 1)
    assert a0["escape"].any() and not a1["escape"].any()
    assert a0["z"].max() > 1.1


def test_hidden_agent_is_invisible_but_still_there():
    sim = Simulation(IdleAgent(2), n_agents=2, face_px=32, start=[(0.0, 0.0, 1.0), (1.2, 0.0, 1.0)])
    before = sim.frames()[0].lum
    sim.hide_agent(1)
    after = sim.frames()[0].lum
    changed = np.abs(after - before) > 0.1
    rows, cols = np.nonzero(changed)
    assert changed.any() and abs(cols.mean() - 180) < 10 and 70 < rows.min() < 95  # straight ahead (+ its shadow)
    assert not changed[:, :120].any() and not changed[:, 240:].any()  # and nowhere else
    assert sim.bodies[1].pos[0] == pytest.approx(1.2)
    assert sim.model.geom_contype[sim.model.body_geomadr[sim.bodies[1].id]] == 0  # a ghost
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


def test_pillars_and_balls_appear_in_the_room_and_in_the_eye():
    from flyhigh.world.scene import RoomParams
    room = RoomParams(pillars=((0.8, 0.0),), balls=((-0.8, 0.0, 1.0),))
    sim = Simulation(IdleAgent(1), n_agents=1, room=room, face_px=32, start=[(0.0, 0.0, 1.0, 0.0)])
    m = sim.model
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(m.ngeom)]
    assert "pillar0" in names and "ball0" in names
    lum = sim.frames()[0].lum
    ahead = lum[85:95, 175:185].mean()   # the pillar, dead ahead
    behind = lum[85:95, 0:10].mean()     # the ball, dead behind: outside both eyes' fields
    assert ahead < 0.3 and behind == pytest.approx(0.5)
    sim.close()


def test_annotated_video_writes_a_watchable_file(tmp_path):
    from flyhigh.viz import AnnotatedVideo
    sim = Simulation(IdleAgent(1), n_agents=1, face_px=32)
    video = AnnotatedVideo(sim, tmp_path / "v.mp4", every=1, width=480)
    sim.run(0.05, on_tick=video)
    video.close(); sim.close()
    assert (tmp_path / "v.mp4").stat().st_size > 5000
