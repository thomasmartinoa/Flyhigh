from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("flybody")
pytest.importorskip("tensorflow")
if not Path("data/flybody/trained-fly-policies/flight").exists():
    pytest.skip("flybody's trained flight policy not downloaded", allow_module_level=True)
pytestmark = pytest.mark.flybody

from flyhigh.flybody.policy import FlightPolicy
from flyhigh.flybody.sim import FlySimulation
from flyhigh.motor.command import MotorCommand
from flyhigh.world.eyes import bearing_to_direction


class Scripted:
    def __init__(self):
        self.cmd = MotorCommand.idle(0.0)

    def tick(self, frames):
        return [self.cmd]


@pytest.fixture(scope="module")
def sim():
    s = FlySimulation(Scripted(), settle_s=0.0)
    yield s
    s.close()


def test_policy_loads_and_acts_on_the_task_observation(sim):
    pol = FlightPolicy()
    a = pol(sim.ts.observation)
    assert a.shape == (12,) and np.isfinite(a).all()
    assert set(pol.keys) == set(sim.ts.observation)


def test_eye_sees_the_hand_where_it_is(sim):
    import mujoco

    m, d = sim.physics.model.ptr, sim.physics.data.ptr
    for az, el in ((0, 0), (60, 0), (-60, 10)):
        d.mocap_pos[sim.hand.mocap_id] = sim.pos + 6.0 * bearing_to_direction(az, el)
        mujoco.mj_forward(m, d)
        f = sim.frames()[0]
        rows, cols = np.nonzero(f.lum < 0.15)
        assert abs(f.az_deg[cols].mean() - az) < 4 and abs(f.el_deg[rows].mean() - el) < 4
    d.mocap_pos[sim.hand.mocap_id] = sim.hand.home
    mujoco.mj_forward(m, d)


def test_the_fly_hovers_flies_forward_turns_climbs_and_hops(sim):
    z0 = sim.pos[2]
    log = sim.run(0.4)
    assert abs(sim.pos[2] - z0) < 0.3 and log["roll_deg"].abs().max() < 30
    p0 = sim.pos.copy()
    sim.agent.cmd = MotorCommand(0.5, 0.0, 0.0, False)
    sim.run(0.4)
    assert 4.0 < (sim.pos - p0)[0] < 7.0  # 15 cm/s, minus the ramp
    sim.agent.cmd = MotorCommand.idle(0.0); sim.run(0.2)
    y0 = sim.yaw_deg
    sim.agent.cmd = MotorCommand(0.0, 0.25, 0.0, False)
    sim.run(0.5)
    turned = (sim.yaw_deg - y0 + 180) % 360 - 180
    assert -30 < turned < -8  # 45°/s minus the ramp; + yaw = right = clockwise = decreasing heading
    sim.agent.cmd = MotorCommand.idle(0.0); sim.run(0.2)
    z0 = sim.pos[2]
    sim.agent.cmd = MotorCommand(0.0, 0.0, 0.5, False)
    sim.run(0.3)
    assert 1.2 < sim.pos[2] - z0 < 2.8  # 7.5 cm/s
    sim.agent.cmd = MotorCommand.idle(0.0); sim.run(0.2)
    z0 = sim.pos[2]
    sim.agent.cmd = MotorCommand(0.0, 0.0, 0.0, True)
    sim.run(0.05)
    sim.agent.cmd = MotorCommand.idle(0.0)
    sim.run(0.25)
    assert sim.pos[2] - z0 > 1.0  # the hop
