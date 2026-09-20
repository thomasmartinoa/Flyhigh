import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from flyhigh.body.brick import Body
from flyhigh.body.hand import Hand
from flyhigh.motor.command import MotorCommand
from flyhigh.world.scene import build_mjcf

DT_MS = 2.0


def make():
    m = mujoco.MjModel.from_xml_string(build_mjcf(1, start=[(0.0, 0.0, 1.0)]))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d, Body(m, d, "agent0")


def fly(m, d, body, cmd, seconds):
    for _ in range(int(seconds * 1000 / DT_MS)):
        body.apply(cmd, DT_MS)
        mujoco.mj_step(m, d)


def test_hovers_level_at_its_start_height_on_idle():
    m, d, body = make()
    fly(m, d, body, MotorCommand.idle(0.0), 2.0)
    assert abs(body.pos[2] - 1.0) < 0.02
    roll, pitch = body.roll_pitch_deg
    assert abs(roll) < 1 and abs(pitch) < 1


def test_forward_command_reaches_the_target_speed_and_keeps_height():
    m, d, body = make()
    fly(m, d, body, MotorCommand(forward=0.5, yaw=0.0, lift=0.0, escape=False), 2.0)
    v, _ = body.local_velocity()
    assert abs(v[0] - 0.5) < 0.05 and abs(v[1]) < 0.05
    assert abs(body.pos[2] - 1.0) < 0.05 and body.pos[0] > 0.5


def test_yaw_command_turns_right_at_the_target_rate():
    m, d, body = make()
    fly(m, d, body, MotorCommand(forward=0.0, yaw=0.5, lift=0.0, escape=False), 1.0)
    _, w = body.local_velocity()
    assert abs(np.degrees(w[2]) - (-90.0)) < 10  # + yaw = turn right = clockwise from above = -z
    assert -100 < body.yaw_deg < -60


def test_lift_climbs_and_then_holds_the_new_height():
    m, d, body = make()
    fly(m, d, body, MotorCommand(forward=0.0, yaw=0.0, lift=1.0, escape=False), 1.0)
    z1 = body.pos[2]
    assert 1.3 < z1 < 1.6
    fly(m, d, body, MotorCommand.idle(0.0), 1.0)
    assert abs(body.pos[2] - z1) < 0.1


def test_escape_is_a_100ms_jump_up_and_back_then_normal_flight_resumes():
    m, d, body = make()
    fly(m, d, body, MotorCommand.idle(0.0), 0.5)
    z0, x0 = body.pos[2], body.pos[0]
    fly(m, d, body, MotorCommand(forward=0.0, yaw=0.0, lift=0.0, escape=True), 0.05)
    assert body.escape_left_ms == pytest.approx(50.0)
    fly(m, d, body, MotorCommand.idle(0.0), 0.25)  # the override keeps going without the flag
    assert body.escape_left_ms == 0.0
    assert body.pos[2] - z0 > 0.1 and body.pos[0] - x0 < -0.03
    fly(m, d, body, MotorCommand.idle(0.0), 1.0)
    v, _ = body.local_velocity()
    assert np.linalg.norm(v) < 0.1


def test_hand_approaches_stops_short_and_retreats():
    m = mujoco.MjModel.from_xml_string(build_mjcf(1))
    d = mujoco.MjData(m)
    hand = Hand(m, d, speed=1.0, stop_dist=0.2)
    home = hand.pos.copy()
    hand.approach([0.0, 0.0, 1.0])
    closest = np.inf
    for _ in range(3000):
        hand.step(0.002)
        closest = min(closest, np.linalg.norm(hand.pos - [0, 0, 1]))
    assert abs(closest - 0.2) < 0.01
    assert hand.state == "idle" and np.allclose(hand.pos, home, atol=1e-6)
