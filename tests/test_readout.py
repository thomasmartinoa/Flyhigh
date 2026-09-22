import numpy as np
import polars as pl
import pytest

from flyhigh.motor.command import MotorCommand
from flyhigh.motor.readout import (
    Readout,
    ReadoutParams,
    escape_channel,
    forward_channel,
    lift_channel,
    yaw_channel,
)
from tests.test_lif import make_connectome

P = ReadoutParams()


def test_escape_requires_giant_fiber_or_gf_plus_dnp04():
    assert escape_channel(gf_hz=60, dnp04_hz=0, p=P) is True
    assert escape_channel(gf_hz=30, dnp04_hz=60, p=P) is True
    assert escape_channel(gf_hz=30, dnp04_hz=10, p=P) is False
    assert escape_channel(gf_hz=0, dnp04_hz=200, p=P) is False


def test_yaw_follows_right_minus_left_and_is_clipped():
    assert yaw_channel(dna_l=0, dna_r=50, hs_l=0, hs_r=0, p=P) > 0
    assert yaw_channel(dna_l=50, dna_r=0, hs_l=0, hs_r=0, p=P) < 0
    assert yaw_channel(dna_l=0, dna_r=0, hs_l=0, hs_r=100, p=P) > 0
    assert -1.0 <= yaw_channel(dna_l=0, dna_r=1000, hs_l=0, hs_r=1000, p=P) <= 1.0
    assert yaw_channel(0, 0, 0, 0, P) == 0.0


def test_lift_and_forward_defaults():
    assert lift_channel(vs_hz=0, p=P) == 0.0
    assert forward_channel(dnp09_hz=0, escape=False, p=P) == pytest.approx(P.forward_bias)
    assert forward_channel(dnp09_hz=100, escape=True, p=P) == 0.0


def test_idle_command():
    assert MotorCommand.idle() == MotorCommand(forward=0.2, yaw=0.0, lift=0.0, escape=False)


def readout_connectome():
    c = make_connectome(8, [])
    c.neurons = c.neurons.with_columns(
        pl.Series("type", ["DNp01", "DNp04", "DNa02", "DNa02", "HSN", "HSN", "VS", "DNp09"]),
        pl.Series("side", ["L", "L", "L", "R", "L", "R", "L", "L"]),
    )
    return c


def test_readout_resolves_neurons_and_builds_commands():
    ro = Readout(readout_connectome())
    assert set(ro.watch.tolist()) == set(range(8))
    rates = np.zeros((2, 8))
    rates[0, 0] = 100.0            # agent 0: giant fiber busy
    rates[1, 3] = 80.0             # agent 1: right DNa02
    cmds = ro.commands(rates)
    assert cmds[0] == MotorCommand(forward=0.0, yaw=0.0, lift=0.0, escape=True)
    assert cmds[1].escape is False and cmds[1].yaw > 0
    assert cmds[1].forward == pytest.approx(P.forward_bias)


def test_readout_tolerates_missing_types():
    c = make_connectome(2, [])
    ro = Readout(c)  # no DN types at all
    assert ro.watch.size == 0
    assert ro.command(np.zeros(2)) == MotorCommand.idle(P.forward_bias)


def test_escape_refractory_suppresses_a_cascade_but_not_the_first_escape():
    ro = Readout(readout_connectome(), ReadoutParams(escape_refractory_ms=100.0))
    loud = np.zeros((1, 8)); loud[0, 0] = 100.0  # the giant fiber keeps firing
    assert ro.commands(loud, dt_ms=10.0)[0].escape is True
    for _ in range(10):  # 100 ms of refractory: no second escape, and the other channels still work
        cmd = ro.commands(loud, dt_ms=10.0)[0]
        assert cmd.escape is False and cmd.forward == pytest.approx(P.forward_bias)
    assert ro.commands(loud, dt_ms=10.0)[0].escape is True  # 100 ms later it may fire again


def test_without_a_refractory_the_escape_repeats_every_tick():
    ro = Readout(readout_connectome())
    loud = np.zeros((1, 8)); loud[0, 0] = 100.0
    assert all(ro.commands(loud)[0].escape for _ in range(5))
