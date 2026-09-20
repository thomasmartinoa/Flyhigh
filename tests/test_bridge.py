import numpy as np
import polars as pl
import torch

from flyhigh.senses.alignment import ColumnAlignment
from flyhigh.senses.bridge import FlyvisBridge


def tiny_alignment():
    """4 flyvis neurons: two T4a, one Mi1, one R1; LIF has 10 neurons.
    L eye → 0,1,2 ; R eye → 5,6,7. On the R eye, LIF 7 stands for both the Mi1 (#2) and the R1 (#3)."""
    table = pl.DataFrame({
        "fv_index": [0, 1, 2, 0, 1, 2, 3], "eye": ["L"] * 3 + ["R"] * 4,
        "lif_index": [0, 1, 2, 5, 6, 7, 7], "weight": [1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
    })
    return ColumnAlignment(table, {"T4a": 1.0, "Mi1": 1.0, "R1": 1.0}, ("x", "z"))


FV_TYPES = np.array(["T4a", "T4a", "Mi1", "R1"])
REST = np.array([0.1, 0.1, 0.5, 0.2])


def test_bridge_subtracts_rest_scales_per_type_and_clamps():
    b = FlyvisBridge(tiny_alignment(), FV_TYPES, REST, n_lif=10, gains={"T4a": 10.0}, default_gain=2.0)
    act = torch.tensor([[0.6, 0.1, 0.5, 0.2],   # agent0 L: T4a#0 up by 0.5, others at rest
                        [0.1, 0.1, 1.5, 1.2],   # agent0 R: Mi1 up by 1.0, R1 up by 1.0
                        [0.0, 0.0, 0.0, 0.0],   # agent1 L: below rest → clamp to 0
                        [0.1, 0.1, 0.5, 0.2]])  # agent1 R: rest
    ext = b.ext_i(act)
    assert ext.shape == (2, 10)
    assert ext[0, 0].item() == 5.0 and ext[0, 1].item() == 0.0
    assert ext[0, 7].item() == 2.0  # mean of the two flyvis cells' drive: 0.5*2 + 0.5*2
    assert ext[0, [2, 3, 4, 5, 6, 8, 9]].abs().sum().item() == 0.0
    assert ext[1].abs().sum().item() == 0.0


def test_bridge_driven_indices_cover_both_eyes_once():
    b = FlyvisBridge(tiny_alignment(), FV_TYPES, REST, n_lif=10, gains={})
    np.testing.assert_array_equal(np.sort(b.driven_indices), [0, 1, 2, 5, 6, 7])


def test_bridge_follows_the_activity_device():
    b = FlyvisBridge(tiny_alignment(), FV_TYPES, REST, n_lif=10, gains={})
    act = torch.full((2, 4), 0.7, dtype=torch.float64)
    ext = b.ext_i(act)
    assert ext.device == act.device and ext.dtype == torch.float32
