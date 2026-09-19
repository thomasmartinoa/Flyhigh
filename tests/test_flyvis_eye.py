import numpy as np
import polars as pl
import pytest
import torch

from flyhigh.senses.flyvis_compat import import_flyvis, preserve_torch_default_device

# pytest.importorskip() does a raw, unwrapped import as its own side effect: if flyvis
# isn't already in sys.modules, that import alone would permanently flip torch's default
# device to cuda for the rest of the pytest session (see flyvis_compat.py). Wrap it too.
with preserve_torch_default_device():
    pytest.importorskip("flyvis")
flyvis = import_flyvis()
pytestmark = pytest.mark.flyvis

from flyhigh.senses.flyvis_eye import FlyvisEye


@pytest.fixture(scope="module")
def eye():
    if not (flyvis.results_dir / "flow/0000/000").exists():
        pytest.skip("pretrained flyvis models not downloaded")
    e = FlyvisEye()
    e.reset(batch_size=2)
    return e


def test_metadata(eye):
    assert eye.n_neurons > 40_000
    assert {"T4a", "T4b", "T4c", "T4d", "T5a", "L1", "Mi1"} <= set(np.unique(eye.types))
    assert eye.u.shape == eye.v.shape == (eye.n_neurons,)
    assert eye.rest.shape == (eye.n_neurons,)


def test_grey_input_stays_at_rest(eye):
    grey = torch.full((2, 721), 0.5)
    for _ in range(20):
        act = eye.step(grey)
    assert act.shape == (2, eye.n_neurons)
    assert torch.allclose(act[0].cpu(), torch.as_tensor(eye.rest), atol=1e-3)


def test_moving_edge_drives_t4_more_than_grey(eye):
    eye.reset(batch_size=1)
    t4 = torch.as_tensor(np.isin(eye.types, ["T4a", "T4b", "T4c", "T4d"]))
    base = eye.step(torch.full((1, 721), 0.5))[0, t4].sum().item()
    # a bright half-field sweeping across the eye: columns with v < k light up one by one
    v = torch.as_tensor(eye_v_of_columns())
    peak = 0.0
    for k in range(-15, 16):
        lum = torch.where(v < k, 1.0, 0.0).float()[None]
        peak = max(peak, eye.step(lum)[0, t4].sum().item())
    assert peak > 2 * base + 1e-3


def eye_v_of_columns():
    from flyvis.utils.hex_utils import get_hex_coords

    return get_hex_coords(15)[1]


def test_edge_offsets_have_the_known_t4_geometry(eye):
    off = eye.edge_offsets()
    assert set(off.columns) == {"s", "t", "du", "dv", "n"}
    row = off.filter((pl.col("s") == "L1") & (pl.col("t") == "Mi1")).row(0, named=True)
    assert row["du"] == 0 and row["dv"] == 0  # L1 -> Mi1 is same-column
    mi9 = off.filter((pl.col("s") == "Mi9") & pl.col("t").str.starts_with("T4"))
    assert mi9.height == 4 and (mi9.select(pl.col("du") ** 2 + pl.col("dv") ** 2).min().item() > 0)
