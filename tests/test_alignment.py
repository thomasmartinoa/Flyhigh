import numpy as np
import polars as pl

from flyhigh.senses.alignment import HEX_SYMMETRIES, ColumnAlignment, apply_symmetry
from flyhigh.senses.type_map import mcns_type


def test_type_map_merges_r1_to_r6_and_keeps_exact_names():
    assert mcns_type("R1") == "R1-R6" and mcns_type("R6") == "R1-R6"
    assert mcns_type("T4a") == "T4a" and mcns_type("Mi1") == "Mi1"
    assert mcns_type("CT1(Lo1)") == "CT1"


def test_twelve_symmetries_are_distinct_and_invertible():
    u, v = np.array([1, 2, 0]), np.array([0, 1, 3])
    images = {tuple(np.concatenate(apply_symmetry(u, v, s))) for s in HEX_SYMMETRIES}
    assert len(images) == 12


def fake_lattice(rotate: int):
    """A male-CNS-like neuron table: two types on a 7x7 hex patch, rotated by a known symmetry
    and shifted, on both sides. Returns (neurons, fv_types, fv_u, fv_v)."""
    uu, vv = np.meshgrid(np.arange(-3, 4), np.arange(-3, 4)); uu, vv = uu.ravel(), vv.ravel()
    fv_u = np.concatenate([uu, uu]); fv_v = np.concatenate([vv, vv])
    fv_types = np.array(["T4a"] * len(uu) + ["Mi1"] * len(uu))
    ru, rv = apply_symmetry(uu, vv, HEX_SYMMETRIES[rotate])
    rows = []
    idx = 0
    for side in ("L", "R"):
        for t in ("T4a", "Mi1"):
            for a, b in zip(ru + 10, rv + 20):  # shift = unknown lattice centre
                rows.append((idx, t, side, float(a), float(b))); idx += 1
    neurons = pl.DataFrame(rows, schema=["index", "type", "side", "hex1", "hex2"], orient="row")
    return neurons, fv_types, fv_u, fv_v


def test_alignment_recovers_rotation_and_matches_every_column():
    neurons, fv_types, fv_u, fv_v = fake_lattice(rotate=4)
    al = ColumnAlignment.build(fv_types, fv_u, fv_v, neurons)
    assert al.coverage["T4a"] == 1.0 and al.coverage["Mi1"] == 1.0
    assert al.table.height == 2 * len(fv_types)  # both eyes
    # a flyvis T4a at (u,v) maps to the LIF T4a at the rotated+shifted (hex1,hex2) on the right side
    r = al.table.filter(pl.col("eye") == "R").join(neurons, left_on="lif_index", right_on="index")
    assert set(r["side"]) == {"R"} and set(r["type"]) == {"T4a", "Mi1"}


def test_alignment_round_trips_through_parquet(tmp_path):
    neurons, fv_types, fv_u, fv_v = fake_lattice(rotate=1)
    al = ColumnAlignment.build(fv_types, fv_u, fv_v, neurons)
    al.save(tmp_path / "al.parquet")
    al2 = ColumnAlignment.load(tmp_path / "al.parquet")
    assert al2.table.equals(al.table)
    np.testing.assert_array_equal(al2.lif_indices("L"), al.lif_indices("L"))
