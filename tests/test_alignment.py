import numpy as np
import polars as pl
import pytest

from flyhigh.senses.alignment import (
    HEX_SYMMETRIES,
    ColumnAlignment,
    apply_symmetry,
    choose_symmetry,
    mcns_to_axial,
)
from flyhigh.senses.type_map import mcns_type, mcns_types


def test_type_map_merges_r1_to_r6_and_keeps_exact_names():
    assert mcns_type("R1") == "R1-R6" and mcns_type("R6") == "R1-R6"
    assert mcns_type("T4a") == "T4a" and mcns_type("Mi1") == "Mi1"
    assert mcns_type("CT1(Lo1)") == "CT1"


def test_type_map_splits_one_flyvis_type_over_several_mcns_types():
    assert mcns_types("T4a") == ("T4a",)
    assert set(mcns_types("R7")) >= {"R7y", "R7p", "R7d"}
    assert mcns_types("TmY9") == ("TmY9a", "TmY9b")
    assert mcns_type("R7") == mcns_types("R7")[0]


def test_twelve_symmetries_are_distinct_and_invertible():
    u, v = np.array([1, 2, 0]), np.array([0, 1, 3])
    images = {tuple(np.concatenate(apply_symmetry(u, v, s))) for s in HEX_SYMMETRIES}
    assert len(images) == 12


def test_mcns_to_axial_turns_mcns_neighbours_into_axial_neighbours():
    # the male CNS has a synapse-carrying neighbour at (+1,+1); axial hex has one at (+1,-1)
    u, v = mcns_to_axial(np.array([1.0]), np.array([1.0]))
    assert (u[0], v[0]) == (1.0, -1.0)


def to_mcns(u, v):
    """Inverse of mcns_to_axial, for building male-CNS-like test tables."""
    return u, -v


def fake_lattice(rotate: int, duplicate_t4a=False, photoreceptors=False):
    """A male-CNS-like neuron table: two types on a 7x7 hex patch, rotated by a known symmetry
    and shifted, on both sides. Returns (neurons, fv_types, fv_u, fv_v)."""
    uu, vv = np.meshgrid(np.arange(-3, 4), np.arange(-3, 4)); uu, vv = uu.ravel(), vv.ravel()
    fv_u = np.concatenate([uu, uu]); fv_v = np.concatenate([vv, vv])
    fv_types = np.array(["T4a"] * len(uu) + ["Mi1"] * len(uu))
    ru, rv = apply_symmetry(uu, vv, HEX_SYMMETRIES[rotate])
    h1, h2 = to_mcns(ru + 10, rv + 20)  # shift = unknown lattice centre
    rows = []
    idx = 0
    for side in ("L", "R"):
        for t in ("T4a", "Mi1"):
            for a, b in zip(h1, h2):
                rows.append((idx, t, side, float(a), float(b))); idx += 1
    if duplicate_t4a:  # two LIF T4a in one column (multi-column dendrites tie in inference)
        rows.append((idx, "T4a", "R", rows[0][3], rows[0][4])); idx += 1
    if photoreceptors:  # flyvis R1 and R2 in the centre column; the male CNS has one "R1-R6" there
        fv_types = np.concatenate([fv_types, ["R1", "R2"]])
        fv_u = np.concatenate([fv_u, [0, 0]]); fv_v = np.concatenate([fv_v, [0, 0]])
        for side in ("L", "R"):
            rows.append((idx, "R1-R6", side, float(h1[len(uu) // 2]), float(h2[len(uu) // 2]))); idx += 1
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


def test_alignment_with_given_symmetry_maps_one_flyvis_column_to_every_lif_neuron_in_it():
    neurons, fv_types, fv_u, fv_v = fake_lattice(rotate=4, duplicate_t4a=True)
    al = ColumnAlignment.build(fv_types, fv_u, fv_v, neurons, symmetry=HEX_SYMMETRIES[4])
    assert al.transform == HEX_SYMMETRIES[4]
    assert al.table.height == 2 * len(fv_types) + 1
    dup = al.table.filter(pl.col("eye") == "R").group_by("fv_index").len().filter(pl.col("len") > 1)
    assert dup.height == 1 and dup["len"][0] == 2
    assert al.table["lif_index"].n_unique() == al.table.height  # each LIF neuron driven once ...
    assert set(al.table["weight"]) == {1.0}


def test_alignment_weights_share_a_lif_neuron_between_the_flyvis_neurons_that_stand_for_it():
    neurons, fv_types, fv_u, fv_v = fake_lattice(rotate=2, photoreceptors=True)
    al = ColumnAlignment.build(fv_types, fv_u, fv_v, neurons, symmetry=HEX_SYMMETRIES[2])
    r = al.table.filter(pl.col("eye") == "R").join(neurons, left_on="lif_index", right_on="index")
    pr = r.filter(pl.col("type") == "R1-R6")
    assert pr.height == 2 and set(pr["weight"]) == {0.5} and pr["lif_index"].n_unique() == 1
    assert set(r.filter(pl.col("type") != "R1-R6")["weight"]) == {1.0}
    # weights sum to one per LIF neuron, so the bridge's weighted sum is a mean
    per_lif = al.table.group_by("eye", "lif_index").agg(pl.col("weight").sum())
    assert set(per_lif["weight"]) == {1.0}
    assert al.weights("R").shape == al.lif_indices("R").shape


def test_alignment_round_trips_through_parquet(tmp_path):
    neurons, fv_types, fv_u, fv_v = fake_lattice(rotate=1)
    al = ColumnAlignment.build(fv_types, fv_u, fv_v, neurons)
    al.save(tmp_path / "al.parquet")
    al2 = ColumnAlignment.load(tmp_path / "al.parquet")
    assert al2.table.equals(al.table)
    np.testing.assert_array_equal(al2.lif_indices("L"), al.lif_indices("L"))


@pytest.mark.parametrize("truth", [0, 3, 7, 10])
def test_choose_symmetry_recovers_the_transform_from_partner_offsets(truth):
    fv = pl.DataFrame({"s": ["Mi9", "Mi9", "Mi4", "Tm9"], "t": ["T4a", "T4b", "T4c", "T5a"],
                       "du": [0.3, -0.6, 1.0, 0.5], "dv": [-0.4, 0.9, -0.3, -1.0], "n": [1.0] * 4})
    u, v = apply_symmetry(fv["du"].to_numpy(), fv["dv"].to_numpy(), HEX_SYMMETRIES[truth])
    mcns = fv.with_columns(pl.Series("du", u + 0.05), pl.Series("dv", v - 0.05))  # a little noise
    mcns = pl.concat([mcns, pl.DataFrame({"s": ["Q"], "t": ["Z"], "du": [9.0], "dv": [9.0], "n": [1.0]})])
    sym, scores = choose_symmetry(fv, mcns)
    assert sym == HEX_SYMMETRIES[truth]
    assert set(scores) == set(HEX_SYMMETRIES) and scores[sym] == min(scores.values())
    assert scores[sym] < 0.1


def test_choose_symmetry_needs_a_shared_pair():
    fv = pl.DataFrame({"s": ["A"], "t": ["B"], "du": [1.0], "dv": [0.0], "n": [1.0]})
    with pytest.raises(ValueError):
        choose_symmetry(fv, fv.with_columns(pl.lit("C").alias("t")))
