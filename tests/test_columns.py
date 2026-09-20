import polars as pl

from flyhigh.senses.columns import infer_columns, partner_offsets


def lattice(types, side, start):
    """`types` on a 3x3 hex patch, one neuron per (type, column). Returns rows (index, type, side, h1, h2)."""
    rows, idx = [], start
    for t in types:
        for h1 in range(3):
            for h2 in range(3):
                rows.append((idx, t, side, float(h1), float(h2))); idx += 1
    return rows


def synthetic():
    """A (annotated) on both sides; B, C, P, S have no coordinates.
    Each B takes 10 synapses from the A of its own column and 3 from a neighbour column.
    C connects only to B (needs a second round). P has no side and 8 synapses from one A.
    S is a singleton (non-columnar) that touches every A. One B ("weak") has only 2 synapses."""
    rows = lattice(["A"], "L", 0) + lattice(["A"], "R", 9)
    a = {(r[2], int(r[3]), int(r[4])): r[0] for r in rows}
    edges, idx = [], 18
    b = {}
    for side in ("L", "R"):
        for h1 in range(3):
            for h2 in range(3):
                rows.append((idx, "B", side, None, None)); b[(side, h1, h2)] = idx
                edges.append((a[(side, h1, h2)], idx, 10))
                edges.append((a[(side, (h1 + 1) % 3, h2)], idx, 3))
                idx += 1
    c_idx = idx  # three C's, each presynaptic to one B; the first sits at L (2,1)
    for target in (b[("L", 2, 1)], b[("L", 0, 0)], b[("R", 1, 1)]):
        rows.append((idx, "C", "L" if target < b[("R", 0, 0)] else "R", None, None))
        edges.append((idx, target, 5)); idx += 1
    p_idx = idx  # two side-less P's; the first touches an R-side A at (0,2)
    for target in (a[("R", 0, 2)], a[("L", 1, 0)]):
        rows.append((idx, "P", None, None, None)); edges.append((idx, target, 8)); idx += 1
    s_idx = idx; rows.append((idx, "S", "L", None, None)); idx += 1
    edges += [(a_i, s_idx, 4) for (side, _, _), a_i in a.items() if side == "L"]
    weak = idx; rows.append((idx, "B", "L", None, None)); idx += 1
    edges.append((a[("L", 1, 1)], weak, 2))
    neurons = pl.DataFrame(rows, schema={"index": pl.Int64, "type": pl.Utf8, "side": pl.Utf8,
                                         "hex1": pl.Float64, "hex2": pl.Float64}, orient="row")
    edges = pl.DataFrame(edges, schema={"pre_idx": pl.Int64, "post_idx": pl.Int64, "syn_count": pl.Int64},
                         orient="row")
    return neurons, edges, b, c_idx, p_idx, s_idx, weak


def test_infer_columns_takes_the_modal_partner_column_and_propagates():
    neurons, edges, b, c_idx, p_idx, s_idx, weak = synthetic()
    out = infer_columns(neurons, edges, ["A", "B", "C", "P", "S"], min_syn=3, min_neurons=2)
    got = {r["index"]: r for r in out.iter_rows(named=True)}
    for (side, h1, h2), i in b.items():
        assert (got[i]["hex1"], got[i]["hex2"], got[i]["hex_source"]) == (h1, h2, "inferred")
        assert abs(got[i]["hex_share"] - 10 / 13) < 1e-9
    assert (got[c_idx]["hex1"], got[c_idx]["hex2"]) == (2, 1)  # via B, second round
    assert (got[p_idx]["side"], got[p_idx]["hex1"], got[p_idx]["hex2"]) == ("R", 0, 2)  # side inferred
    assert got[s_idx]["hex1"] is None and got[weak]["hex1"] is None
    # annotated rows are untouched
    assert got[0]["hex_source"] == "annotated" and got[0]["hex_share"] is None
    assert out.height == neurons.height and out["index"].to_list() == neurons["index"].to_list()


def test_infer_columns_only_touches_requested_types():
    neurons, edges, *_ = synthetic()
    out = infer_columns(neurons, edges, ["A", "C"], min_syn=3, min_neurons=1)
    assert out.filter(pl.col("type") == "B")["hex1"].null_count() == out.filter(pl.col("type") == "B").height
    assert out.filter(pl.col("type") == "C")["hex1"].null_count() == 3  # C's only partners (B) stay unknown


def test_partner_offsets_are_synapse_weighted_pre_minus_post_in_axial_convention():
    # A at (0,0) -> B at (1,2): mcns offset (-1,-2); axial convention flips hex2 -> (-1, +2)
    neurons = pl.DataFrame({"index": [0, 1, 2], "type": ["A", "B", "A"], "side": ["L", "L", "L"],
                            "hex1": [0.0, 1.0, 2.0], "hex2": [0.0, 2.0, 2.0]})
    edges = pl.DataFrame({"pre_idx": [0, 2], "post_idx": [1, 1], "syn_count": [3, 1]})
    off = partner_offsets(neurons, edges, [("A", "B"), ("A", "Z")])
    assert off.height == 1
    r = off.row(0, named=True)
    assert (r["s"], r["t"], r["n"]) == ("A", "B", 4)
    assert abs(r["du"] - (3 * -1 + 1 * 1) / 4) < 1e-9 and abs(r["dv"] - (3 * 2 + 1 * 0) / 4) < 1e-9
