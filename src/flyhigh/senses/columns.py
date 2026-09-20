"""Give eye-column coordinates to optic-lobe neurons that the male-CNS release left without.

Only 15 columnar cell types carry `hex1`/`hex2` in male-CNS v1.0 (L1–L3, L5, Mi1, Mi4, Mi9, T1,
Tm1, Tm2, Tm4, Tm9, Tm20, C2, C3); T4/T5 and everything else in flyvis's 60-odd types do not.
But a columnar neuron takes most of its synapses from partners in its own column (a T4a gets
~99 % of its Mi1 input from one Mi1), so the column that receives the most synapse votes from
coordinate-bearing partners is its column. Repeating this a few rounds reaches types whose
partners were themselves unlabelled (T4 → LPLC2-side TmY types → …).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from flyhigh.senses.alignment import mcns_to_axial


def infer_columns(
    neurons: pl.DataFrame,
    edges: pl.DataFrame,
    types,
    *,
    min_syn: int = 10,
    min_neurons: int = 30,
    max_rounds: int = 4,
) -> pl.DataFrame:
    """Return `neurons` (same rows, same order) with `hex1`/`hex2` filled in for the given
    `types` where possible, plus `hex_source` ("annotated" | "inferred") and `hex_share`
    (fraction of a neuron's synapse votes that went to the chosen column; null if annotated).

    A neuron votes with every synapse (in or out) to a partner that already has a column, on
    the same side; a neuron with no `side` takes the winning partner's side too (photoreceptors
    are side-less in v1.0). Types with fewer than `min_neurons` members are non-columnar
    (CT1, Am1: one giant cell per side) and are left alone, as are neurons with fewer than
    `min_syn` votes.
    """
    types = list(types)
    counts = neurons.filter(pl.col("type").is_in(types)).group_by("type").len()
    columnar = counts.filter(pl.col("len") >= min_neurons)["type"].to_list()
    out = neurons.with_columns(
        pl.when(pl.col("hex1").is_not_null()).then(pl.lit("annotated")).alias("hex_source"),
        pl.lit(None, dtype=pl.Float64).alias("hex_share"),
    )
    e = edges.select("pre_idx", "post_idx", "syn_count")
    votes_both_ways = pl.concat([
        e.select(pl.col("pre_idx").alias("a"), pl.col("post_idx").alias("b"), "syn_count"),
        e.select(pl.col("post_idx").alias("a"), pl.col("pre_idx").alias("b"), "syn_count"),
    ])
    for _ in range(max_rounds):
        known = out.filter(pl.col("hex1").is_not_null()).select(
            pl.col("index").alias("b"), pl.col("side").alias("b_side"),
            pl.col("hex1").alias("b_hex1"), pl.col("hex2").alias("b_hex2"),
        )
        unknown = out.filter(pl.col("hex1").is_null() & pl.col("type").is_in(columnar)).select(
            pl.col("index").alias("a"), pl.col("side").alias("a_side"),
        )
        if unknown.height == 0:
            break
        votes = (
            votes_both_ways.join(unknown, on="a").join(known, on="b")
            .filter(pl.col("a_side").is_null() | (pl.col("a_side") == pl.col("b_side")))
            .group_by("a", "b_side", "b_hex1", "b_hex2").agg(pl.col("syn_count").sum().alias("syn"))
        )
        total = votes.group_by("a").agg(pl.col("syn").sum().alias("total"))
        winner = (
            # ties broken by column, so the result is reproducible run to run
            votes.sort(["syn", "b_hex1", "b_hex2"], descending=[True, False, False])
            .group_by("a", maintain_order=True).first()
            .join(total, on="a")
            .with_columns((pl.col("syn") / pl.col("total")).alias("share"))
            .filter(pl.col("total") >= min_syn)
            .select(pl.col("a").alias("index"), "b_side", "b_hex1", "b_hex2", "share")
        )
        if winner.height == 0:
            break
        out = (
            out.join(winner, on="index", how="left")
            .with_columns(
                pl.coalesce("hex1", "b_hex1").alias("hex1"),
                pl.coalesce("hex2", "b_hex2").alias("hex2"),
                pl.coalesce("side", "b_side").alias("side"),
                pl.coalesce("hex_share", "share").alias("hex_share"),
                pl.when(pl.col("b_hex1").is_not_null()).then(pl.lit("inferred"))
                .otherwise(pl.col("hex_source")).alias("hex_source"),
            )
            .drop("b_side", "b_hex1", "b_hex2", "share")
        )
    return out


def partner_offsets(neurons: pl.DataFrame, edges: pl.DataFrame, pairs) -> pl.DataFrame:
    """Synapse-weighted mean column offset (pre − post) for each (pre_type, post_type) in
    `pairs`, same side only, in flyvis's axial convention (see `mcns_to_axial`).
    Columns: s, t, du, dv, n. Pairs with no same-side synapses are absent."""
    u, v = mcns_to_axial(neurons["hex1"].to_numpy(), neurons["hex2"].to_numpy())
    k = neurons.select("index", "type", "side").with_columns(pl.Series("u", u), pl.Series("v", v))
    k = k.filter(pl.col("u").is_not_null())
    e = (
        edges.join(k.rename({"index": "pre_idx", "type": "s", "side": "ss", "u": "us", "v": "vs"}), on="pre_idx")
        .join(k.rename({"index": "post_idx", "type": "t", "side": "ts", "u": "ut", "v": "vt"}), on="post_idx")
        .filter(pl.col("ss") == pl.col("ts"))
        .filter(pl.struct("s", "t").is_in([{"s": s, "t": t} for s, t in pairs]))
    )
    return (
        e.group_by("s", "t").agg(
            ((pl.col("us") - pl.col("ut")) * pl.col("syn_count")).sum().alias("du"),
            ((pl.col("vs") - pl.col("vt")) * pl.col("syn_count")).sum().alias("dv"),
            pl.col("syn_count").sum().alias("n"),
        )
        .with_columns(pl.col("du") / pl.col("n"), pl.col("dv") / pl.col("n"))
        .sort("t", "s")
    )


def radial_index(neurons: pl.DataFrame, edges: pl.DataFrame, alignment, fv_col, geometries, post_type: str,
                 min_inputs: int = 20) -> np.ndarray:
    """Expansion selectivity of `post_type`'s T4/T5 inputs, per cell, as seen through an alignment:
    +1 if every input's preferred direction points away from the cell's receptive-field centre
    (a looming detector, e.g. LPLC2), −1 if towards it, ~0 if unrelated. A pure-anatomy cross-check
    of the lattice symmetry: only the true one makes LPLC2 a looming detector.

    `fv_col[fv_index]` is the flyvis neuron's column index into `geometries[eye].az_deg/el_deg`."""
    typ = neurons["type"].to_numpy(); side = neurons["side"].to_numpy()
    t45 = np.isin(typ, [f"T{k}{d}" for k in "45" for d in "abcd"])
    pos = {}
    for fv, s, lif in zip(alignment.table["fv_index"].to_numpy(), alignment.table["eye"].to_numpy(),
                          alignment.table["lif_index"].to_numpy()):
        if t45[lif]:
            pos[lif] = (geometries[s].az_deg[fv_col[fv]], geometries[s].el_deg[fv_col[fv]])
    pd_of = {"a": (1.0, 0.0), "b": (-1.0, 0.0), "c": (0.0, 1.0), "d": (0.0, -1.0)}  # right eye: +az = front-to-back
    post = np.where(typ == post_type)[0]
    e = edges.filter(pl.col("post_idx").is_in(post.tolist()) & pl.col("pre_idx").is_in(np.where(t45)[0].tolist()))
    e = e.select("pre_idx", "post_idx", "syn_count").to_numpy()
    out = []
    for p in post:
        rows = [(pre, w) for pre, q, w in e[e[:, 1] == p] if pre in pos]
        if len(rows) < min_inputs:
            continue
        P = np.array([pos[pre] for pre, _ in rows]); W = np.array([w for _, w in rows], dtype=float)
        centre = (P * W[:, None]).sum(0) / W.sum()
        sgn = 1.0 if side[p] == "R" else -1.0
        cosines = []
        for (pre, _), pp in zip(rows, P):
            pd = np.array(pd_of[typ[pre][-1]]); pd[0] *= sgn
            r = pp - centre; nr = np.linalg.norm(r)
            cosines.append(0.0 if nr < 1e-6 else float(np.dot(r / nr, pd)))
        out.append(np.average(cosines, weights=W))
    return np.array(out)
