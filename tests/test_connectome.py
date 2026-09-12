import numpy as np
import polars as pl
import torch

from flyhigh.data.connectome import Connectome


def test_load_keeps_only_traced_neurons(raw_dir):
    c = Connectome.load(raw_dir, cache=False)
    assert c.n_neurons == 5
    assert c.neurons["body_id"].to_list() == [10, 20, 30, 40, 50]
    assert c.neurons["index"].to_list() == [0, 1, 2, 3, 4]


def test_load_assigns_neurotransmitter_sign(raw_dir):
    c = Connectome.load(raw_dir, cache=False)
    sign = dict(zip(c.neurons["body_id"], c.neurons["nt_sign"]))
    assert sign[10] == 1  # acetylcholine
    assert sign[30] == -1  # gaba
    assert sign[50] == -1  # histamine (photoreceptor -> lamina is sign-inverting)
    assert sign[40] == 1  # consensus unclear -> fall back to cell-type prediction (ACh)


def test_load_drops_edges_touching_non_neurons(raw_dir):
    c = Connectome.load(raw_dir, cache=False)
    assert 60 not in c.edges["pre_idx"].to_list() and 60 not in c.edges["post_idx"].to_list()
    assert c.n_edges == 5  # 6 rows minus the one to the glia body


def test_edge_weight_is_syn_count_times_presynaptic_sign(raw_dir):
    c = Connectome.load(raw_dir, cache=False)
    e = c.edges.filter((pl.col("pre_idx") == 2) & (pl.col("post_idx") == 1))  # GNG042 (gaba) -> MN9
    assert e["syn_count"].item() == 7
    assert e["weight"].item() == -7


def test_min_syn_filters_weak_edges(raw_dir):
    c = Connectome.load(raw_dir, min_syn=2, cache=False)
    assert c.n_edges == 4


def test_ids_by_type_regex(raw_dir):
    c = Connectome.load(raw_dir, cache=False)
    np.testing.assert_array_equal(c.ids_by_type(r"^MN9$"), [1])
    np.testing.assert_array_equal(c.ids_by_type(["LB3a", "GNG042"]), [0, 2])


def test_to_sparse_is_post_by_pre_with_weights(raw_dir):
    c = Connectome.load(raw_dir, cache=False)
    W = c.to_sparse(device="cpu").to_dense()
    assert W.shape == (5, 5)
    assert W[1, 0] == 12  # LB3a -> MN9
    assert W[1, 2] == -7  # GNG042 -| MN9
    assert W[0, 1] == 1  # MN9 -> LB3a (weak edge kept with default min_syn)
    assert W[3, 4] == -5  # R1-R6 -| T4a (histamine)


def test_subgraph_reindexes_neurons_and_edges(raw_dir):
    c = Connectome.load(raw_dir, cache=False)
    sub = c.subgraph(body_ids=[20, 30, 10])
    assert sub.n_neurons == 3
    assert sub.neurons["body_id"].to_list() == [10, 20, 30]  # original order preserved
    assert sub.neurons["index"].to_list() == [0, 1, 2]
    W = sub.to_sparse(device="cpu").to_dense()
    assert W[1, 0] == 12 and W[1, 2] == -7 and W[0, 1] == 1
    assert sub.n_edges == 3


def test_subgraph_by_region(raw_dir):
    c = Connectome.load(raw_dir, cache=False)
    brain = c.subgraph(region="brain")
    assert set(brain.neurons["body_id"].to_list()) == {10, 20, 30}
    optic = c.subgraph(region="optic")
    assert set(optic.neurons["body_id"].to_list()) == {40, 50}


def test_load_uses_parquet_cache(raw_dir, tmp_path):
    cache = tmp_path / "cache"
    c1 = Connectome.load(raw_dir, cache_dir=cache)
    assert (cache / "neurons.parquet").exists() and (cache / "edges.parquet").exists()
    c2 = Connectome.load(raw_dir, cache_dir=cache)
    assert c2.neurons.equals(c1.neurons) and c2.edges.equals(c1.edges)


def test_describe_counts(raw_dir):
    d = Connectome.load(raw_dir, cache=False).describe()
    assert d["n_neurons"] == 5 and d["n_edges"] == 5 and d["n_synapses"] == 28
    assert d["by_region"]["brain"] == 3 and d["by_region"]["optic"] == 2
