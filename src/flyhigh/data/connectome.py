"""The male-CNS connectome as an indexed neuron table plus a signed edge list.

Built from the three FlyEM "flat connectome" tables (see `flyhigh.data.download`):
  body-annotations   : one row per segment; `status == "Traced"` marks real neurons (~165k)
  body-neurotransmitters : per-neuron NT prediction (`consensus_nt`, fallback `celltype_predicted_nt`)
  connectome-weights : body_pre -> body_post synapse counts for *all* segments (~152M rows)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch

ANNOTATIONS_FILE = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NEUROTRANSMITTERS_FILE = "body-neurotransmitters-male-cns-v1.0.feather"
WEIGHTS_FILE = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"

# Dale's-law sign per neurotransmitter, following Shiu et al. 2024 (GABA/Glu inhibitory,
# everything else excitatory) plus histamine → inhibitory: fly photoreceptors release
# histamine onto histamine-gated chloride channels in lamina neurons (sign-inverting).
NT_SIGN = {
    "acetylcholine": 1,
    "dopamine": 1,
    "octopamine": 1,
    "serotonin": 1,
    "gaba": -1,
    "glutamate": -1,
    "histamine": -1,
}

# superclass → coarse region used by `subgraph(region=...)`.
REGION_OF_SUPERCLASS = {
    "ol_intrinsic": "optic",
    "ol_sensory": "optic",
    "visual_projection": "optic",
    "visual_centrifugal": "optic",
    "visual_projection_tbc": "optic",
    "cb_intrinsic": "brain",
    "cb_sensory": "brain",
    "cb_motor": "brain",
    "cb_endocrine": "brain",
    "cb_efferent": "brain",
    "cb_sensory_tbc": "brain",
    "descending_neuron": "brain",
    "descending_neuron_tbc": "brain",
    "sensory_descending": "brain",
    "efferent_descending": "brain",
    "vnc_intrinsic": "vnc",
    "vnc_sensory": "vnc",
    "vnc_motor": "vnc",
    "vnc_efferent": "vnc",
    "vnc_endocrine": "vnc",
    "vnc_tbc": "vnc",
    "vnc_sensory_tbc": "vnc",
    "ascending_neuron": "vnc",
    "sensory_ascending": "vnc",
    "sensory_ascending_tbc": "vnc",
    "efferent_ascending": "vnc",
    "ENS": "vnc",
}

NEURON_COLUMNS = [
    "body_id", "index", "type", "instance", "class", "superclass", "region", "side",
    "dimorphism", "hex1", "hex2", "nt", "nt_sign", "nt_confidence",
]


@dataclass
class Connectome:
    neurons: pl.DataFrame  # one row per neuron, `index` is the dense 0..N-1 id used in edges
    edges: pl.DataFrame  # pre_idx, post_idx, syn_count, weight = syn_count * nt_sign[pre]

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(
        cls,
        raw_dir: str | Path,
        min_syn: int = 1,
        cache: bool = True,
        cache_dir: str | Path | None = None,
    ) -> Connectome:
        raw_dir = Path(raw_dir)
        cache_dir = Path(cache_dir) if cache_dir else raw_dir.parent / "cache"
        suffix = "" if min_syn == 1 else f"-minsyn{min_syn}"
        neurons_pq = cache_dir / f"neurons{suffix}.parquet"
        edges_pq = cache_dir / f"edges{suffix}.parquet"
        if cache and neurons_pq.exists() and edges_pq.exists():
            return cls(pl.read_parquet(neurons_pq), pl.read_parquet(edges_pq))

        neurons = _build_neurons(raw_dir)
        edges = _build_edges(raw_dir, neurons, min_syn)
        if cache:
            cache_dir.mkdir(parents=True, exist_ok=True)
            neurons.write_parquet(neurons_pq)
            edges.write_parquet(edges_pq)
        return cls(neurons, edges)

    # ---------------------------------------------------------------- basic info
    @property
    def n_neurons(self) -> int:
        return self.neurons.height

    @property
    def n_edges(self) -> int:
        return self.edges.height

    def describe(self) -> dict:
        by = lambda col: dict(self.neurons.group_by(col).len().sort(col).iter_rows())
        return {
            "n_neurons": self.n_neurons,
            "n_edges": self.n_edges,
            "n_synapses": int(self.edges["syn_count"].sum()),
            "by_region": by("region"),
            "by_superclass": by("superclass"),
            "by_nt": by("nt"),
        }

    # ------------------------------------------------------------------ queries
    def ids_by_type(self, pattern: str | list[str]) -> np.ndarray:
        """Dense indices of neurons whose `type` matches a regex or is in a list of names."""
        if isinstance(pattern, str):
            mask = self.neurons["type"].str.contains(pattern)
        else:
            mask = self.neurons["type"].is_in(pattern)
        return self.neurons.filter(mask.fill_null(False))["index"].to_numpy()

    def subgraph(
        self,
        body_ids=None,
        region: str | None = None,
        mask: np.ndarray | None = None,
    ) -> Connectome:
        """Restrict to a neuron subset and re-index densely (original order preserved)."""
        if mask is None:
            if body_ids is not None:
                mask = self.neurons["body_id"].is_in(list(body_ids)).to_numpy()
            elif region is not None:
                mask = (self.neurons["region"] == region).to_numpy()
            else:
                raise ValueError("give one of body_ids, region, mask")
        keep = self.neurons.filter(pl.Series(mask))
        old_to_new = np.full(self.n_neurons, -1, dtype=np.int64)
        old_to_new[keep["index"].to_numpy()] = np.arange(keep.height)
        edges = self.edges.with_columns(
            pl.col("pre_idx").map_batches(lambda s: pl.Series(old_to_new[s.to_numpy()])),
            pl.col("post_idx").map_batches(lambda s: pl.Series(old_to_new[s.to_numpy()])),
        ).filter((pl.col("pre_idx") >= 0) & (pl.col("post_idx") >= 0))
        neurons = keep.with_columns(pl.Series("index", np.arange(keep.height, dtype=np.int64)))
        return Connectome(neurons, edges)

    def drop_edges_between(self, pre, post) -> Connectome:
        """Copy without the edges going from any of `pre` to any of `post` (dense indices)."""
        pre_s = pl.Series(list(pre), dtype=pl.Int64)
        post_s = pl.Series(list(post), dtype=pl.Int64)
        drop = pl.col("pre_idx").is_in(pre_s.implode()) & pl.col("post_idx").is_in(post_s.implode())
        return Connectome(self.neurons, self.edges.filter(~drop))

    def to_sparse(self, device="cpu", dtype=torch.float32) -> torch.Tensor:
        """Sparse CSR W with W[post, pre] = signed synapse count, so g += W @ spikes."""
        post = torch.from_numpy(self.edges["post_idx"].to_numpy().copy())
        pre = torch.from_numpy(self.edges["pre_idx"].to_numpy().copy())
        w = torch.from_numpy(self.edges["weight"].to_numpy().copy()).to(dtype)
        n = self.n_neurons
        coo = torch.sparse_coo_tensor(
            torch.stack([post, pre]), w, (n, n), check_invariants=True
        ).coalesce()
        return coo.to_sparse_csr().to(device)


# --------------------------------------------------------------------- builders
def _build_neurons(raw_dir: Path) -> pl.DataFrame:
    ann = pl.read_ipc(raw_dir / ANNOTATIONS_FILE, columns=[
        "bodyId", "type", "instance", "class", "superclass", "somaSide", "status",
        "dimorphism", "assignedOlHex1", "assignedOlHex2",
    ]).filter(pl.col("status") == "Traced")
    nt = pl.read_ipc(raw_dir / NEUROTRANSMITTERS_FILE, columns=[
        "body", "consensus_nt", "celltype_predicted_nt", "predicted_nt_confidence",
    ]).rename({"body": "bodyId"})
    df = ann.join(nt, on="bodyId", how="left")
    # consensus first; if "unclear"/missing fall back to the cell-type-level prediction
    nt_col = (
        pl.when(pl.col("consensus_nt").is_in(list(NT_SIGN)))
        .then(pl.col("consensus_nt"))
        .otherwise(pl.col("celltype_predicted_nt"))
        .fill_null("unclear")
    )
    df = (
        df.with_columns(nt_col.alias("nt"))
        .with_columns(
            pl.col("nt").replace_strict(NT_SIGN, default=1, return_dtype=pl.Int8).alias("nt_sign"),
            pl.col("superclass").replace_strict(REGION_OF_SUPERCLASS, default="unknown").alias("region"),
        )
        .sort("bodyId")
        .with_row_index("index")
        .rename({
            "bodyId": "body_id", "somaSide": "side", "assignedOlHex1": "hex1",
            "assignedOlHex2": "hex2", "predicted_nt_confidence": "nt_confidence",
        })
        .with_columns(pl.col("index").cast(pl.Int64))
        .select(NEURON_COLUMNS)
    )
    return df


def _build_edges(raw_dir: Path, neurons: pl.DataFrame, min_syn: int) -> pl.DataFrame:
    idx = neurons.select("body_id", "index", "nt_sign")
    edges = (
        pl.scan_ipc(raw_dir / WEIGHTS_FILE)
        .filter(pl.col("weight") >= min_syn)
        .join(idx.lazy().rename({"body_id": "body_pre", "index": "pre_idx"}), on="body_pre", how="inner")
        .join(
            idx.lazy().select(pl.col("body_id").alias("body_post"), pl.col("index").alias("post_idx")),
            on="body_post", how="inner",
        )
        .select(
            "pre_idx", "post_idx",
            pl.col("weight").cast(pl.Int32).alias("syn_count"),
            (pl.col("weight") * pl.col("nt_sign")).cast(pl.Int32).alias("weight"),
        )
        .sort("post_idx", "pre_idx")
        .collect()
    )
    return edges
