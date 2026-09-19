# scripts/build_alignment.py
"""Build data/cache/alignment.parquet from the real connectome and the flyvis model; print a report.

Steps: (1) give T4/T5 and the other unlabelled columnar types a column from their partners,
(2) pick the lattice symmetry from Mi9/Mi4→T4 and Tm9/Tm4/Tm2→T5 offsets, (3) match columns.
"""
import numpy as np
import polars as pl

from flyhigh.data.connectome import Connectome
from flyhigh.senses.alignment import ColumnAlignment, choose_symmetry
from flyhigh.senses.columns import infer_columns, partner_offsets
from flyhigh.senses.flyvis_eye import FlyvisEye
from flyhigh.senses.type_map import mcns_types

# Connections whose column offset is set by direction selectivity, hence orientation-revealing.
OFFSET_PAIRS = [(s, f"T4{d}") for s in ("Mi9", "Mi4") for d in "abcd"] + \
               [(s, f"T5{d}") for s in ("Tm9", "Tm4", "Tm2") for d in "abcd"]

c = Connectome.load("data/raw")
eye = FlyvisEye()
targets = sorted({m for t in np.unique(eye.types) for m in mcns_types(t)})

neurons = infer_columns(c.neurons, c.edges, targets)
inferred = neurons.filter(pl.col("hex_source") == "inferred")
print(f"columns inferred for {inferred.height} neurons of {inferred['type'].n_unique()} types; "
      f"median vote share {inferred['hex_share'].median():.2f}")
print(inferred.group_by("type").agg(pl.len().alias("n"), pl.col("hex_share").median().alias("share"))
      .sort("share").head(8))

sym, scores = choose_symmetry(eye.edge_offsets(), partner_offsets(neurons, c.edges, OFFSET_PAIRS))
ranked = sorted(scores.items(), key=lambda kv: kv[1])
print("symmetry:", sym, " rms(columns):", " ".join(f"{s}={r:.2f}" for s, r in ranked[:3]), "...")

al = ColumnAlignment.build(eye.types, eye.u, eye.v, neurons, symmetry=sym)
al.save("data/cache/alignment.parquet")
print("matched rows:", al.table.height, " flyvis neurons:", eye.n_neurons, "x 2 eyes")
cov = pl.DataFrame({"type": list(al.coverage), "coverage": list(al.coverage.values())}).sort("coverage")
print(cov.head(12)); print("mean coverage:", cov["coverage"].mean())
print("types with zero coverage (add to type_map or accept):", cov.filter(pl.col("coverage") == 0)["type"].to_list())
