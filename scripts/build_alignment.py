# scripts/build_alignment.py
"""Build data/cache/alignment.parquet from the real connectome and the flyvis model; print coverage."""
import polars as pl

from flyhigh.data.connectome import Connectome
from flyhigh.senses.alignment import ColumnAlignment
from flyhigh.senses.flyvis_eye import FlyvisEye

c = Connectome.load("data/raw")
eye = FlyvisEye()
al = ColumnAlignment.build(eye.types, eye.u, eye.v, c.neurons)
al.save("data/cache/alignment.parquet")
print("chosen symmetry:", al.transform, " matched rows:", al.table.height, "of", 2 * eye.n_neurons)
cov = pl.DataFrame({"type": list(al.coverage), "coverage": list(al.coverage.values())}).sort("coverage")
print(cov.head(15)); print("mean coverage:", cov["coverage"].mean())
print("types with zero coverage (add to type_map or accept):", cov.filter(pl.col("coverage") == 0)["type"].to_list())
