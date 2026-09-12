"""Shared fixtures: a tiny synthetic male-CNS-shaped dataset written as feather files."""

from pathlib import Path

import polars as pl
import pytest

# Mirrors the columns we actually read from the real tables (see flyhigh/data/connectome.py).
# 6 bodies: 5 traced neurons + 1 glia segment that must be dropped.
ANNOTATIONS = pl.DataFrame(
    {
        "bodyId": [10, 20, 30, 40, 50, 60],
        "type": ["LB3a", "MN9", "GNG042", "T4a", "R1-R6", None],
        "instance": ["LB3a_L", "MN9_R", "GNG042_L", "T4a", "R1-R6", None],
        "class": ["gustatory", None, None, None, None, None],
        "superclass": [
            "cb_sensory", "cb_motor", "cb_intrinsic", "ol_intrinsic", "ol_sensory", None,
        ],
        "somaSide": ["L", "R", "L", "R", "R", None],
        "status": ["Traced", "Traced", "Traced", "Traced", "Traced", "Glia"],
        "dimorphism": [None, None, "male-specific", None, None, None],
        "assignedOlHex1": [None, None, None, 3.0, 3.0, None],
        "assignedOlHex2": [None, None, None, -2.0, -2.0, None],
    }
)

NEUROTRANSMITTERS = pl.DataFrame(
    {
        "body": [10, 20, 30, 40, 50],
        "consensus_nt": ["acetylcholine", "acetylcholine", "gaba", "unclear", "histamine"],
        "celltype_predicted_nt": ["acetylcholine", "acetylcholine", "gaba", "acetylcholine", "histamine"],
        "predicted_nt_confidence": [0.9, 0.8, 0.95, 0.4, 0.99],
    }
)

# pre -> post with synapse counts. Includes an edge to the glia body (must be dropped)
# and a weak edge (weight 1) that a min_syn filter can remove.
WEIGHTS = pl.DataFrame(
    {
        "body_pre": [10, 30, 50, 40, 10, 20],
        "body_post": [20, 20, 40, 30, 60, 10],
        "weight": [12, 7, 5, 3, 9, 1],
    }
)


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    ANNOTATIONS.write_ipc(tmp_path / "body-annotations-male-cns-v1.0-minconf-0.5.feather")
    NEUROTRANSMITTERS.write_ipc(tmp_path / "body-neurotransmitters-male-cns-v1.0.feather")
    WEIGHTS.write_ipc(tmp_path / "connectome-weights-male-cns-v1.0-minconf-0.5.feather")
    return tmp_path
