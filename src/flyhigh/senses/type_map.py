"""flyvis cell-type names → male-CNS `type` names. Exact matches need no entry."""

FLYVIS_TO_MCNS = {
    **{f"R{i}": "R1-R6" for i in range(1, 7)},
    "CT1(Lo1)": "CT1",
    "CT1(M10)": "CT1",
    "Am": "Am1",
}


def mcns_type(flyvis_type: str) -> str:
    return FLYVIS_TO_MCNS.get(flyvis_type, flyvis_type)
