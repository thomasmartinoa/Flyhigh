"""flyvis cell-type names → male-CNS `type` names. Exact matches need no entry.

One flyvis type can stand for several male-CNS types (the male CNS splits R7/R8 by rhabdomere
subtype and TmY9 into a/b); every listed male-CNS neuron is then driven by the same flyvis cell.
Mi3, Mi11, Mi12 and Tm28 have no male-CNS type at all in v1.0 and are simply never matched.
"""

FLYVIS_TO_MCNS: dict[str, tuple[str, ...]] = {
    **{f"R{i}": ("R1-R6",) for i in range(1, 7)},
    "R7": ("R7y", "R7p", "R7d", "R7_unclear"),
    "R8": ("R8y", "R8p", "R8d", "R8_unclear"),
    "CT1(Lo1)": ("CT1",),
    "CT1(M10)": ("CT1",),
    "Am": ("Am1",),
    "TmY9": ("TmY9a", "TmY9b"),
}


def mcns_types(flyvis_type: str) -> tuple[str, ...]:
    return FLYVIS_TO_MCNS.get(flyvis_type, (flyvis_type,))


def mcns_type(flyvis_type: str) -> str:
    """The first (canonical) male-CNS name, for labels and one-to-one uses."""
    return mcns_types(flyvis_type)[0]
