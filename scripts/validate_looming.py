"""Looming-escape pathway check: activate the looming-sensitive visual projection neurons
LC4 and LPLC2 -> the giant fiber (DNp01) must spike. This is the M2 escape reflex.
"""

import sys

from _common import experiment, load, summary


def main():
    c = load()
    lc4 = c.ids_by_type(r"^LC4$")
    lplc2 = c.ids_by_type(r"^LPLC2$")
    gf = c.ids_by_type(r"^DNp01$")
    print(f"LC4: {len(lc4)}   LPLC2: {len(lplc2)}   GF (DNp01): {len(gf)}")

    ok = True
    for hz in (50.0, 100.0):
        r, rec = experiment(c, [(lc4, hz), (lplc2, hz)], duration_ms=500.0, record=gf)
        active, resp, top = summary(c, r, exclude_types=("LC4", "LPLC2"))
        df = rec.to_polars()
        first = df.filter(df["neuron"].is_in(list(gf)))["t_ms"].min()
        print(f"[LC4+LPLC2 @{hz:.0f} Hz]  GF L/R = {r[gf[0]]:.0f}/{r[gf[1]]:.0f} Hz  first GF spike at {first} ms  active={active}  >5Hz={resp}  top={top}")
        if hz == 100.0:
            ok &= r[gf].max() > 0 and first is not None and first < 50.0
            ok &= active < 0.02 * c.n_neurons
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
