"""Reproduce the gustatory-connectome paper's model experiment on the male CNS.

Protocol (Cell 2025/26, "The complete gustatory connectome of adult Drosophila", Fig. S17):
  * activate the appetitive labellar GRNs LB3a/b/c at 200 Hz  -> proboscis motor neuron MN9 fires
  * additionally silence Clavicle (ANXXX462a)                   -> MN9 firing decreases (their prediction)
Plus our own sanity checks: no input -> no activity; activity stays local (< 2 % of neurons).

Exit code 0 iff the required checks pass. The Clavicle prediction is reported but not required
(see docs/02-lif-brain.md for why).
"""

import sys

from _common import MAX_RATE_HZ, experiment, load, summary


def main():
    c = load()
    lb3 = c.ids_by_type(r"^LB3[abc]$")
    mn9 = c.ids_by_type(r"^MN9$")
    clavicle = c.ids_by_type(r"^ANXXX462a$")
    print(f"LB3 GRNs: {len(lb3)}   MN9: {len(mn9)}   Clavicle: {len(clavicle)}   neurons: {c.n_neurons}")

    ok = True
    r0, rec0 = experiment(c, [])
    print(f"[baseline]            spikes={rec0.counts.sum().item()}")
    ok &= rec0.counts.sum().item() == 0

    r1, rec1 = experiment(c, [(lb3, 200.0)])
    active, resp, top = summary(c, r1, exclude_types=("LB3",))
    mn9_1 = r1[mn9].mean()
    print(f"[LB3 @200 Hz]         MN9 L/R = {r1[mn9[0]]:.0f}/{r1[mn9[1]]:.0f} Hz ({mn9_1 / MAX_RATE_HZ:.0%} of max)  active={active}  >5Hz={resp}  top={top}")
    ok &= mn9_1 > 5.0
    ok &= active < 0.02 * c.n_neurons

    r2, rec2 = experiment(c, [(lb3, 200.0)], silence=clavicle)
    mn9_2 = r2[mn9].mean()
    active2, resp2, top2 = summary(c, r2, exclude_types=("LB3",))
    print(f"[LB3 @200 Hz -Clav]   MN9 L/R = {r2[mn9[0]]:.0f}/{r2[mn9[1]]:.0f} Hz  active={active2}  >5Hz={resp2}")
    print(f"  paper prediction (MN9 decreases without Clavicle): {'reproduced' if mn9_2 < 0.8 * mn9_1 else 'NOT reproduced'}  (not required)")

    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
