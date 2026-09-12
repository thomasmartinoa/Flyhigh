# 01 — Meet the connectome

## What a connectome is (and isn't)

A **connectome** is a wiring diagram: for every neuron, which other neurons it forms
synapses onto, and roughly how many synapses. The male-CNS connectome is the first one
covering an entire central nervous system of an adult animal with complex behaviour:

| | |
|---|---|
| Animal | one adult male *Drosophila melanogaster* |
| Regions | central brain + both optic lobes + ventral nerve cord (VNC, the fly's "spinal cord") |
| Neurons | **165,122** traced ("Traced" status in the annotation table) |
| Synapses | **124 million** post-synaptic sites between traced neurons |
| Connections | 25.6 million distinct neuron→neuron edges |
| Imaging | FIB-SEM at 8 nm isotropic voxels (Janelia FlyEM) |
| Reconstruction | Google flood-filling networks + PATHFINDER, then human proofreading |
| Paper | Cell 2026, doi:10.1016/j.cell.2026.08.015 (CC-BY) |

What it **does not** contain:

- **Synaptic strength.** Synapse *count* is a proxy; the real efficacy of a connection is unknown.
- **Neuron dynamics.** No time constants, thresholds, ion channels, spike shapes.
- **Neuromodulation and plasticity.** Dopamine, octopamine, learning — all invisible.
- **Sensory transduction.** How light, taste or touch becomes spikes is outside the wiring.
- **Gap junctions** (electrical synapses) are not annotated.

The only functional information layered on top is the **neurotransmitter prediction** for
each neuron (a classifier run on the EM images of its synapses), from which we infer the
*sign* of each connection: acetylcholine, dopamine, octopamine, serotonin → excitatory;
GABA, glutamate → inhibitory (Dale's law: one neuron, one main transmitter). We also treat
histamine as inhibitory, because fly photoreceptors release histamine onto histamine-gated
chloride channels in the lamina.

## The three tables we use

Downloaded by `python -m flyhigh.data.download` into `data/raw/` from
`https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/`:

| File | Rows | What we take from it |
|---|---|---|
| `body-annotations-…feather` (14 MB) | 211,577 segments | `status == "Traced"` → neurons; `type`, `superclass`, `class`, `somaSide`, `dimorphism`, eye column `assignedOlHex1/2` (on columnar lamina/medulla neurons such as L1/L2/Tm1, ≈ 880 columns per eye) |
| `body-neurotransmitters-…feather` (43 MB) | 1.8 M segments | `consensus_nt` (fallback `celltype_predicted_nt`) → sign |
| `connectome-weights-…feather` (1.05 GB) | 152 M segment pairs | `body_pre, body_post, weight` (= synapse count) |

`Connectome.load()` joins them, keeps only edges between traced neurons (that drops 27 M
synapses onto glia, orphan fragments and untraced segments) and caches the result as two
parquet files in `data/cache/` (110 MB, loads in ~1 s).

## Where the neurons live

| region (from `superclass`) | neurons | in-synapses / neuron |
|---|---|---|
| optic lobes (`ol_*`, `visual_projection`, `visual_centrifugal`) | 103,270 | 481 |
| central brain (`cb_*`, `descending_neuron`, …) | 38,546 | 1,078 |
| VNC (`vnc_*`, `ascending_neuron`, …) | 22,790 | 943 |

Two thirds of all neurons are in the eyes. The central brain is the densest.

Some neuron types you will meet again in this project:

| type | what it is | why we care |
|---|---|---|
| `R1-R6`, `R7`, `R8` | photoreceptors (histaminergic) | our camera input goes here (M2) |
| `L1`, `L2` | lamina neurons | first stage of motion vision |
| `T4a-d`, `T5a-d` | direction-selective motion detectors | one type per direction |
| `HSN/HSE/HSS`, `VS` | lobula-plate tangential cells | integrate wide-field motion → optomotor reflex |
| `LC4`, `LPLC2` | looming-sensitive visual projection neurons | "something is approaching" |
| `DNp01` (giant fiber, GF) | descending neuron | triggers the escape take-off |
| `DNa01`, `DNa02` | descending neurons | steering / turning |
| `LB3a/b/c` | labellar sugar-sensing taste neurons | our validation input |
| `MN9` | proboscis motor neuron | our validation output ("feeding") |
| `KC*` | Kenyon cells (mushroom body) | memory; 4,064 cells, 55 % of their input is other KCs |
| `APL` | one GABAergic neuron per hemisphere | keeps Kenyon cells sparse |

Sex-specific biology is annotated too: `dimorphism` marks 1,258 male-specific and 771
sexually dimorphic neurons, and `fruDsx` marks expression of the courtship genes
*fruitless* / *doublesex*.

## Quirks worth knowing

- Sensory neurons have **no `somaSide`** — their cell bodies are in the periphery.
- The gustatory neuron types use anatomical names (`LB3b`, not `Gr64f`); the companion
  gustatory paper maps them to modalities (LB3 = sugar/water/low salt, LB1 = bitter).
- Cell-type names are shared with FlyWire and the hemibrain where possible
  (`flywireType`, `hemibrainType` columns), so results from those datasets transfer.
- 3,275 traced neurons have an "unclear" transmitter; we treat them as excitatory.

See notebook `01_meet_the_connectome.ipynb` for the same content as live code and plots.
