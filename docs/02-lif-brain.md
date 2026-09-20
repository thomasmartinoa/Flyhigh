# 02 — Running a fly brain: the LIF model

## The model

Every one of the 165,122 neurons is a **leaky integrate-and-fire** (LIF) unit — a leaky
capacitor with a threshold. This is the model of Shiu et al. 2024 (*Nature*, "A Drosophila
computational brain model reveals sensorimotor processing"), which we re-implemented in
PyTorch so it runs on the GPU and can take live sensory input every step.

```
dv/dt = (v_rest − v + g) / t_mbr        membrane voltage (mV), frozen while refractory
dg/dt = −g / tau                        synaptic input, frozen while refractory
spike  when v > v_th
   →   v = v_reset,  g = 0,  refractory for t_rfc
each presynaptic spike, after delay t_dly:   g[post] += w_syn × (±synapse count)
```

| parameter | value | source |
|---|---|---|
| v_rest = v_reset | −52 mV | Kakaria & de Bivort 2017 |
| v_th | −45 mV | " |
| t_mbr (membrane τ) | 20 ms | " |
| tau (synaptic τ) | 5 ms | Jürgensen et al. 2021 |
| t_rfc (refractory) | 2.2 ms | Lazar et al. 2021 |
| t_dly (delay) | 1.8 ms | Paul et al. 2015 |
| w_syn | **0.275 mV / synapse** (FlyWire) | free parameter, calibrated by Shiu et al. |
| dt | 0.1 ms | |

Two details matter more than they look:

1. **`g = 0` on spike.** The synaptic variable is wiped every time a neuron fires, so a
   neuron has to receive *fresh* input worth ≥ 7 mV to fire again. This is what keeps the
   network from running away. Without it (our first attempt) 17,000 neurons light up
   instead of 400.
2. **Frozen during refractory, but still accumulating.** `v` and `g` do not decay during
   the 2.2 ms refractory period, yet incoming spikes still add to `g` (Brian2's
   `unless refractory` semantics).

Optogenetic-style activation follows Shiu: a Poisson train of "kicks" (0.275 × 250 mV
straight into `v`, so every kick fires the neuron) at the chosen rate, with the target's
refractory period set to 0.

### Integration

We integrate the linear ODE pair exactly over one step (exponential Euler):

```
a = exp(−dt/t_mbr),  b = exp(−dt/tau),  c = tau/(tau − t_mbr) · (b − a)
v ← v_rest + (v − v_rest)·a + g·c + I_ext·(1 − a)
g ← g·b
```

`I_ext` is a graded drive in mV (the steady-state depolarisation a constant current would
produce) — that is how camera-derived photoreceptor currents will enter in M2.

### Implementation notes

- **Batched agents.** State tensors are `(n_agents, N)`. Two flies = two rows, same wiring.
- **Event-driven propagation.** Typically < 0.1 % of neurons spike per step, so instead of
  a full sparse matmul we gather only the out-edges of the neurons that spiked
  (`_propagate_event`, CSR indexed by presynaptic neuron). A full SpMV kernel is kept for
  cross-checking (`propagation="spmv"`); tests assert both give identical spikes.
- **Speed on the RTX 3070 Ti Laptop (full CNS, 0.1 ms steps):** 0.56 ms/step at full
  GPU clocks (≈ 5.6 s of compute per simulated second, independent of agent count).
  When the laptop drops the GPU to P8 / 210 MHz (battery / power-saver) it is 1.3 ms/step —
  check `nvidia-smi` if numbers look off. GPU memory: < 1 GB.

## Validation 1 — the engine reproduces the published model

We ran our engine on Shiu's own FlyWire-783 connectivity table with their 20 right-hemisphere
sugar-sensing GRNs (their `example.ipynb`):

| | Shiu et al. (Brian2) | ours (PyTorch) |
|---|---|---|
| sugar GRNs @ 200 Hz → neurons with any activity | "about 400" | **405** |
| spikes per 1 s trial | ~13 k | 20 k |
| MN9 (proboscis motor neuron) | fires | 104 Hz |

Good enough to trust the engine.

## Validation 2 — the same model on the male CNS ignites

The male CNS is not FlyWire. FIB-SEM at 8 nm resolves far more of the small synapses:

| | FlyWire-783 (ssTEM) | male CNS v1.0 (FIB-SEM) |
|---|---|---|
| synapses / neuron (input) | 397 | **751** |
| spectral radius of the signed synapse matrix | 2,164 | **4,089** |

So at w_syn = 0.275 every neuron receives ~1.9× the drive Shiu calibrated for, and the
network explodes: activating the 51 sugar GRNs (LB3a/b/c) recruits 20,000 neurons firing
at the refractory ceiling. Simply scaling w_syn down by the spectral-radius ratio
(→ 0.146 mV) is not enough: after ~100 ms of correct, local activity the antennal lobe
ignites through a clique of 30 **cholinergic** local neurons (`lLN1_bc`, ≥ 162 synapses
onto each other — a one-spike-sufficient loop absent from FlyWire) and pulls in the
projection neurons and then the 4,064 Kenyon cells (55 % of whose input is other Kenyon
cells). Real flies keep those populations quiet with inhibition, synaptic depression and
neuromodulation that a plain LIF has no way to express.

We tried, and rejected, two structural hacks (dropping KC→KC synapses; dropping all
synapses onto sensory neurons) — neither stops the ignition on its own.

### Our calibration: lower gain + short-term synaptic depression

`LIFBrain.for_male_cns()` uses

- **w_syn = 0.20 mV**
- **Tsodyks–Markram short-term depression** on every *central* synapse: each presynaptic
  neuron has a resource `x ∈ [0,1]` scaling its outgoing weights; `x ← x·(1 − u)` on each
  spike with `u = 0.2`, recovering with `τ_rec = 300 ms`. Sustained firing at 400 Hz thus
  weakens a neuron's output ~25×, while 20 Hz signalling keeps ~45 % strength.
- **Sensory afferents exempt** from depression (photoreceptors, GRNs, ORNs, mechanosensors):
  their synapses carry receptor-driven input we will supply from outside.

Sweep (LB3 sugar GRNs, 1 s, one trial; "active" = neurons with ≥ 1 spike out of 165,122):

| w_syn | STD | LB3 @100 Hz | LB3 @200 Hz |
|---|---|---|---|
| 0.275 | off | 20,889 active, MN9 102 Hz | 20,582 active |
| 0.146 | off | 199 active, MN9 0 | ignites at ~100 ms → 11,295 active |
| 0.146 | central, sensory exempt | 157 active, MN9 0 | 244 active, MN9 2 Hz |
| **0.20** | **central, sensory exempt** | 810 active, MN9 5 Hz | **926 active, MN9 12 Hz** |
| 0.275 | central, sensory exempt | 13,604 active | 14,651 active |

### What `scripts/validate_gustatory.py` checks (all pass)

- No input → **zero** spikes (the model has no spontaneous activity, as in the paper).
- LB3a/b/c @ 200 Hz → the paper's second-order neurons fire in the right order
  (GNG038 at 5.4 ms, Clavicle `ANXXX462a`, Quasimodo `GNG042`, GNG175, GNG229, DNg67),
  and **MN9 fires** (12 Hz on the left; the right MN9 stays silent — the LB3→MN9 path in
  this reconstruction is asymmetric, worth a look in the notebook).
- Activity stays local: 926 active neurons (0.6 %).

**Not reproduced:** the paper's prediction that silencing Clavicle reduces MN9 firing
(we get 12 → 11 Hz). At our gain, MN9 is barely above threshold and the Clavicle route is
one of several. We report this rather than tune for it.

### What `scripts/validate_looming.py` checks (passes)

LC4 + LPLC2 (looming-sensitive visual projection neurons, 311 cells) @ 100 Hz → the
**giant fiber `DNp01` spikes after 3.6 ms** and settles near 200 Hz, together with DNp04
and the PVLP interneurons — the escape circuit we will drive from the camera in M2.
2,034 neurons active (1.2 %).

## Honest limits

- Weights are synapse counts × one constant; no synapse is actually 0.2 mV.
- No neuromodulation, no plasticity, no gap junctions, no graded (non-spiking) neurons
  (many optic-lobe neurons are graded in reality).
- The calibration is ours, made on one circuit. Connecting the eyes (M2) did require
  revisiting it for the optic lobe — its intrinsic neurons are exempt from depression and
  their inhibitory synapses are ×8 in `for_male_cns` — while the central calibration above
  and both validation scripts are unchanged. See `docs/03-see-and-move.md` §3.
