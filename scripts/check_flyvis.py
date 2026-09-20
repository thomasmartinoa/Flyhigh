# scripts/check_flyvis.py
"""Day-one check: load the pretrained flyvis model and run 100 ms of grey on the GPU."""
import time

import flyvis
import torch

view = flyvis.NetworkView(flyvis.results_dir / "flow/0000/000")
net = view.init_network().to(flyvis.device).eval()
nodes = net.connectome.nodes
print(f"flyvis device={flyvis.device}  neurons={len(nodes.type[:])}  types={len(net.connectome.unique_cell_types[:])}")
print("input cell types:", [t.decode() for t in net.connectome.input_cell_types[:]])
print("hexals per eye:", net.stimulus.n_input_elements)
t = time.perf_counter()
state = net.steady_state(t_pre=1.0, dt=0.01, batch_size=1)
torch.cuda.synchronize()
print(f"steady state after 1 s grey: {time.perf_counter() - t:.2f} s, mean activity {state.nodes.activity.mean():.4f}")
