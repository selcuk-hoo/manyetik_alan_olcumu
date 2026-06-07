import numpy as np
import json
from paper_plot_utils import load_R, Fcos
from build_response_matrix import setup_fields, run_sim, read_cod_quads

with open("params.json") as f: config = json.load(f)
R = load_R()
n_q = 48
F2 = Fcos(2, n_q)
quad_dy = 10e-6 * F2

alanlar, state0 = setup_fields(config)
config["t2"] = 2e-3
run_sim(alanlar, state0, config, quad_dy=quad_dy, quad_dx=np.zeros(n_q), quad_dG=np.zeros(n_q))
_, y_true = read_cod_quads(48)

Mc = R @ F2
Mc_norm = np.linalg.norm(Mc)
m_hat = Mc / Mc_norm

A_fit = np.sum(y_true * m_hat) / Mc_norm
print(f"A_fit: {A_fit*1e6:.3f} um")
print(f"y_true norm: {np.linalg.norm(y_true)*1e6:.3f}")
print(f"Mc norm: {Mc_norm*1e6:.3f} (scaled to 10um: {Mc_norm*10:.3f})")
