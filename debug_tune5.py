import numpy as np
import os
import json
from integrator import integrate_particle, FieldParams

config = json.load(open("params.json"))

alanlar = FieldParams()
alanlar.R0 = config["R0"]
p_magic_base = 0.938272046 / np.sqrt(1.792847356)
alanlar.E0 = -(p_magic_base * (p_magic_base / np.sqrt(p_magic_base**2 + 0.938272046**2)) / alanlar.R0) * 1e9
alanlar.quadG1 = config["g1"]
alanlar.quadG0 = config["g1"]
alanlar.nFODO = config["nFODO"]
alanlar.quadLen = config["quadLen"]
alanlar.driftLen = config["driftLen"]
alanlar.poincare_quad_index = -1.0
alanlar.quadSwitch = 1.0

direction = config.get("direction", -1)
gamma0 = 1.0 / np.sqrt(1.0 - (p_magic_base/np.sqrt(p_magic_base**2+0.938272046**2))**2)
p_mag = gamma0 * 1.672621777e-27 * 299792458.0 * (p_magic_base/np.sqrt(p_magic_base**2+0.938272046**2))

y0 = [0.0, 0.0, 0.0, p_mag*np.sin(1e-3), 0.0, p_mag*np.cos(1e-3)*direction, 0.0, 0.0, direction]

hist, poin_local, _ = integrate_particle(y0, 0, 5e-4, 1e-11, fields=alanlar, return_steps=1000)

x_pc = poin_local[:, 0] * 1000
pz_pc = poin_local[:, 5]
xp_pc = (poin_local[:, 3] / pz_pc) * 1000

uc = x_pc - x_pc.mean()
upc = xp_pc - xp_pc.mean()

phi = np.arctan2(upc, uc)
dphi = np.diff(np.unwrap(phi))

print("First 20 dphi (degrees):")
print(np.degrees(dphi[:20]))

beta = np.var(uc)
alpha = -np.cov(uc, upc)[0, 1]
U = uc / np.sqrt(beta)
UP = (uc * alpha + upc * beta) / np.sqrt(beta)

phi_norm = np.arctan2(UP, U)
dphi_norm = np.diff(np.unwrap(phi_norm))

print("First 20 normalized dphi (degrees):")
print(np.degrees(dphi_norm[:20]))

print(f"Mean raw dphi (deg): {np.degrees(np.mean(dphi)):.2f}")
print(f"Mean norm dphi (deg): {np.degrees(np.mean(dphi_norm)):.2f}")

