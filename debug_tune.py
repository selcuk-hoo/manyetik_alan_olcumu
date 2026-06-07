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
alanlar.nFODO = config["nFODO"]
alanlar.quadLen = config["quadLen"]
alanlar.driftLen = config["driftLen"]
alanlar.poincare_quad_index = -1.0
alanlar.quadSwitch = 1.0

direction = -1
gamma0 = 1.0 / np.sqrt(1.0 - (p_magic_base/np.sqrt(p_magic_base**2+0.938272046**2))**2)
p_mag = gamma0 * 1.672621777e-27 * 299792458.0 * (p_magic_base/np.sqrt(p_magic_base**2+0.938272046**2))

y0 = [1e-3, 0.0, 0.0, 0.0, p_mag*direction, 0.0, 0.0, 0.0, direction]

_, poin_local, _ = integrate_particle(y0, 0, 5e-4, 1e-11, fields=alanlar, return_steps=1000)

x = poin_local[:, 0]
px = poin_local[:, 3]
ps = poin_local[:, 4]
xp = px / ps

xc = x - x.mean()
xpc = xp - xp.mean()
phi = np.arctan2(xpc, xc)
dphi = np.diff(np.unwrap(phi))

print(f"Mean dphi: {np.mean(dphi):.5f}")
print(f"Q: {alanlar.nFODO * abs(np.mean(dphi)) / (2 * np.pi):.5f}")

