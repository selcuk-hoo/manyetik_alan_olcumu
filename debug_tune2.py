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

direction = config.get("direction", -1)
gamma0 = 1.0 / np.sqrt(1.0 - (p_magic_base/np.sqrt(p_magic_base**2+0.938272046**2))**2)
p_mag = gamma0 * 1.672621777e-27 * 299792458.0 * (p_magic_base/np.sqrt(p_magic_base**2+0.938272046**2))

theta0_hor = 1e-3
theta0_ver = 0.0

p0_x = p_mag * np.sin(theta0_hor) * np.cos(theta0_ver)
p0_y = p_mag * np.sin(theta0_ver)
p0_z = p_mag * np.cos(theta0_hor) * np.cos(theta0_ver) * direction

y0 = [0.0, 0.0, 0.0, p0_x, p0_y, p0_z, 0.0, 0.0, direction]

T_END = 5e-4
hist, poin_local, _ = integrate_particle(y0, 0, T_END, 1e-11, fields=alanlar, return_steps=10000)

x_hist = hist[:, 0]
T_turn = 2 * np.pi * alanlar.R0 / (0.59838 * 299792458.0) # roughly

win = np.hanning(len(x_hist))
spec = np.abs(np.fft.rfft((x_hist - x_hist.mean()) * win))
df = 1.0 / T_END
pk = int(np.argmax(spec[1:])) + 1
f_beta = pk * df
print(f"FFT Q: {f_beta * T_turn:.5f}")

x = poin_local[:, 0]
px = poin_local[:, 3]
ps = poin_local[:, 5]
xp = px / ps

xc = x - x.mean()
xpc = xp - xp.mean()

phi = np.arctan2(xpc, xc)
dphi = np.diff(np.unwrap(phi))
mean_dphi = np.mean(dphi)
print(f"Poincare Q: {alanlar.nFODO * abs(mean_dphi) / (2 * np.pi):.5f}")

