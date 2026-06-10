"""
Single-arc diagnostic: does the C++ T-BMT give dSz/dt = (G+1/gamma)*A_r for A_r=1e-9 T?

Run for just 2 turns with dense history (return_steps=100000) to see
if Sz actually changes within the arc, and by how much.
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from integrator import integrate_particle, FieldParams

M2    = 0.938272046
AMU   = 1.792847356
C     = 299792458.0
M1    = 1.672621777e-27
Q_E   = 1.602176565e-19

p_magic = M2 / np.sqrt(AMU)
E_tot   = np.sqrt(p_magic**2 + M2**2)
beta0   = p_magic / E_tot
gamma0  = 1.0 / np.sqrt(1.0 - beta0**2)
R0      = 95.49

E0_V_m = -(p_magic * beta0 / R0) * 1e9

# Theory
A_r = 1e-9
theory_dSzdt = (Q_E / M1) * (AMU + 1.0/gamma0) * A_r
T_rev = 2.0 * np.pi * R0 / (beta0 * C) + 4.0 * 24.0 * (2.0833 + 0.2) / (beta0 * C)
print(f"Theory dSz/dt       = {theory_dSzdt:.6f} rad/s (per unit spin)")
print(f"T_rev               = {T_rev*1e6:.4f} us")
print(f"Theory ΔSz/turn     = {theory_dSzdt*T_rev:.4e}")
print()

# Initial conditions: on-orbit, perfect frozen spin
fields = FieldParams()
fields.R0         = R0
fields.E0         = E0_V_m
fields.E0_power   = 1.0
fields.quadG1     = 0.0        # quads OFF
fields.quadSwitch = 0.0
fields.direction  = -1.0
fields.nFODO      = 24.0
fields.quadLen    = 0.4
fields.driftLen   = 2.0833
fields.poincare_quad_index = -1.0  # every arc (24/turn) for dense poincaré
fields.B0rad_harm_amp = A_r
fields.B0rad_harm_N   = 0.0

# Exact on-orbit initial condition
p_mag = gamma0 * M1 * C * beta0
# Local: x=0, y=0, z=0, px=0, py=0, pz=p_mag*dir=p_mag*(-1)
# Spin frozen: sx=0, sy=0, sz=-1 (longitudinal in local → Sy_global=-1, Sz_global=0)
y0 = [0.0, 0.0, 0.0,
      0.0, 0.0, -p_mag,
      0.0, 0.0, -1.0]

h    = 1e-11   # 10 ps step
t0   = 0.0
t_end = 5 * T_rev  # 5 turns

print(f"Running {t_end/T_rev:.0f} turns with poincaré at every arc...")
return_steps = 5000

hist, poin, poin_t = integrate_particle(y0, t0, t_end, h, fields=fields,
                                         return_steps=return_steps)

num_p = len(poin_t)
print(f"Poincaré points: {num_p} (expected {5*24}={5*24})")

if num_p == 0:
    print("ERROR: No Poincaré points recorded!")
    sys.exit(1)

# poin[:,7] = vertical spin (global Sz = state[8])
# poin[:,8] = tangential spin (global Sy = state[7])
Sz_poin = poin[:, 7]
Sy_poin = poin[:, 8]

print("\nPoincaré Sz (vertical spin) first 50 points:")
for i in range(min(50, num_p)):
    turn_frac = poin_t[i] / T_rev
    print(f"  pt {i:3d} (t={poin_t[i]*1e6:.4f}us, turn~{turn_frac:.2f}): "
          f"Sy={Sy_poin[i]:.8f}, Sz={Sz_poin[i]:.4e}")

# Check Sz at turn boundaries (every 24 points)
print("\nSz at each full turn boundary:")
pts_per_turn = 24
for turn in range(min(6, num_p // pts_per_turn)):
    idx = turn * pts_per_turn
    print(f"  Turn {turn}: Sz = {Sz_poin[idx]:.6e}")

# Linear fit over all Poincaré points
if num_p > 5:
    slope_t, intercept = np.polyfit(poin_t[:num_p], Sz_poin[:num_p], 1)
    print(f"\nLinear fit (vs time):")
    print(f"  dSz/dt slope = {slope_t:.6e} rad/s")
    print(f"  Theory dSz/dt = {theory_dSzdt:.6e} rad/s")
    print(f"  Ratio = {slope_t/theory_dSzdt:.6f}")

    # Also fit per turn
    turns_arr = poin_t[:num_p] / T_rev
    slope_n, _ = np.polyfit(turns_arr, Sz_poin[:num_p], 1)
    print(f"\nLinear fit (vs turns):")
    print(f"  ΔSz/turn = {slope_n:.6e}")
    print(f"  Theory ΔSz/turn = {theory_dSzdt*T_rev:.6e}")
    print(f"  Ratio = {slope_n/(theory_dSzdt*T_rev):.6f}")

# Also check dense history Sz
print("\nDense history Sz (first 20 points):")
for i in range(min(20, len(hist))):
    print(f"  step {i:4d}: Sy={hist[i,8]:.8f}, Sz={hist[i,7]:.4e}")
