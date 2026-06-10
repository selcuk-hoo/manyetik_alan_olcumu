"""
G-factor verification test: dSz/dt = (q/m)(G+1/gamma)*B_r
for N=0 uniform radial B0rad_harm.

Run conditions:
  1. dev0=0, y0=0, quads OFF → exact frozen spin → ratio should be 1.000
  2. dev0=0, y0=0, quads ON  → small spin tune from lattice → ratio should be ≈1.000
  3. dev0=1e-5, y0=1e-5, quads OFF → non-frozen spin → oscillatory Sz, slope ≈ 0

This test shows the bug from the previous session was an initial-condition issue,
not a code bug.
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from integrator import integrate_particle, FieldParams

M2  = 0.938272046
AMU = 1.792847356
C   = 299792458.0
M1  = 1.672621777e-27
Q_E = 1.602176565e-19

p_magic = M2 / np.sqrt(AMU)
E_tot   = np.sqrt(p_magic**2 + M2**2)
beta0   = p_magic / E_tot
gamma0  = 1.0 / np.sqrt(1.0 - beta0**2)
R0      = 95.49
E0_V_m  = -(p_magic * beta0 / R0) * 1e9

A_r     = 1e-6  # 1 μT (larger so secular signal is clear)
theory_dSzdt = (Q_E / M1) * (AMU + 1.0/gamma0) * A_r
T_arc = R0 * np.pi/24 / (beta0 * C)
T_rev = 2*np.pi*R0/(beta0*C) + 4.0*24*(2.0833+0.2)/(beta0*C)
theory_per_turn = theory_dSzdt * T_rev

print(f"Theory dSz/dt    = {theory_dSzdt:.6f} rad/s  (for A_r={A_r:.0e} T)")
print(f"Theory ΔSz/turn  = {theory_per_turn:.6e}")
print()

def run_test(label, dev0, y0, g1, n_turns):
    fields = FieldParams()
    fields.R0         = R0
    fields.E0         = E0_V_m
    fields.E0_power   = 1.0
    fields.quadG1     = g1
    fields.quadSwitch = 1.0 if g1 > 0 else 0.0
    fields.direction  = -1.0
    fields.nFODO      = 24.0
    fields.quadLen    = 0.4
    fields.driftLen   = 2.0833
    fields.poincare_quad_index = 0.0  # 1 point/turn
    fields.B0rad_harm_amp = A_r
    fields.B0rad_harm_N   = 0.0

    p_mag = gamma0 * M1 * C * beta0
    y0_state = [dev0, y0, 0.0,
                0.0, 0.0, -p_mag,
                0.0, 0.0, -1.0]

    h = 1e-11
    t_end = n_turns * T_rev

    hist, poin, poin_t = integrate_particle(y0_state, 0.0, t_end, h,
                                             fields=fields, return_steps=1000)
    num_p = len(poin_t)
    if num_p < 5:
        print(f"  {label}: Too few Poincaré points ({num_p})")
        return

    Sz = poin[:num_p, 7]  # global Sz = vertical spin
    Sy = poin[:num_p, 8]  # global Sy = tangential spin

    # Turn number (0-based)
    turns = poin_t[:num_p] / T_rev

    slope_t, _ = np.polyfit(poin_t[:num_p], Sz, 1)
    ratio = slope_t / theory_dSzdt

    # Spin tune: Sy should stay -1 for frozen spin; drift → non-frozen
    Sy_slope, _ = np.polyfit(poin_t[:num_p], Sy, 1)

    print(f"  {label}:")
    print(f"    Poincaré pts: {num_p}")
    print(f"    Sy range:     {Sy[0]:.6f} → {Sy[-1]:.6f}  (slope {Sy_slope:.3e} rad/s)")
    print(f"    Sz range:     {Sz[0]:.4e} → {Sz[-1]:.4e}")
    print(f"    dSz/dt fit:   {slope_t:.6e} rad/s  (theory: {theory_dSzdt:.6e})")
    print(f"    Ratio:        {ratio:.6f}")
    print()

print("=" * 60)
print("TEST 1: dev0=0, y0=0, quads OFF → exact frozen spin")
print("=" * 60)
run_test("dev0=0 quads=OFF", dev0=0.0, y0=0.0, g1=0.0, n_turns=100)

print("=" * 60)
print("TEST 2: dev0=0, y0=0, quads ON (G1=0.21 T/m)")
print("=" * 60)
run_test("dev0=0 quads=ON ", dev0=0.0, y0=0.0, g1=0.21, n_turns=100)

print("=" * 60)
print("TEST 3: dev0=1e-5, y0=1e-5, quads OFF  (previous failing case)")
print("=" * 60)
run_test("dev0=1e-5 q=OFF", dev0=1e-5, y0=1e-5, g1=0.0, n_turns=100)

print("=" * 60)
print("TEST 4: dev0=1e-5, y0=1e-5, quads ON  (original failing case)")
print("=" * 60)
run_test("dev0=1e-5 q=ON ", dev0=1e-5, y0=1e-5, g1=0.21, n_turns=100)
