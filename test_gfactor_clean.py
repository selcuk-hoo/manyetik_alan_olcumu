"""
Clean G-factor verification: uses A_r=1e-9 T, 20 turns, exact orbit.
Tests with quads OFF and ON. Linear regime (orbit not significantly perturbed).

Key physics: dSz/dt = (q/m)(G + 1/gamma) * B_r = 248.5e-9 rad/s for A_r=1e-9 T
"""
import numpy as np
import sys, os
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

A_r     = 1e-9  # 1 nT — small enough to not perturb orbit significantly
theory_dSzdt = (Q_E / M1) * (AMU + 1.0/gamma0) * A_r
T_arc = R0 * np.pi/24 / (beta0 * C)
T_rev = 2*np.pi*R0/(beta0*C) + 4.0*24*(2.0833+0.2)/(beta0*C)
theory_per_turn = theory_dSzdt * T_rev

print("=" * 60)
print("G-FACTOR VERIFICATION: dSz/dt = (q/m)(G+1/gamma)*B_r")
print("=" * 60)
print(f"  G_proton    = {AMU:.9f}")
print(f"  gamma0      = {gamma0:.7f}")
print(f"  G+1/gamma   = {AMU + 1.0/gamma0:.7f}")
print(f"  A_r         = {A_r:.0e} T")
print(f"  T_rev       = {T_rev*1e6:.4f} us")
print(f"  Theory dSz/dt = {theory_dSzdt:.6e} rad/s")
print(f"  Theory ΔSz/turn = {theory_per_turn:.6e}")
print()

def run_gfactor_test(label, g1, n_turns):
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
    fields.poincare_quad_index = -1.0   # 24 pts/turn (every arc)
    fields.B0rad_harm_amp = A_r
    fields.B0rad_harm_N   = 0.0

    p_mag = gamma0 * M1 * C * beta0
    # Exact on-orbit: dev0=0, y0=0, spin frozen (sz_local=-1)
    y0 = [0.0, 0.0, 0.0,
          0.0, 0.0, -p_mag,
          0.0, 0.0, -1.0]

    hist, poin, poin_t = integrate_particle(y0, 0.0, n_turns*T_rev, 1e-11,
                                             fields=fields, return_steps=1000)
    num_p = len(poin_t)
    if num_p == 0:
        print(f"  {label}: No Poincaré data!")
        return

    Sz = poin[:num_p, 7]   # global Sz
    Sy = poin[:num_p, 8]   # global Sy (tangential)
    turns = poin_t[:num_p] / T_rev

    slope_t, _ = np.polyfit(poin_t[:num_p], Sz, 1)
    ratio = slope_t / theory_dSzdt

    Sy_slope, _ = np.polyfit(poin_t[:num_p], Sy, 1)

    # Sz at each turn boundary (every 24 poincaré points)
    turn_Sz = []
    for k in range(n_turns):
        idx = k * 24
        if idx < num_p:
            turn_Sz.append(Sz[idx])

    print(f"  {label} ({n_turns} turns, {num_p} poincaré pts):")
    print(f"    Sy:   {Sy[0]:.8f} → {Sy[-1]:.8f}  (spin tune {Sy_slope:.3e} rad/s)")
    print(f"    Sz at turn boundaries: {', '.join(f'{x:.4e}' for x in turn_Sz[:min(5,len(turn_Sz))])}")
    print(f"    dSz/dt (fit) = {slope_t:.6e} rad/s")
    print(f"    Theory       = {theory_dSzdt:.6e} rad/s")
    print(f"    Ratio        = {ratio:.6f}  ← G-factor accuracy")
    print()

    return ratio

print("--- Test A: quads OFF (ideal ring) ---")
rA = run_gfactor_test("quads=OFF", g1=0.0, n_turns=20)

print("--- Test B: quads ON (G1=0.21 T/m, realistic FODO) ---")
rB = run_gfactor_test("quads=ON ", g1=0.21, n_turns=20)

print("=" * 60)
print(f"SUMMARY: G-factor ratios: quads_off={rA:.6f}, quads_on={rB:.6f}")
if rA is not None and rB is not None:
    if abs(rA - 1.0) < 0.001 and abs(rB - 1.0) < 0.01:
        print("  PASS: G-factor correctly implemented in T-BMT")
    else:
        print("  WARN: Ratio deviates significantly from 1.0")
print("=" * 60)
