"""
Full Python T-BMT simulation with E-field AND B0rad_harm.
Tests whether secular Sz growth from B0rad_harm (N=0) occurs.

Analytic prediction: dSz/dt = (Q_E/M_P)*(G+1/gamma)*A_r = 248.5 rad/s (per unit spin)
For A_r=1e-9 T: ΔSz/turn = 248.5e-9 * T_rev

The key question: does frozen spin (E-field T-BMT) allow secular Sz growth?
"""
import numpy as np

# Physical constants
C_LIGHT = 299792458.0
M_P     = 1.672621777e-27
Q_E     = 1.602176565e-19
G_P     = 1.792847356
M_P_GEV = 0.938272046

# Ring parameters
R0      = 95.49
nFODO   = 24
driftLen = 2.0833
quadLen  = 0.4
direction = -1   # CW

p_magic = M_P_GEV / np.sqrt(G_P)  # GeV/c
E_tot_gev = np.sqrt(p_magic**2 + M_P_GEV**2)
beta0   = p_magic / E_tot_gev
gamma0  = 1.0 / np.sqrt(1.0 - beta0**2)

# E0 (centripetal): same formula as run_simulation.py
E0_V_m = -(p_magic * beta0 / R0) * 1e9  # V/m, negative = inward

print(f"Magic momentum:  {p_magic:.6f} GeV/c")
print(f"beta0={beta0:.6f}, gamma0={gamma0:.6f}")
print(f"E0 = {E0_V_m/1e6:.4f} MV/m")
print(f"G*(gamma^2-1) = {G_P*(gamma0**2-1):.12f}  (should be 1.0)")

# Convert to SI momentum
p_SI = p_magic * 1e9 * Q_E / C_LIGHT  # kg*m/s
v_mag = beta0 * C_LIGHT

# Arc geometry
Phi_def = np.pi / nFODO  # arc angle per half-cell [rad] = π/24

T_arc = R0 * Phi_def / v_mag  # time per arc [s]
T_drift = driftLen / v_mag
T_quad  = quadLen / v_mag
T_rev   = 2 * np.pi * R0 / v_mag + 2 * nFODO * (2 * driftLen + quadLen) / v_mag
print(f"Phi_def = {np.degrees(Phi_def):.3f} deg")
print(f"T_arc = {T_arc*1e9:.3f} ns, T_rev = {T_rev*1e6:.3f} us")

# B0rad_harm
A_r = 1e-9  # T, N=0 (uniform radial)
theory_dSzdt = (Q_E / M_P) * (G_P + 1.0/gamma0) * A_r
theory_delta_per_turn = theory_dSzdt * T_rev
print(f"\nTheory dSz/dt = {theory_dSzdt:.4f} rad/s")
print(f"Theory ΔSz/turn = {theory_delta_per_turn:.4e}")
print()

def tmbt_rhs(S, pos, vel, E0_field, Br_harm_amp, element_type, phi_local):
    """
    Compute dS/dt = Omega × S using T-BMT.
    pos, vel in GLOBAL Cartesian (X, Y, Z).
    Returns (dSx, dSy, dSz).
    """
    X, Y, Z = pos
    vx, vy, vz = vel

    v_sq = vx**2 + vy**2 + vz**2
    beta = np.array([vx, vy, vz]) / C_LIGHT

    p_sq = p_SI**2  # we're on magic momentum, fixed
    gam = gamma0    # fixed (no RF)

    # Fields
    R = np.sqrt(X**2 + Y**2)

    E = np.zeros(3)
    B = np.zeros(3)

    if element_type == 0 and R > 1e-6:
        # Arc: cylindrical capacitor
        cos_th = X / R
        sin_th = Y / R
        E_r = E0_field * R0 / R
        E[0] = E_r * cos_th
        E[1] = E_r * sin_th

        # B0rad_harm: N=0 → B_r = A_r everywhere
        phi = np.arctan2(Y, X)
        B_r0 = Br_harm_amp  # cos(0*phi) = 1
        B[0] += B_r0 * np.cos(phi)
        B[1] += B_r0 * np.sin(phi)
    else:
        # Straight: B0rad_harm gives B_x = A_r (radial in rotating frame)
        B[0] += Br_harm_amp

    # T-BMT
    AMU = G_P
    beta_dot_B = np.dot(beta, B)
    beta_dot_E = np.dot(beta, E)
    beta_cross_E = np.cross(beta, E)

    Omega = np.zeros(3)
    for i in range(3):
        mdm_term = (B[i] * (AMU + 1.0/gam)
                    - beta[i] * beta_dot_B * AMU * gam / (gam + 1.0)
                    - beta_cross_E[i] * (AMU + 1.0/(gam + 1.0)) / C_LIGHT)
        Omega[i] = -(mdm_term) * (Q_E / M_P)

    dSdt = np.cross(Omega, S)
    return dSdt

def integrate_arc_euler(S0, global_azimuth, n_steps, E0_field, Br_harm_amp):
    """
    Integrate spin through one arc of angle Phi_def, starting at global_azimuth.
    CW ring: azimuth decreases from global_azimuth to global_azimuth - Phi_def.
    Uses simple Euler integration with small steps.
    Returns (S_end, ΔSz)
    """
    S = S0.copy()
    dt = T_arc / n_steps

    for i in range(n_steps):
        # Current azimuth within this arc
        frac = (i + 0.5) / n_steps
        phi = global_azimuth - frac * Phi_def  # CW: decreasing

        # Position and velocity in global frame
        pos = np.array([R0 * np.cos(phi), R0 * np.sin(phi), 0.0])
        vel = v_mag * np.array([-np.sin(phi), -np.cos(phi), 0.0])  # CW tangential

        dSdt = tmbt_rhs(S, pos, vel, E0_field, Br_harm_amp, 0, phi)
        S = S + dSdt * dt
        # Renormalize
        S /= np.linalg.norm(S)

    return S

def integrate_straight_euler(S0, global_azimuth, length, n_steps, E0_field, Br_harm_amp):
    """Integrate through a straight element at constant global azimuth."""
    S = S0.copy()
    dt = length / v_mag / n_steps

    phi = global_azimuth  # azimuth doesn't change in straight
    pos = np.array([R0 * np.cos(phi), R0 * np.sin(phi), 0.0])
    vel = v_mag * np.array([-np.sin(phi), -np.cos(phi), 0.0])  # CW

    for i in range(n_steps):
        dSdt = tmbt_rhs(S, pos, vel, E0_field, Br_harm_amp, 1, phi)
        S = S + dSdt * dt
        S /= np.linalg.norm(S)

    return S

# ============================================================
# SIMULATION
# ============================================================
print("=" * 60)
print("CASE 1: E-field ON, B0rad_harm ON")
print("=" * 60)

N_TURNS = 20
n_arc_steps = 200  # steps per arc integration
n_str_steps = 50   # steps per straight integration

# Initial spin: frozen = tangential CW at azimuth 0
# S = (sin(0), -cos(0), 0) = (0, -1, 0) in global
S = np.array([0.0, -1.0, 0.0])

# In the Python model, the cell order is:
# Cell k: DRIFT, QF, DRIFT, ARC2, DRIFT, QD, DRIFT, ARC1
# ARC1 of cell k is at global azimuth: -(2*k + 1) * Phi_def  [after ARC2 of cell k]
# ARC2 of cell k is at global azimuth: -2*k * Phi_def        [after DRIFT of cell k]
#
# Wait — following C++ order with start_elem=1:
# Cell k processes: 1(DRIFT),2(QF),3(DRIFT),4(ARC2),5(DRIFT),6(QD),7(DRIFT),0(ARC1)
#
# The global azimuth at the START of cell k is after ARC1 of cell k-1:
# After ARC1 of cell (k-1), the azimuth has advanced by 2*(k-1+1)*Phi_def = 2k*Phi_def total.
# But since the frame was reset by rotate_all after each arc, in the rotating frame
# everything starts from (R0, 0). The GLOBAL azimuth at each element entry is
# tracked by theta_e = current_fodo * 2 * Phi_def + arcs_done * Phi_def.
#
# Let's use the simple global tracking:
# arc_index goes 0, 1, ..., 47 per turn (2 arcs per cell × 24 cells)
# arc 0: ARC2 of cell 0, global_azimuth = -0 * Phi_def = 0 (first element processed in cell 0 after the initial position)
# But actually in C++ the first arc processed is ARC2 of cell 0 (elem=4, second arc of cell 0).
#
# Let's just track azimuth: start at 0, each arc decrements by Phi_def.
# Each drift/quad is at the current azimuth (no angular advance).

global_azimuth = 0.0  # start at phi=0
Sz_values = []

for turn in range(N_TURNS):
    Sz_before = S[2]

    for cell in range(nFODO):
        # Cell element order: DRIFT, QF, DRIFT, ARC2, DRIFT, QD, DRIFT, ARC1

        # DRIFT1
        S = integrate_straight_euler(S, global_azimuth, driftLen, n_str_steps, E0_V_m, A_r)

        # QF (treat as drift for spin purposes — at reference orbit B_quad=0)
        S = integrate_straight_euler(S, global_azimuth, quadLen, n_str_steps, E0_V_m, A_r)

        # DRIFT2
        S = integrate_straight_euler(S, global_azimuth, driftLen, n_str_steps, E0_V_m, A_r)

        # ARC2
        S = integrate_arc_euler(S, global_azimuth, n_arc_steps, E0_V_m, A_r)
        global_azimuth -= Phi_def  # CW: azimuth decreases

        # DRIFT3
        S = integrate_straight_euler(S, global_azimuth, driftLen, n_str_steps, E0_V_m, A_r)

        # QD (treat as drift)
        S = integrate_straight_euler(S, global_azimuth, quadLen, n_str_steps, E0_V_m, A_r)

        # DRIFT4
        S = integrate_straight_euler(S, global_azimuth, driftLen, n_str_steps, E0_V_m, A_r)

        # ARC1
        S = integrate_arc_euler(S, global_azimuth, n_arc_steps, E0_V_m, A_r)
        global_azimuth -= Phi_def  # CW: azimuth decreases

    Sz_after = S[2]
    Sz_values.append(Sz_after)
    if turn < 5 or turn == N_TURNS - 1:
        print(f"Turn {turn+1:3d}: Sx={S[0]:.6f}, Sy={S[1]:.6f}, Sz={Sz_after:.6e}, "
              f"ΔSz={Sz_after - Sz_before:.3e}")

Sz_arr = np.array(Sz_values)
turns = np.arange(1, N_TURNS + 1)
if N_TURNS > 2:
    slope, intercept = np.polyfit(turns, Sz_arr, 1)
    print(f"\nLinear fit slope: dSz/turn = {slope:.4e}")
    print(f"Theory dSz/turn  = {theory_delta_per_turn:.4e}")
    print(f"Ratio sim/theory = {slope / theory_delta_per_turn:.6f}")

print()
print("=" * 60)
print("CASE 2: E-field OFF, B0rad_harm ON (should see spin not frozen)")
print("=" * 60)

S2 = np.array([0.0, -1.0, 0.0])
global_azimuth2 = 0.0
Sz_values2 = []

for turn in range(5):
    Sz_before = S2[2]
    for cell in range(nFODO):
        S2 = integrate_straight_euler(S2, global_azimuth2, driftLen, n_str_steps, 0.0, A_r)
        S2 = integrate_straight_euler(S2, global_azimuth2, quadLen, n_str_steps, 0.0, A_r)
        S2 = integrate_straight_euler(S2, global_azimuth2, driftLen, n_str_steps, 0.0, A_r)
        S2 = integrate_arc_euler(S2, global_azimuth2, n_arc_steps, 0.0, A_r)
        global_azimuth2 -= Phi_def
        S2 = integrate_straight_euler(S2, global_azimuth2, driftLen, n_str_steps, 0.0, A_r)
        S2 = integrate_straight_euler(S2, global_azimuth2, quadLen, n_str_steps, 0.0, A_r)
        S2 = integrate_straight_euler(S2, global_azimuth2, driftLen, n_str_steps, 0.0, A_r)
        S2 = integrate_arc_euler(S2, global_azimuth2, n_arc_steps, 0.0, A_r)
        global_azimuth2 -= Phi_def
    print(f"Turn {turn+1}: Sx={S2[0]:.6f}, Sy={S2[1]:.6f}, Sz={S2[2]:.6e}")
