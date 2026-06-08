#!/usr/bin/env python3
"""check_tunes.py — Q_x ve Q_y sıfır-geçiş metoduyla, coupling tahmini."""
import json, math, sys, os
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
from integrator import integrate_particle, FieldParams

with open("params.json") as fh:
    cfg = json.load(fh)

M2=0.938272046; AMU=1.792847356; C_light=299792458.0; M1=1.672621777e-27
p_magic = M2/math.sqrt(AMU)
E_tot   = math.sqrt(p_magic**2 + M2**2)
beta0   = p_magic/E_tot
gamma0  = 1/math.sqrt(1-beta0**2)
R0      = cfg['R0']; direction = float(cfg.get('direction',-1))
p_mag   = gamma0*M1*C_light*beta0
E0_V_m  = -(p_magic*beta0/R0)*1e9
T_turn  = 2*math.pi*R0 / (beta0*C_light)
n_q     = 2*int(cfg['nFODO'])

print(f"beta0={beta0:.5f}  T_turn={T_turn*1e6:.4f} μs  1/T_turn={1/T_turn:.0f} Hz\n")

def run(theta_hor, theta_ver, tilt_val, T_END=3e-4, return_steps=6000):
    f = FieldParams()
    f.R0=R0; f.E0=E0_V_m; f.E0_power=cfg.get('E0_power',1.0)
    f.quadG1=float(cfg['g1']); f.quadG0=float(cfg['g1'])
    f.quadSwitch=1.0; f.sextSwitch=0.0; f.EDMSwitch=0.0
    f.direction=direction; f.nFODO=float(cfg['nFODO'])
    f.quadLen=float(cfg['quadLen']); f.driftLen=float(cfg['driftLen'])
    f.poincare_quad_index=-1.0; f.rfSwitch=0.0
    Px = p_mag * theta_hor
    Pz = p_mag * theta_ver
    Py = p_mag * direction * math.sqrt(max(0, 1-theta_hor**2-theta_ver**2))
    y0s=[0,0,0,Px,Pz,Py,0,0,direction]
    qt=np.full(n_q, tilt_val)
    hist,_,_ = integrate_particle(
        y0s,0,T_END,float(cfg['dt']),fields=f,return_steps=return_steps,
        quad_dy=np.zeros(n_q),quad_dx=np.zeros(n_q),
        dipole_tilt=np.zeros(n_q),quad_tilt=qt,quad_dG=np.zeros(n_q))
    return hist[:,0], hist[:,1]

def tune_from_zero_crossings(signal, T_END, label):
    zc = np.where(np.diff(np.sign(signal)))[0]
    if len(zc) < 4:
        print(f"  {label}: çok az sıfır-geçişi ({len(zc)})")
        return float('nan')
    dt_eff = T_END / len(signal)
    half_periods = np.diff(zc) * dt_eff
    hp_med = np.median(half_periods)
    T_bet = 2 * hp_med
    Q = T_turn / T_bet
    n_cycles = T_END / T_bet
    print(f"  {label}: {len(zc)} geçiş, medyan yarı-periyot={hp_med*1e6:.4f} μs, "
          f"T_bet={T_bet*1e6:.4f} μs, Q={Q:.4f}  ({n_cycles:.1f} tur/salınım)")
    return Q

# ── 1. Yatay kick → Q_x ──────────────────────────────────────────────────────
T1 = 5e-4
print(f"--- Yatay kick, tilt=0 (Q_x için, T={T1*1e3:.1f}ms) ---")
x_hor, z_hor = run(theta_hor=1e-3, theta_ver=0, tilt_val=0, T_END=T1, return_steps=10000)
Q_x = tune_from_zero_crossings(x_hor, T1, "x (yatay)")
tune_from_zero_crossings(z_hor, T1, "z (dikey)")

# ── 2. Dikey kick → Q_y ──────────────────────────────────────────────────────
print(f"\n--- Dikey kick, tilt=0 (Q_y için, T={T1*1e3:.1f}ms) ---")
x_ver, z_ver = run(theta_hor=0, theta_ver=1e-3, tilt_val=0, T_END=T1, return_steps=10000)
Q_y = tune_from_zero_crossings(z_ver, T1, "z (dikey)")
tune_from_zero_crossings(x_ver, T1, "x (yatay)")

print(f"\n{'='*50}")
print(f"SONUÇ: Q_x = {Q_x:.4f}   Q_y = {Q_y:.4f}")
print(f"       DeltaQ = Q_x - Q_y = {Q_x-Q_y:.4f}")

# ── 3. β_y'yi Q_y ile yeniden doğrula ───────────────────────────────────────
R_dy = np.load("R_dy_1.npy")
G_F=float(cfg['g1']); G_D=float(cfg['g0']); L_Q=float(cfg['quadLen'])
Brho = p_magic/0.29979246
KL_F=G_F*L_Q/Brho; KL_D=G_D*L_Q/Brho
cot_y = math.cos(math.pi*Q_y)/math.sin(math.pi*Q_y)
KL_sign_y = np.where(np.arange(n_q)%2==0, +KL_F, -KL_D)
diag_dy = np.diag(R_dy)
beta_y = diag_dy / (KL_sign_y * cot_y / 2)
print(f"\nbeta_y: QF(çift)={np.mean(beta_y[0::2]):.2f} m  QD(tek)={np.mean(beta_y[1::2]):.2f} m")
print(f"cot(pi*Q_y) = {cot_y:.5f}  (Q_y={Q_y:.4f})")

# ── 4. beta_x ve DeltaQ ile kuplaj tahmini ──────────────────────────────────
if not math.isnan(Q_x) and not math.isnan(Q_y):
    # β_x: FODO simetrisi (β_x_QF ≈ β_y_QD, β_x_QD ≈ β_y_QF)
    beta_y_QF = np.mean(beta_y[0::2])
    beta_y_QD = np.mean(beta_y[1::2])
    beta_x_QF = abs(beta_y_QD)   # FODO simetrisi
    beta_x_QD = abs(beta_y_QF)
    sqrt_bxby_QF = math.sqrt(abs(beta_x_QF * beta_y_QF))
    sqrt_bxby_QD = math.sqrt(abs(beta_x_QD * beta_y_QD))
    print(f"\nbeta_x (FODO): QF={beta_x_QF:.2f} m  QD={beta_x_QD:.2f} m")
    print(f"sqrt(bx*by):   QF={sqrt_bxby_QF:.2f} m  QD={sqrt_bxby_QD:.2f} m")

    DQ = Q_x - Q_y
    phi_y = np.array([j*2*math.pi*Q_y/n_q for j in range(n_q)])
    phi_x = np.array([j*2*math.pi*Q_x/n_q for j in range(n_q)])
    dphi = phi_x - phi_y

    print(f"\n{'='*50}")
    print(f"KUPLAJ TAHMİNİ (Q_x={Q_x:.4f}, Q_y={Q_y:.4f}, DQ={DQ:.4f})")
    print(f"|sin(pi*DQ)| = {abs(math.sin(math.pi*DQ)):.4f}")
    print(f"|Sigma exp(i*j*2pi*DQ/N)| = "
          f"{abs(math.sin(math.pi*DQ))/abs(math.sin(math.pi*DQ/n_q)):.2f}\n")
    print(f"{'theta[mrad]':>10s}  {'|C|/(4pi)':>10s}  {'A_y/A_x pred':>14s}  "
          f"{'Tracker y/x':>12s}  {'pred/meas':>10s}")
    print("-"*62)

    # Tracker y/x değerleri (analytic_coupling.py'den)
    tracker_vals = {1:0.0094, 2:0.0188, 5:0.0469, 10:0.0931}

    for theta_mrad in [1, 2, 5, 10]:
        theta = theta_mrad*1e-3
        ks_QF=2*theta*KL_F; ks_QD=2*theta*KL_D
        Cre=sum((ks_QF if j%2==0 else ks_QD)*(sqrt_bxby_QF if j%2==0 else sqrt_bxby_QD)
                *math.cos(dphi[j]) for j in range(n_q))
        Cim=sum((ks_QF if j%2==0 else ks_QD)*(sqrt_bxby_QF if j%2==0 else sqrt_bxby_QD)
                *math.sin(dphi[j]) for j in range(n_q))
        C_mag = math.sqrt(Cre**2+Cim**2)/(4*math.pi)
        pred = C_mag / abs(math.sin(math.pi*DQ))
        meas = tracker_vals.get(theta_mrad, float('nan'))
        ratio = pred/meas if not math.isnan(meas) and meas>0 else float('nan')
        print(f"{theta_mrad:>10.1f}  {C_mag:>10.5f}  {pred:>14.4f}  "
              f"{meas:>12.4f}  {ratio:>10.3f}")
