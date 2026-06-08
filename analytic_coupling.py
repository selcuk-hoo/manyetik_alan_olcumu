#!/usr/bin/env python3
"""analytic_coupling.py — Analitik x-y kuplaj tahmini.

R_dy[i,j] = KL_sign_y(j) * G_y(s_i, s_j)
  QF (j çift): KL_sign = +KL_F   (QF y'de defokalize → +kick)
  QD (j tek) : KL_sign = -KL_D   (QD y'de fokalize → -kick)

G[i,i] = beta_i * cot(pi*Q) / 2

R_dx analogous for x (x'de: QF fokalize → -KL_F, QD defokalize → +KL_D)
"""
import json, math
import numpy as np
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

R_dy = np.load("R_dy_1.npy")
R_dx = np.load("R_dx_1.npy")

with open("params.json") as f:
    cfg = json.load(f)

N_Q   = 2 * int(cfg["nFODO"])   # 48
L_Q   = float(cfg["quadLen"])
G_F   = float(cfg["g1"])        # 0.21 T/m  QF
G_D   = float(cfg["g0"])        # 0.20 T/m  QD

M2  = 0.938272046; AMU = 1.792847356
p_magic = M2 / math.sqrt(AMU)
Brho = p_magic / 0.29979246   # T·m

KL_F = G_F * L_Q / Brho   # m^-1
KL_D = G_D * L_Q / Brho

# ── Diyagonalden beta_y çıkart ─────────────────────────────────────────────
# R_dy[j,j] = KL_sign_y(j) * beta_yj * cot(pi*Q_y) / 2
# KL_sign_y: even(QF) = +KL_F, odd(QD) = -KL_D
# Q_y = 1.760, Q_x = 1.967 (check_tunes.py sıfır-geçiş metoduyla doğrulandı)
Q_y = 1.760
cot_y = math.cos(math.pi * Q_y) / math.sin(math.pi * Q_y)

diag_dy = np.diag(R_dy)
KL_sign_y = np.where(np.arange(N_Q) % 2 == 0, +KL_F, -KL_D)

beta_y = diag_dy / (KL_sign_y * cot_y / 2)

print("=== Kafes parametreleri (R matrisinden) ===\n")
print(f"Brho = {Brho:.4f} T·m")
print(f"KL_F = {KL_F:.5f} m⁻¹   KL_D = {KL_D:.5f} m⁻¹")
print(f"cot(pi*Q_y) = {cot_y:.4f}  (Q_y={Q_y})\n")
print(f"beta_y: QF (çift) = {np.mean(beta_y[0::2]):.2f} m  "
      f"QD (tek) = {np.mean(beta_y[1::2]):.2f} m")

beta_y_QF = np.mean(beta_y[0::2])
beta_y_QD = np.mean(beta_y[1::2])

# ── Q_x ve beta_x çıkart ─────────────────────────────────────────────────────
# R_dx[j,j] = KL_sign_x(j) * beta_xj * cot(pi*Q_x) / 2
# x'de: QF fokalize → kick = -KL_F (karşı yönde)  → KL_sign_x(QF) = -KL_F
#        QD defokalize → kick = +KL_D              → KL_sign_x(QD) = +KL_D
KL_sign_x = np.where(np.arange(N_Q) % 2 == 0, -KL_F, +KL_D)

diag_dx = np.diag(R_dx)

# FODO simetrisi: beta_x(QF) ≈ beta_y(QD), beta_x(QD) ≈ beta_y(QF)
beta_x_QF_est = beta_y_QD   # FODO simetrisi tahmini
beta_x_QD_est = beta_y_QF

# Gözlemlenen cot(pi*Q_x) hesabı (QF ve QD ayrı ayrı):
# R_dx[QF,QF] = (-KL_F) * beta_x(QF) * cot(pi*Q_x) / 2
# Q_x: check_tunes.py sıfır-geçiş metoduyla doğrudan ölçüldü
Q_x = 1.967   # sıfır-geçiş: Q_x = 1.9674

print(f"\nQ_x = {Q_x:.4f}  (check_tunes.py sıfır-geçiş metoduyla ölçüldü)")

# beta_x: FODO simetrisi (β_x(QF) ≈ β_y(QD), β_x(QD) ≈ β_y(QF))
beta_x_QF = abs(beta_y_QD)
beta_x_QD = abs(beta_y_QF)
beta_x = np.where(np.arange(N_Q)%2==0, beta_x_QF, beta_x_QD)
print(f"\nbeta_x (FODO simetrisi): QF = {beta_x_QF:.2f} m   QD = {beta_x_QD:.2f} m")

DeltaQ = Q_x - Q_y
print(f"\nDeltaQ = Q_x - Q_y = {Q_x:.4f} - {Q_y:.4f} = {DeltaQ:.4f}")
print(f"|sin(pi*DeltaQ)| = {abs(math.sin(math.pi*DeltaQ)):.4f}")

# ── Kuplaj katsayısı (uniform tilt) ──────────────────────────────────────────
# sqrt(beta_x * beta_y): FODO simetrisi nedeniyle QF ve QD için aynı!
# sqrt(beta_x(QF)*beta_y(QF)) = sqrt(beta_y(QD)*beta_y(QF)) = sqrt(beta_x(QD)*beta_y(QD))
sqrt_bxby_QF = math.sqrt(beta_x_QF * beta_y_QF)
sqrt_bxby_QD = math.sqrt(beta_x_QD * beta_y_QD)
print(f"\nsqrt(beta_x*beta_y): QF={sqrt_bxby_QF:.2f} m   QD={sqrt_bxby_QD:.2f} m")

# Uniform faz tahmini: phi_j = j * 2*pi*Q/N_Q
phi_y_est = np.array([j * 2*math.pi*Q_y / N_Q for j in range(N_Q)])
phi_x_est = np.array([j * 2*math.pi*Q_x / N_Q for j in range(N_Q)])

# Faz farkı sumu
dphi = phi_x_est - phi_y_est   # = j * 2*pi*DeltaQ/N_Q

# Teorik: |sum exp(i*dphi_j)| = |sin(pi*DeltaQ)| / |sin(pi*DeltaQ/N_Q)|
sum_exp_theory = abs(math.sin(math.pi * DeltaQ)) / abs(math.sin(math.pi * DeltaQ / N_Q))
sum_exp_numeric = abs(np.sum(np.exp(1j * dphi)))
print(f"\n|Sigma_j exp(i*dphi_j)| teorik={sum_exp_theory:.2f}  numerik={sum_exp_numeric:.2f}")

print("\n=== Kuplaj tahmini ===\n")
print(f"{'theta[mrad]':>10s}  {'|C|/(4pi)':>10s}  {'A_y/A_x':>10s}")
print("-" * 36)

results_analytic = {}
for theta_mrad in [0, 1, 2, 5, 10]:
    if theta_mrad == 0:
        print(f"{'0':>10s}  {'—':>10s}  {'0.0000':>10s}")
        results_analytic[theta_mrad] = 0.0
        continue
    theta = theta_mrad * 1e-3

    # C = (1/4pi) * sum_j ks_j * sqrt(bxj*byj) * exp(i*(phixj-phiyj))
    ks_QF = 2 * theta * KL_F
    ks_QD = 2 * theta * KL_D

    C_re = 0.0; C_im = 0.0
    for j in range(N_Q):
        ks_j  = ks_QF if j % 2 == 0 else ks_QD
        sb    = sqrt_bxby_QF if j % 2 == 0 else sqrt_bxby_QD
        phase = dphi[j]
        C_re += ks_j * sb * math.cos(phase)
        C_im += ks_j * sb * math.sin(phase)

    C_mag_4pi = math.sqrt(C_re**2 + C_im**2) / (4 * math.pi)
    ratio = C_mag_4pi / abs(math.sin(math.pi * DeltaQ))
    print(f"{theta_mrad:>10.1f}  {C_mag_4pi:>10.5f}  {ratio:>10.4f}")
    results_analytic[theta_mrad] = ratio

print("\n=== verify_quad_tilt.py sonuçlarıyla karşılaştırma ===")
print("(verify_quad_tilt.py çalıştırılıyor...)\n")

import subprocess, sys
result = subprocess.run(
    [sys.executable, "verify_quad_tilt.py"],
    capture_output=True, text=True, cwd=os.getcwd()
)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[:500])

print("\n=== Karşılaştırma özeti ===\n")
print(f"{'theta[mrad]':>10s}  {'Analitik A_y/A_x':>18s}  {'Tracker y/x':>12s}  {'Oran':>8s}")
print("-" * 56)

# tracker çıktısından y/x oranlarını parse et
tracker_ratios = {}
for line in result.stdout.splitlines():
    try:
        parts = line.split()
        if len(parts) >= 4 and parts[0].replace('.','').replace('-','').isdigit():
            t_mrad = float(parts[0])
            ratio_t = float(parts[3])
            tracker_ratios[t_mrad] = ratio_t
    except Exception:
        pass

for t_mrad in [1, 2, 5, 10]:
    a_pred = results_analytic.get(t_mrad, float('nan'))
    t_meas = tracker_ratios.get(float(t_mrad), float('nan'))
    if not math.isnan(t_meas) and a_pred > 0:
        ratio_at = t_meas / a_pred
        print(f"{t_mrad:>10.1f}  {a_pred:>18.4f}  {t_meas:>12.4f}  {ratio_at:>8.3f}")
    else:
        print(f"{t_mrad:>10.1f}  {a_pred:>18.4f}  {'—':>12s}  {'—':>8s}")

print(f"""
Kafes özeti:
  Q_y = {Q_y},  Q_x = {Q_x:.3f},  DeltaQ = {DeltaQ:.3f}
  beta_y(QF) = {beta_y_QF:.1f} m,  beta_y(QD) = {beta_y_QD:.1f} m
  beta_x(QF) = {beta_x_QF:.1f} m,  beta_x(QD) = {beta_x_QD:.1f} m
  sqrt(beta_x*beta_y) ≈ {(sqrt_bxby_QF+sqrt_bxby_QD)/2:.1f} m
  KL_F = {KL_F:.5f} m⁻¹,  KL_D = {KL_D:.5f} m⁻¹

Tahmin formülü:
  |A_y/A_x| = |C| / |sin(pi*DeltaQ)|
  C = (1/4pi) * 2*theta * KL * sqrt(bx*by) * Sigma_j exp(i*j*2pi*DeltaQ/N_Q)
  "Sigma" büyütme: {sum_exp_theory:.1f}  (N_Q={N_Q}, DeltaQ={DeltaQ:.3f})
""")
