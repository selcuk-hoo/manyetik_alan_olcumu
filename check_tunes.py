#!/usr/bin/env python3
"""check_tunes.py — Q_x ve Q_y: sıfır-geçiş + FFT, kuplaj tahmini.

Düzeltmeler:
  • İntegraratörde QF: +quadG1, QD: -quadG1 (aynı büyüklük).
    quadG0 yalnızca modülasyonlu QUAD_F_MOD içindir.
    Dolayısıyla KL_F = KL_D = g1*L/Brho.
  • Q tahmininde Hanning-pencereli FFT eklendi (takma-ad yok).
"""
import json, math, sys, os
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
from integrator import integrate_particle, FieldParams

with open("params.json") as fh:
    cfg = json.load(fh)

M2 = 0.938272046; AMU = 1.792847356; C_light = 299792458.0; M1 = 1.672621777e-27
p_magic = M2 / math.sqrt(AMU)
E_tot   = math.sqrt(p_magic**2 + M2**2)
beta0   = p_magic / E_tot
gamma0  = 1 / math.sqrt(1 - beta0**2)
R0      = cfg['R0']; direction = float(cfg.get('direction', -1))
p_mag   = gamma0 * M1 * C_light * beta0
E0_V_m  = -(p_magic * beta0 / R0) * 1e9
T_turn  = 2 * math.pi * R0 / (beta0 * C_light)
n_q     = 2 * int(cfg['nFODO'])

print(f"beta0={beta0:.5f}  T_turn={T_turn*1e6:.5f} μs  1/T_turn={1/T_turn:.0f} Hz\n")


def run(theta_hor, theta_ver, tilt_val, T_END=5e-4, return_steps=10000):
    f = FieldParams()
    f.R0 = R0; f.E0 = E0_V_m; f.E0_power = cfg.get('E0_power', 1.0)
    f.quadG1 = float(cfg['g1']); f.quadG0 = float(cfg['g1'])
    f.quadSwitch = 1.0; f.sextSwitch = 0.0; f.EDMSwitch = 0.0
    f.direction = direction; f.nFODO = float(cfg['nFODO'])
    f.quadLen = float(cfg['quadLen']); f.driftLen = float(cfg['driftLen'])
    f.poincare_quad_index = -1.0; f.rfSwitch = 0.0
    Px = p_mag * theta_hor
    Pz = p_mag * theta_ver
    Py = p_mag * direction * math.sqrt(max(0, 1 - theta_hor**2 - theta_ver**2))
    y0s = [0, 0, 0, Px, Pz, Py, 0, 0, direction]
    qt = np.full(n_q, tilt_val)
    hist, _, _ = integrate_particle(
        y0s, 0, T_END, float(cfg['dt']), fields=f, return_steps=return_steps,
        quad_dy=np.zeros(n_q), quad_dx=np.zeros(n_q),
        dipole_tilt=np.zeros(n_q), quad_tilt=qt, quad_dG=np.zeros(n_q))
    return hist[:, 0], hist[:, 1]


def tune_from_zero_crossings(signal, T_END, label):
    zc = np.where(np.diff(np.sign(signal)))[0]
    if len(zc) < 4:
        print(f"  {label}: çok az sıfır-geçişi ({len(zc)})")
        return float('nan')
    dt_eff = T_END / len(signal)
    hp_med = np.median(np.diff(zc) * dt_eff)
    T_bet  = 2 * hp_med
    Q      = T_turn / T_bet
    print(f"  {label}: {len(zc):4d} geçiş  T_half={hp_med*1e6:.4f} μs  Q_ZC ={Q:.5f}")
    return Q


def tune_from_fft(signal, T_END, label):
    """Hanning-pencereli FFT + parabolik interpolasyon → Q.
    Sürekli zaman serisi kullanır: takma-ad riski yok.
    ZC'nin kuantizasyon yanlılığına (T_half tamsayı dt_eff katına yuvarlanması) karşı sağlam."""
    n    = len(signal)
    win  = np.hanning(n)
    spec = np.abs(np.fft.rfft((signal - signal.mean()) * win))
    df   = 1.0 / T_END                          # Hz/bin
    pk   = int(np.argmax(spec[1:])) + 1          # DC'yi atla
    # Parabolik interpolasyon: alt-bin hassasiyeti
    if 1 < pk < len(spec) - 1:
        d = spec[pk-1] - 2*spec[pk] + spec[pk+1]
        pk_frac = pk + 0.5*(spec[pk-1] - spec[pk+1]) / d if abs(d) > 0 else pk
    else:
        pk_frac = float(pk)
    f_beta = pk_frac * df
    Q      = f_beta * T_turn
    print(f"  {label}: f_beta={f_beta:9.1f} Hz  Q_FFT={Q:.5f}")
    return Q


# ── 1. Yatay kick → Q_x ─────────────────────────────────────────────────────
T1 = 5e-4
print(f"--- Yatay kick, tilt=0  (T={T1*1e3:.1f} ms ≈ {T1/T_turn:.0f} tur) ---")
x_hor, z_hor = run(theta_hor=1e-3, theta_ver=0, tilt_val=0, T_END=T1)
Q_x_zc  = tune_from_zero_crossings(x_hor, T1, "x (yatay) ZC ")
Q_x_fft = tune_from_fft(x_hor,            T1, "x (yatay) FFT")
tune_from_zero_crossings(z_hor, T1, "z (dikey)  ZC ")
tune_from_fft(z_hor,            T1, "z (dikey)  FFT")

# ── 2. Dikey kick → Q_y ─────────────────────────────────────────────────────
print(f"\n--- Dikey kick, tilt=0  (T={T1*1e3:.1f} ms) ---")
x_ver, z_ver = run(theta_hor=0, theta_ver=1e-3, tilt_val=0, T_END=T1)
Q_y_zc  = tune_from_zero_crossings(z_ver, T1, "z (dikey)  ZC ")
Q_y_fft = tune_from_fft(z_ver,            T1, "z (dikey)  FFT")
tune_from_zero_crossings(x_ver, T1, "x (yatay) ZC ")
tune_from_fft(x_ver,            T1, "x (yatay) FFT")

# FFT en güvenilir tahmin
Q_x = Q_x_fft
Q_y = Q_y_fft

print(f"\n{'='*55}")
print(f"SONUÇ  ZC : Q_x={Q_x_zc:.5f}  Q_y={Q_y_zc:.5f}  ΔQ={Q_x_zc-Q_y_zc:.5f}")
print(f"SONUÇ  FFT: Q_x={Q_x_fft:.5f}  Q_y={Q_y_fft:.5f}  ΔQ={Q_x_fft-Q_y_fft:.5f}")
# makine tarafından okunabilir satır (analytic_coupling.py tarafından parse edilir)
print(f"Q_FFT_x={Q_x_fft:.6f} Q_FFT_y={Q_y_fft:.6f}")

# ── 3. β_y çıkart ────────────────────────────────────────────────────────────
# İntegraratörde her iki quad tipi de |quadG1| büyüklüğünü kullanır:
#   QUAD_F (elem_type=2): +quadG1,  QUAD_D (elem_type=3): -quadG1
# Bu nedenle KL_F = KL_D = KL = g1 * L_Q / Brho
R_dy  = np.load("R_dy_1.npy")
G     = float(cfg['g1'])
L_Q   = float(cfg['quadLen'])
Brho  = p_magic / 0.29979246
KL    = G * L_Q / Brho          # m⁻¹ — QF ve QD için ortak

cot_y     = math.cos(math.pi * Q_y) / math.sin(math.pi * Q_y)
KL_sign_y = np.where(np.arange(n_q) % 2 == 0, +KL, -KL)
diag_dy   = np.diag(R_dy)
beta_y    = diag_dy / (KL_sign_y * cot_y / 2)

beta_y_QF = float(np.mean(beta_y[0::2]))
beta_y_QD = float(np.mean(beta_y[1::2]))
print(f"\nKL = {KL:.5f} m⁻¹  cot(π·Q_y) = {cot_y:.5f}")
print(f"beta_y: QF={beta_y_QF:.2f} m  QD={beta_y_QD:.2f} m")

# ── 4. β_x ve kuplaj tahmini ─────────────────────────────────────────────────
if not (math.isnan(Q_x) or math.isnan(Q_y)):
    beta_x_QF    = abs(beta_y_QD)          # FODO simetrisi
    beta_x_QD    = abs(beta_y_QF)
    # FODO simetrisi: sqrt(bx_QF*by_QF) = sqrt(by_QD*by_QF) = sqrt(bx_QD*by_QD)
    sqrt_bxby    = math.sqrt(abs(beta_y_QF * beta_y_QD))
    print(f"beta_x: QF={beta_x_QF:.2f} m  QD={beta_x_QD:.2f} m")
    print(f"sqrt(bx*by) = {sqrt_bxby:.2f} m  (QF ve QD için aynı)")

    DQ    = Q_x - Q_y
    phi_y = np.array([j * 2 * math.pi * Q_y / n_q for j in range(n_q)])
    phi_x = np.array([j * 2 * math.pi * Q_x / n_q for j in range(n_q)])
    dphi  = phi_x - phi_y

    print(f"\n{'='*55}")
    print(f"KUPLAJ (Q_x={Q_x:.5f}, Q_y={Q_y:.5f}, ΔQ={DQ:.5f})")
    print(f"|sin(πΔQ)| = {abs(math.sin(math.pi*DQ)):.5f}")
    boost = abs(math.sin(math.pi*DQ)) / abs(math.sin(math.pi*DQ/n_q))
    print(f"|Σ exp(i·dphi_j)| = {boost:.2f}  (N_Q={n_q}, rezonans büyütmesi)\n")

    print(f"{'theta[mrad]':>10s}  {'|C|/(4π)':>10s}  {'A_y/A_x':>12s}  "
          f"{'Tracker y/x':>12s}  {'oran':>8s}")
    print("-" * 58)

    # İzleyici (verify_quad_tilt.py) değerleri — sıfır yanlılık hatalı gradyan yok
    tracker_vals = {1: 0.0094, 2: 0.0188, 5: 0.0469, 10: 0.0931}

    for theta_mrad in [1, 2, 5, 10]:
        theta = theta_mrad * 1e-3
        # Her iki quad tipinde KL eşit → ks = 2*theta*KL
        ks    = 2 * theta * KL
        Cre   = ks * sqrt_bxby * float(np.sum(np.cos(dphi)))
        Cim   = ks * sqrt_bxby * float(np.sum(np.sin(dphi)))
        C_mag = math.sqrt(Cre**2 + Cim**2) / (4 * math.pi)
        pred  = C_mag / abs(math.sin(math.pi * DQ))
        meas  = tracker_vals.get(theta_mrad, float('nan'))
        ratio = pred / meas if not math.isnan(meas) and meas > 0 else float('nan')
        print(f"{theta_mrad:>10.1f}  {C_mag:>10.5f}  {pred:>12.5f}  "
              f"{meas:>12.4f}  {ratio:>8.3f}")

    print(f"""
Not: pred/meas oranı ~ {ratio:.2f} beklenen sistematik faktörü yansıtır.
  • Analitik: Courant-Snyder genlik oranı  |A_y / A_x|
  • İzleyici: y_rms / x_rms   (gözlem noktasındaki β etkisi var)
  Dönüşüm: y_rms/x_rms = (A_y/A_x) × sqrt(β_y/β_x)_gözlem_noktası
""")
