#!/usr/bin/env python3
"""analytic_coupling.py — Analitik x-y kuplaj tahmini (düzeltilmiş).

Düzeltmeler (v2):
  1. KL: integratörde QF +quadG1, QD -quadG1 → KL_F = KL_D = g1·L/Bρ.
         (quadG0 yalnızca QUAD_F_MOD modülasyon tabanı içindir.)
  2. Q_x, Q_y: hardcode yerine Hanning-FFT ile simülasyondan ölçülüyor.

Formüller:
  R_dy[j,j] = KL_sign_y(j) · β_yj · cot(πQ_y) / 2
  C = (1/4π) · Σ_j k_sj · sqrt(β_xj·β_yj) · exp(i·(φ_xj − φ_yj))
  A_y/A_x = |C| / |sin(πΔQ)|
"""
import json, math, sys, os
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

R_dy = np.load("R_dy_1.npy")
R_dx = np.load("R_dx_1.npy")

with open("params.json") as fh:
    cfg = json.load(fh)

N_Q = 2 * int(cfg["nFODO"])   # 48
L_Q = float(cfg["quadLen"])
G   = float(cfg["g1"])         # 0.21 T/m — QF ve QD için ortak büyüklük

M2 = 0.938272046; AMU = 1.792847356; C_light = 299792458.0; M1 = 1.672621777e-27
p_magic = M2 / math.sqrt(AMU)
E_tot   = math.sqrt(p_magic**2 + M2**2)
beta0   = p_magic / E_tot
gamma0  = 1 / math.sqrt(1 - beta0**2)
R0      = cfg['R0']; direction = float(cfg.get('direction', -1))
p_mag   = gamma0 * M1 * C_light * beta0
E0_V_m  = -(p_magic * beta0 / R0) * 1e9
T_turn  = 2 * math.pi * R0 / (beta0 * C_light)

Brho = p_magic / 0.29979246           # T·m
KL   = G * L_Q / Brho                  # m⁻¹ — QF ve QD ortak KL


# ── FFT ile tune ölçümü ──────────────────────────────────────────────────────
def _run_kick(theta_hor, theta_ver, T_END=5e-4, return_steps=10000):
    from integrator import integrate_particle, FieldParams
    f = FieldParams()
    f.R0 = R0; f.E0 = E0_V_m; f.E0_power = cfg.get('E0_power', 1.0)
    f.quadG1 = G; f.quadG0 = G
    f.quadSwitch = 1.0; f.sextSwitch = 0.0; f.EDMSwitch = 0.0
    f.direction = direction; f.nFODO = float(cfg['nFODO'])
    f.quadLen = L_Q; f.driftLen = float(cfg['driftLen'])
    f.poincare_quad_index = -1.0; f.rfSwitch = 0.0
    Px = p_mag * theta_hor
    Pz = p_mag * theta_ver
    Py = p_mag * direction * math.sqrt(max(0, 1 - theta_hor**2 - theta_ver**2))
    n_q = N_Q
    qt = np.zeros(n_q)
    hist, _, _ = integrate_particle(
        [0, 0, 0, Px, Pz, Py, 0, 0, direction], 0, T_END, float(cfg['dt']),
        fields=f, return_steps=return_steps,
        quad_dy=np.zeros(n_q), quad_dx=np.zeros(n_q),
        dipole_tilt=np.zeros(n_q), quad_tilt=qt, quad_dG=np.zeros(n_q))
    return hist[:, 0], hist[:, 1]   # x, z


def _fft_tune(signal, T_END):
    """Hanning-pencereli FFT + parabolik interpolasyon → Q = f_beta · T_turn."""
    n    = len(signal)
    spec = np.abs(np.fft.rfft((signal - signal.mean()) * np.hanning(n)))
    pk   = int(np.argmax(spec[1:])) + 1
    if 1 < pk < len(spec) - 1:
        d = spec[pk-1] - 2*spec[pk] + spec[pk+1]
        pk = pk + 0.5*(spec[pk-1] - spec[pk+1]) / d if abs(d) > 0 else pk
    return (pk / T_END) * T_turn


print("=== Q_x ve Q_y FFT ile ölçülüyor ===")
T_FFT = 5e-4
x_h, _ = _run_kick(1e-3, 0,   T_END=T_FFT)
_, z_v  = _run_kick(0,   1e-3, T_END=T_FFT)
Q_x = _fft_tune(x_h, T_FFT)
Q_y = _fft_tune(z_v, T_FFT)
print(f"  Q_x (FFT) = {Q_x:.5f}")
print(f"  Q_y (FFT) = {Q_y:.5f}")
print(f"  ΔQ = {Q_x - Q_y:.5f}\n")


# ── β_y çıkart ───────────────────────────────────────────────────────────────
# R_dy[j,j] = KL_sign_y(j) · β_yj · cot(πQ_y) / 2
# KL_sign_y: QF (çift) = +KL,  QD (tek) = -KL
cot_y = math.cos(math.pi * Q_y) / math.sin(math.pi * Q_y)
KL_sign_y = np.where(np.arange(N_Q) % 2 == 0, +KL, -KL)
diag_dy   = np.diag(R_dy)
beta_y    = diag_dy / (KL_sign_y * cot_y / 2)

beta_y_QF = float(np.mean(beta_y[0::2]))
beta_y_QD = float(np.mean(beta_y[1::2]))

print("=== Kafes parametreleri (R matrisinden) ===\n")
print(f"Brho = {Brho:.4f} T·m")
print(f"KL = {KL:.5f} m⁻¹  (QF ve QD ortak, g1={G} T/m)")
print(f"cot(π·Q_y) = {cot_y:.4f}  (Q_y={Q_y:.5f})\n")
print(f"beta_y: QF={beta_y_QF:.2f} m  QD={beta_y_QD:.2f} m")

# ── β_x (FODO simetrisi) ──────────────────────────────────────────────────────
# FODO: β_x(QF) ≈ β_y(QD),  β_x(QD) ≈ β_y(QF)
beta_x_QF = abs(beta_y_QD)
beta_x_QD = abs(beta_y_QF)
print(f"beta_x: QF={beta_x_QF:.2f} m  QD={beta_x_QD:.2f} m  (FODO simetrisi)")

# FODO simetrisi: sqrt(bx·by) QF ve QD için aynı = sqrt(by_QF · by_QD)
sqrt_bxby = math.sqrt(abs(beta_y_QF * beta_y_QD))
print(f"sqrt(bx·by) = {sqrt_bxby:.2f} m")

DeltaQ = Q_x - Q_y
print(f"\nΔQ = {Q_x:.5f} − {Q_y:.5f} = {DeltaQ:.5f}")
print(f"|sin(πΔQ)| = {abs(math.sin(math.pi * DeltaQ)):.5f}")

# Faz farkı dizisi (uniform faz ilerleme tahmini)
phi_y = np.array([j * 2 * math.pi * Q_y / N_Q for j in range(N_Q)])
phi_x = np.array([j * 2 * math.pi * Q_x / N_Q for j in range(N_Q)])
dphi  = phi_x - phi_y

sum_exp = np.sum(np.exp(1j * dphi))
boost   = abs(math.sin(math.pi * DeltaQ)) / abs(math.sin(math.pi * DeltaQ / N_Q))
print(f"\n|Σ exp(i·dphi_j)|  teorik={boost:.2f}  numerik={abs(sum_exp):.2f}")

# ── Kuplaj tahmini (uniform tilt) ────────────────────────────────────────────
print("\n=== Kuplaj tahmini ===\n")
print(f"{'theta[mrad]':>10s}  {'|C|/(4π)':>10s}  {'A_y/A_x':>10s}")
print("-" * 36)

results_analytic = {}
for theta_mrad in [0, 1, 2, 5, 10]:
    if theta_mrad == 0:
        print(f"{'0':>10s}  {'—':>10s}  {'0.0000':>10s}")
        results_analytic[theta_mrad] = 0.0
        continue
    theta = theta_mrad * 1e-3
    # k_s = 2·theta·KL  (QF ve QD aynı, sqrt(bx·by) da aynı)
    ks    = 2 * theta * KL
    Cre   = ks * sqrt_bxby * float(np.sum(np.cos(dphi)))
    Cim   = ks * sqrt_bxby * float(np.sum(np.sin(dphi)))
    C_mag = math.sqrt(Cre**2 + Cim**2) / (4 * math.pi)
    ratio = C_mag / abs(math.sin(math.pi * DeltaQ))
    print(f"{theta_mrad:>10.1f}  {C_mag:>10.5f}  {ratio:>10.5f}")
    results_analytic[theta_mrad] = ratio

# ── verify_quad_tilt.py ile karşılaştırma ────────────────────────────────────
print("\n=== verify_quad_tilt.py sonuçlarıyla karşılaştırma ===")
import subprocess
result = subprocess.run(
    [sys.executable, "verify_quad_tilt.py"],
    capture_output=True, text=True, cwd=os.getcwd()
)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[:500])

# tracker çıktısını parse et
tracker_ratios = {}
for line in result.stdout.splitlines():
    try:
        parts = line.split()
        if len(parts) >= 4 and parts[0].replace('.', '').replace('-', '').isdigit():
            t_mrad   = float(parts[0])
            ratio_t  = float(parts[3])
            tracker_ratios[t_mrad] = ratio_t
    except Exception:
        pass

print("\n=== Karşılaştırma özeti ===\n")
print(f"{'theta[mrad]':>10s}  {'Analitik A_y/A_x':>18s}  {'Tracker y/x':>12s}  {'Oran':>8s}")
print("-" * 56)
for t_mrad in [1, 2, 5, 10]:
    a_pred = results_analytic.get(t_mrad, float('nan'))
    t_meas = tracker_ratios.get(float(t_mrad), float('nan'))
    if not math.isnan(t_meas) and a_pred > 0:
        ratio_at = t_meas / a_pred
        print(f"{t_mrad:>10.1f}  {a_pred:>18.5f}  {t_meas:>12.4f}  {ratio_at:>8.3f}")
    else:
        print(f"{t_mrad:>10.1f}  {a_pred:>18.5f}  {'—':>12s}  {'—':>8s}")

print(f"""
Kafes özeti (FFT Q değerleri):
  Q_y = {Q_y:.5f},  Q_x = {Q_x:.5f},  ΔQ = {DeltaQ:.5f}
  beta_y(QF) = {beta_y_QF:.1f} m,  beta_y(QD) = {beta_y_QD:.1f} m
  beta_x(QF) = {beta_x_QF:.1f} m,  beta_x(QD) = {beta_x_QD:.1f} m
  sqrt(bx·by) = {sqrt_bxby:.1f} m
  KL = {KL:.5f} m⁻¹  (QF = QD, g1={G} T/m)

Tahmin formülü:
  A_y/A_x = |C| / |sin(πΔQ)|
  C = (θ·KL·sqrt(bx·by)/2π) · Σ_j exp(i·j·2πΔQ/N_Q)
  Rezonans büyütme: {boost:.1f}  (N_Q={N_Q}, ΔQ={DeltaQ:.5f})

Sistematik oran (tracker/analitik) ≈ tracker = y_rms/x_rms,
analitik = A_y/A_x Courant-Snyder; dönüşüm faktörü sqrt(β_y/β_x)
gözlem noktasında tam olarak 1 değil.
""")
