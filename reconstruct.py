"""
reconstruct.py — iki gradient ayarında simülasyon koşup quad hizalama
rekonstrüksiyon performansını detaylı rapor eder.

İş akışı:
  1. params.json okur, hizalama hatalarını run_simulation.py ile aynı şekilde
     (uniform dağılım, aynı seed) üretir
  2. İki simülasyon koşar:
        sim 1: g_nom  = config['g1']
        sim 2: g_pert = g_nom · (1 + EPS)
  3. Her simülasyonda BPM okumalarını (QF ve QD giriş konumları) çıkarır
  4. fodo_lattice.py'den analitik R₁ ve R₂ matrislerini alır
  5. Dört yöntemin karşılaştırması:
        v₁ = R₁⁻¹·y₁              (tek-kmod, gradient 1)
        v₂ = R₂⁻¹·y₂              (tek-kmod, gradient 2)
        ort = (v₁+v₂)/2           (iki-kmod ortalama)
        ΔR = (R₁-R₂)⁻¹·(y₁-y₂)   (ΔR yöntemi, ofset iptali)
  6. Her yöntemin RMS hata, max hata ve korelasyonunu raporlar
  7. v₁ vs v₂ farkını gösterir (BPM ofset kalıntısı işareti)

Kullanım:
    python reconstruct.py

Önce params.json'da hizalama hatalarını (quad_random_dy_max vb.) ayarla.
"""

import json
import numpy as np

from fodo_lattice import (
    compute_twiss_at_quads,
    signed_KL,
    build_response_matrix,
    calibrate_K_x_arc,
    direct_invert,
)

# Gradient pertürbasyon oranı ve simülasyon süresi — test_params.json'dan okunur
with open("test_params.json", "r") as _f:
    _tp = json.load(_f)
EPS   = float(_tp["EPS"])
T_END = float(_tp["T_END"])


# =============================================================================
# 1. Hata vektörlerini run_simulation.py ile bire bir aynı şekilde üret
# =============================================================================
def generate_misalignments(config):
    """run_simulation.py satır 126-162 ile aynı mantık."""
    nF  = int(config['nFODO'])
    n_q = 2 * nF

    dy = np.zeros(n_q)
    dx = np.zeros(n_q)
    dt = np.zeros(n_q)  # quad tilt
    dip_t = np.zeros(n_q)  # dipole tilt

    # Tek quad hatası
    eq_idx = config.get("error_quad_index", -1)
    if 0 <= eq_idx < n_q:
        dy[eq_idx] += config.get("error_quad_dy", 0.0)
        dx[eq_idx] += config.get("error_quad_dx", 0.0)

    # Rastgele quad dy, dx (uniform, aynı seed - run_simulation.py ile aynı)
    dy_max = config.get("quad_random_dy_max", 0.0)
    dx_max = config.get("quad_random_dx_max", 0.0)
    if dy_max > 0 or dx_max > 0:
        rng_q = np.random.default_rng(config.get("quad_random_seed", 42))
        if dy_max > 0:
            dy += rng_q.uniform(-dy_max, dy_max, n_q)
        if dx_max > 0:
            dx += rng_q.uniform(-dx_max, dx_max, n_q)

    # Rastgele quad tilt
    qt_max = config.get("quad_random_tilt_max", 0.0)
    if qt_max > 0:
        rng_t = np.random.default_rng(config.get("quad_random_tilt_seed", 44))
        dt += rng_t.uniform(-qt_max, qt_max, n_q)

    # Rastgele dipole tilt
    dt_max = config.get("dipole_random_tilt_max", 0.0)
    if dt_max > 0:
        rng_d = np.random.default_rng(config.get("dipole_random_seed", 43))
        dip_t += rng_d.uniform(-dt_max, dt_max, n_q)

    return dy, dx, dt, dip_t


# =============================================================================
# 2. Simülasyon koş, BPM vektörlerini çıkar
# =============================================================================
def run_simulation(config, g_value, dy, dx, dt_quad, dip_t, t_end=T_END):
    """Belirli gradient ve hata dizileriyle simülasyon, BPM çıkışı verir."""
    from integrator import integrate_particle, FieldParams

    M2 = 0.938272046; AMU = 1.792847356; C = 299792458.0; M1 = 1.672621777e-27
    p_magic = M2 / np.sqrt(AMU)
    E_tot   = np.sqrt(p_magic**2 + M2**2)
    beta0   = p_magic / E_tot
    gamma0  = 1.0 / np.sqrt(1.0 - beta0**2)
    R0      = config['R0']
    E0_V_m  = -(p_magic * beta0 / R0) * 1e9
    direction = config.get('direction', -1)
    p_mag   = gamma0 * M1 * C * beta0

    # Başlangıç: dev0=0, y0=0 (saf COD ölçümü)
    y0 = [0.0, 0.0, 0.0,
          0.0, 0.0, p_mag * direction,
          0.0, 0.0, 1.0]

    fields = FieldParams()
    fields.R0 = R0
    fields.E0 = E0_V_m
    fields.E0_power = config.get('E0_power', 1.0)
    fields.quadG1 = float(g_value)
    fields.quadG0 = float(g_value)
    fields.quadSwitch = 1.0
    fields.sextSwitch = 0.0
    fields.EDMSwitch  = 0.0
    fields.direction  = float(direction)
    fields.nFODO      = float(config['nFODO'])
    fields.quadLen    = float(config['quadLen'])
    fields.driftLen   = float(config['driftLen'])
    fields.poincare_quad_index = -1.0
    fields.rfSwitch   = 0.0

    nF  = int(config['nFODO'])
    n_q = 2 * nF
    dt_step = float(config.get('dt', 1e-11))

    integrate_particle(
        y0, 0.0, t_end, dt_step, fields=fields, return_steps=1000,
        quad_dy=dy, quad_dx=dx,
        dipole_tilt=dip_t, quad_tilt=dt_quad,
        quad_dG=np.zeros(n_q),
    )

    # cod_data.txt'yi oku ve BPM konumlarını çıkar
    cod = np.loadtxt("cod_data.txt", skiprows=1)
    x_bpm = np.empty(n_q); y_bpm = np.empty(n_q)
    for k in range(nF):
        x_bpm[2*k    ] = cod[k*8 + 2, 1] * 1e-3   # QF entry: elem=2
        y_bpm[2*k    ] = cod[k*8 + 2, 2] * 1e-3
        x_bpm[2*k + 1] = cod[k*8 + 6, 1] * 1e-3   # QD entry: elem=6
        y_bpm[2*k + 1] = cod[k*8 + 6, 2] * 1e-3
    return x_bpm, y_bpm


# =============================================================================
# 3. Yanıt matrisi
# =============================================================================
def build_R(config, g, plane, K_x_arc=None):
    cfg = dict(config); cfg['g1'] = g
    beta, phi, Q = compute_twiss_at_quads(cfg, plane, K_x_arc=K_x_arc)
    KL = signed_KL(cfg, plane)
    R  = build_response_matrix(beta, phi, Q, KL)
    return R, beta, phi, Q, KL


# =============================================================================
# 4. Rekonstrüksiyon ve metrikler
# =============================================================================
def metrics(dq_hat, dq_true):
    err = dq_hat - dq_true
    rms = np.sqrt(np.mean(err ** 2))
    mx  = np.max(np.abs(err))
    if np.std(dq_hat) > 0 and np.std(dq_true) > 0:
        cr = np.corrcoef(dq_hat, dq_true)[0, 1]
    else:
        cr = np.nan
    return rms, mx, cr


def report_plane(plane, dq_true, y1, y2, config):
    K_x_arc = calibrate_K_x_arc(config) if plane == 'x' else None
    g_nom   = config['g1']
    g_pert  = g_nom * (1.0 + EPS)

    R1, beta, phi, Q1, KL1 = build_R(config, g_nom,  plane, K_x_arc)
    R2, _,    _,   Q2, KL2 = build_R(config, g_pert, plane, K_x_arc)
    dR = R1 - R2

    # Dört yöntem
    v1     = direct_invert(R1, y1)
    v2     = direct_invert(R2, y2)
    avg    = 0.5 * (v1 + v2)
    dq_dR  = direct_invert(dR, y1 - y2)

    # v₁ ile v₂ arasındaki fark → BPM ofset / model uyumsuzluk göstergesi
    v_diff_rms = np.sqrt(np.mean((v1 - v2) ** 2))

    rms_dq = np.sqrt(np.mean(dq_true ** 2))
    rms_y1 = np.sqrt(np.mean(y1 ** 2))
    rms_y2 = np.sqrt(np.mean(y2 ** 2))

    print(f"\n========== Düzlem: {plane} ==========")
    print(f"  Q (g_nom)               : {Q1:.6f}")
    print(f"  Q (g_pert)              : {Q2:.6f}")
    print(f"  κ(R₁) = {np.linalg.cond(R1):.1f}   "
          f"κ(R₂) = {np.linalg.cond(R2):.1f}   "
          f"κ(ΔR) = {np.linalg.cond(dR):.1f}")
    print()
    print(f"  Gerçek Δq RMS          : {rms_dq*1e6:8.2f} μm")
    print(f"  BPM y₁ RMS (sim)       : {rms_y1*1e6:8.2f} μm")
    print(f"  BPM y₂ RMS (sim)       : {rms_y2*1e6:8.2f} μm")
    print(f"  BPM (y₁-y₂) RMS        : {np.sqrt(np.mean((y1-y2)**2))*1e6:8.2f} μm")
    print()
    print(f"  v₁-v₂ rekonstrüksiyon farkı RMS : {v_diff_rms*1e6:8.2f} μm")
    print(f"  (büyük olursa: BPM ofseti, gürültü, veya model uyumsuzluğu var)")
    print()
    print("  Yöntem                    RMS hata   max hata   korelasyon")
    print("  -----------------------   --------   --------   ----------")
    for name, dq_hat in [
        ("v₁ (tek-kmod, g_nom)",   v1),
        ("v₂ (tek-kmod, g_pert)",  v2),
        ("(v₁+v₂)/2 (iki-kmod)",   avg),
        ("ΔR (offset iptali)",     dq_dR),
    ]:
        rms, mx, cr = metrics(dq_hat, dq_true)
        print(f"  {name:25s}  {rms*1e6:7.2f}μm  {mx*1e6:7.2f}μm  {cr:9.6f}")


# =============================================================================
# 5. Ana akış
# =============================================================================
def main():
    with open("params.json", "r") as f:
        config = json.load(f)

    print("=" * 64)
    print("reconstruct.py — iki-gradient quad hizalama rekonstrüksiyonu")
    print("=" * 64)
    print(f"g_nom = {config['g1']:.6f} T/m")
    print(f"EPS   = {EPS:.4f}  (test_params.json)")
    print(f"g_pert = {config['g1']*(1+EPS):.6f} T/m   (Δg/g = {EPS:+.1%})")
    print(f"Simülasyon süresi: {T_END*1e3:.2f} ms (~{T_END/2.0e-7:.0f} tur)  (test_params.json)")

    # 1. Hata vektörlerini üret (run_simulation.py ile aynı)
    dy, dx, dt_quad, dip_t = generate_misalignments(config)
    print()
    print(f"Üretilen hata RMS (run_simulation.py ile aynı):")
    print(f"  quad dy     RMS = {np.sqrt(np.mean(dy**2))*1e6:.2f} μm")
    print(f"  quad dx     RMS = {np.sqrt(np.mean(dx**2))*1e6:.2f} μm")
    print(f"  quad tilt   RMS = {np.sqrt(np.mean(dt_quad**2))*1e6:.2f} μrad")
    print(f"  dipole tilt RMS = {np.sqrt(np.mean(dip_t**2))*1e6:.2f} μrad")

    # 2. İki simülasyon koş
    g_nom  = config['g1']
    g_pert = g_nom * (1.0 + EPS)

    print(f"\n[1/2] Simülasyon, g = {g_nom:.6f} T/m ...")
    x1_bpm, y1_bpm = run_simulation(config, g_nom,  dy, dx, dt_quad, dip_t)

    print(f"\n[2/2] Simülasyon, g = {g_pert:.6f} T/m ...")
    x2_bpm, y2_bpm = run_simulation(config, g_pert, dy, dx, dt_quad, dip_t)

    # 3. Rekonstrüksiyon, her düzlem için ayrı rapor
    report_plane('y', dy, y1_bpm, y2_bpm, config)
    report_plane('x', dx, x1_bpm, x2_bpm, config)


if __name__ == "__main__":
    main()
