"""
verify_quad_tilt.py — quad tilt'in simülasyonda doğru uygulandığını doğrulayan
temiz fizik testi.

Test fikri: hiç misalignment koymadan parçacığa sadece YATAY başlangıç kicki
verilir (theta0_hor). Bu durumda:
  • Tilt = 0 ise: dikey hareket asla başlatılmaz, y(t) ≡ 0 olmalı.
  • Tilt ≠ 0 ise: skew-quadrupol kuplaji x betatron salınımını y'ye taşır,
    y(t) salınımları görülür ve genliği tilt ile doğrusal artmalı.

Bu, Stage D'de tilt etkisinin neden küçük göründüğünü de aydınlatır:
  • COD ölçümünde başlangıç kick'i yok, sadece misalignment kaynaklı x_co var.
  • x_co kendisi pertürbasyon olduğundan tilt × x_co ikinci mertebe küçük.
  • Burada explicit horizontal kick verince kuplaj birinci mertebe → büyük etki.

Kullanım:
    python verify_quad_tilt.py
"""

import json
import numpy as np

T_END = 5e-5   # ~17 tur, kuplaj görmek için yeterli
THETA0_HOR = 1e-3   # 1 mrad başlangıç yatay açısal kicki


def run_one(config, quad_tilt_uniform):
    """Tüm quadlara aynı tilt verir, simülasyon koşar, x ve y geçmişini döner."""
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

    # Başlangıç: konum merkezde, yatay açısal kick var, dikey hareket sıfır
    # p = p_mag * direction (Y yönünde), kick eklemek için Px'e küçük bir değer
    Px = p_mag * THETA0_HOR  # küçük açı için Px ≈ p_total · θ
    Py = p_mag * direction * np.sqrt(1.0 - THETA0_HOR**2)

    y0 = [0.0, 0.0, 0.0,
          Px, 0.0, Py,
          0.0, 0.0, 1.0]

    fields = FieldParams()
    fields.R0 = R0
    fields.E0 = E0_V_m
    fields.E0_power = config.get('E0_power', 1.0)
    fields.quadG1 = float(config['g1'])
    fields.quadG0 = float(config['g1'])
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

    qt = np.full(n_q, quad_tilt_uniform)

    history_local, _, _ = integrate_particle(
        y0, 0.0, T_END, dt_step, fields=fields, return_steps=2000,
        quad_dy=np.zeros(n_q), quad_dx=np.zeros(n_q),
        dipole_tilt=np.zeros(n_q), quad_tilt=qt,
        quad_dG=np.zeros(n_q),
    )
    # history_local: [x_dev, z_vert, y_long, vx, vz, vy, sx, sy, sz]
    x = history_local[:, 0]   # radyal sapma [m]
    y = history_local[:, 1]   # dikey konum [m]
    return x, y


def main():
    with open("params.json", "r") as f:
        config = json.load(f)

    print("=" * 64)
    print("verify_quad_tilt.py — skew-quadrupol kuplaji doğrulaması")
    print("=" * 64)
    print(f"Test: sıfır misalignment, yalnız θ_x = {THETA0_HOR*1e3:.2f} mrad kicki")
    print(f"Süre: {T_END*1e3:.2f} ms (~{T_END/3e-6:.0f} tur)")
    print()
    print("Beklenti:")
    print("  • tilt = 0   → y(t) ≡ 0    (kuplaj kanalı yok)")
    print("  • tilt > 0   → y(t) salınır, genlik ∝ tilt")
    print("  • genlik 2× tilt yapıldığında 2× artar → skew fiziği doğrulanır")
    print()

    tilt_levels = [0.0, 1e-3, 2e-3, 5e-3, 10e-3]
    print(f"{'tilt [mrad]':>12s}  {'x RMS [μm]':>12s}  {'y RMS [μm]':>12s}  "
          f"{'y/x oranı':>10s}")
    print("-" * 56)

    results = []
    for tilt in tilt_levels:
        x, y = run_one(config, tilt)
        x_rms = np.sqrt(np.mean(x**2)) * 1e6
        y_rms = np.sqrt(np.mean(y**2)) * 1e6
        ratio = y_rms / x_rms if x_rms > 0 else 0.0
        print(f"{tilt*1e3:>12.3f}  {x_rms:>12.3f}  {y_rms:>12.3f}  "
              f"{ratio:>10.4f}")
        results.append((tilt, x_rms, y_rms))

    # Doğrusallık kontrolü: tilt > 0 için y_rms / tilt sabit olmalı
    print()
    print("Doğrusallık (y_rms / tilt sabit olmalı, sıfır olmayan tiltler için):")
    for tilt, x_rms, y_rms in results[1:]:
        print(f"  tilt = {tilt*1e3:5.2f} mrad → y_rms/tilt = "
              f"{y_rms/(tilt*1e3):8.3f} μm/mrad")

    print()
    if results[0][2] < 1e-3:   # tilt=0 → y_rms < 1 nm
        print("  ✓ tilt=0'da y hareketi yok (< 1 nm)")
    else:
        print(f"  ✗ tilt=0'da bile y_rms = {results[0][2]:.3f} μm "
              "(beklenmeyen kaynak var!)")

    y_per_tilt = [r[2]/(r[0]*1e3) for r in results[1:]]
    spread = (max(y_per_tilt) - min(y_per_tilt)) / np.mean(y_per_tilt)
    if spread < 0.20:
        print(f"  ✓ y_rms/tilt sabitliği iyi (yayılma = {spread*100:.1f}%) "
              "→ skew fiziği DOĞRUSAL")
    else:
        print(f"  ⚠ y_rms/tilt yayılması büyük ({spread*100:.1f}%) "
              "→ doğrusal rejim dışına çıkıldı mı?")


if __name__ == "__main__":
    main()
