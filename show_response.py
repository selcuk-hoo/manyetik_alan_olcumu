"""
show_response.py — analitik R matrislerini ve kondisyon sayılarını rapor eder.

Simülasyon koşmaz, sadece fodo_lattice.py modülünü kullanarak iki gradient
ayarında (g_nom ve g_pert = g_nom·(1+eps)) yanıt matrislerini inşa eder.

Kullanım:
    python show_response.py

İsteğe bağlı: betiğin tepesindeki EPS değiştirilerek farklı gradient
pertürbasyonu denenebilir.
"""

import json
import numpy as np

from fodo_lattice import (
    compute_twiss_at_quads,
    signed_KL,
    build_response_matrix,
    calibrate_K_x_arc,
)

# Gradient pertürbasyon oranı: g_pert = g_nom · (1 + EPS)
EPS = 0.02


def build_R(config, g, plane, K_x_arc=None):
    cfg = dict(config); cfg['g1'] = g
    beta, phi, Q = compute_twiss_at_quads(cfg, plane, K_x_arc=K_x_arc)
    KL = signed_KL(cfg, plane)
    R  = build_response_matrix(beta, phi, Q, KL)
    return R, beta, phi, Q, KL


def report_plane(plane, config):
    g_nom  = config['g1']
    g_pert = g_nom * (1.0 + EPS)

    K_x_arc = calibrate_K_x_arc(config) if plane == 'x' else None

    R1, beta1, phi1, Q1, KL1 = build_R(config, g_nom,  plane, K_x_arc)
    R2, beta2, phi2, Q2, KL2 = build_R(config, g_pert, plane, K_x_arc)
    dR = R1 - R2

    print(f"\n========== Düzlem: {plane} ==========")
    print(f"  g_nom              : {g_nom:.6f} T/m")
    print(f"  g_pert             : {g_pert:.6f} T/m   (Δg/g = {EPS:+.1%})")
    print(f"  Q   (g_nom)        : {Q1:.6f}")
    print(f"  Q   (g_pert)       : {Q2:.6f}")
    print(f"  β aralığı (g_nom)  : [{beta1.min():7.3f}, {beta1.max():7.3f}] m")
    print(f"  |KL| (g_nom)       : {np.abs(KL1).mean():.6f} m⁻¹")
    print(f"  Matris boyutu      : {R1.shape}")
    print()
    print(f"  κ(R₁)              : {np.linalg.cond(R1):10.2f}")
    print(f"  κ(R₂)              : {np.linalg.cond(R2):10.2f}")
    print(f"  κ(ΔR)              : {np.linalg.cond(dR):10.2f}")
    print(f"  κ(ΔR) / κ(R₁)      : {np.linalg.cond(dR)/np.linalg.cond(R1):8.2f}×")
    print()

    # En küçük ve en büyük tekil değerler
    s1 = np.linalg.svd(R1, compute_uv=False)
    s2 = np.linalg.svd(R2, compute_uv=False)
    sd = np.linalg.svd(dR, compute_uv=False)
    print(f"  R₁ singüler: max={s1.max():.4f}  min={s1.min():.4f}")
    print(f"  R₂ singüler: max={s2.max():.4f}  min={s2.min():.4f}")
    print(f"  ΔR singüler: max={sd.max():.4f}  min={sd.min():.4f}")


def main():
    with open("params.json", "r") as f:
        config = json.load(f)

    print("=" * 60)
    print("show_response.py — analitik yanıt matrisi raporu")
    print(f"Pertürbasyon oranı  ε = {EPS:+.3f}")
    print("=" * 60)
    print(f"R₀         : {config['R0']} m")
    print(f"nFODO      : {config['nFODO']}")
    print(f"quadLen    : {config['quadLen']} m")
    print(f"driftLen   : {config['driftLen']} m")

    report_plane('y', config)
    report_plane('x', config)

    print()
    print("Yorum:")
    print("  • κ(R) ≈ 150 → tek-kmod inversiyonu iyi koşullu.")
    print("  • κ(ΔR) çok daha büyük → ΔR yöntemi gürültüye karşı zayıf.")
    print("  • ΔR yöntemi BPM ofsetini iptal eder ama bu pahalıya mal olur.")


if __name__ == "__main__":
    main()
