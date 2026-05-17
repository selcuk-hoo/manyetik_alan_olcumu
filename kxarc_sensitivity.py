"""
kxarc_sensitivity.py — Test 3 (yapilacaklar-2.md §Test 3)

Inverse-crime kontrolü: forward simülasyon TRUE K_x_arc ile, inverse
rekonstrüksiyon PERTURBED K_x_arc·(1+δ) ile yapılır. δ ∈ [-10%, +10%].

Reconstruction RMS hatasının δ'ya bağımlılığı, modelin gerçek halkada
LOCO ile <1% hassasiyetle kalibre edildiğinde yeterli olup olmadığını
gösterir.

Y düzlemi: K_y_arc = 0 Maxwell garantisi — kontrolde değişmemeli.
X düzlemi: hassasiyetin kritik test edildiği düzlem.

Kullanım:
    python kxarc_sensitivity.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fodo_lattice import (
    compute_twiss_at_quads, signed_KL, build_response_matrix,
    calibrate_K_x_arc, direct_invert,
)
from reconstruct import generate_misalignments, run_simulation, EPS

with open("test_params.json", "r") as _f:
    _tp = json.load(_f)
_t3 = _tp["test3"]
DELTA_MIN = float(_t3["delta_min"])
DELTA_MAX = float(_t3["delta_max"])
DELTA_N   = int(_t3["delta_n"])


def build_R(config, g, plane, K_x_arc=None):
    cfg = dict(config); cfg['g1'] = g
    beta, phi, Q = compute_twiss_at_quads(cfg, plane, K_x_arc=K_x_arc)
    KL = signed_KL(cfg, plane)
    return build_response_matrix(beta, phi, Q, KL)


def rms(v): return float(np.sqrt(np.mean(v ** 2)))


def main():
    with open("params.json", "r") as f:
        config = json.load(f)

    print("=" * 72)
    print("kxarc_sensitivity.py — Test 3 (inverse-crime kontrolü)")
    print("=" * 72)

    # 1. Forward simülasyonu TRUE K_x_arc ile koş (reconstruct.py akışı)
    dy_q, dx_q, dt_q, dip_t = generate_misalignments(config)
    g_nom = config['g1']; g_pert = g_nom * (1.0 + EPS)

    print(f"\n[forward] g_nom  = {g_nom:.5f} ...")
    x1, y1 = run_simulation(config, g_nom,  dy_q, dx_q, dt_q, dip_t)
    print(f"[forward] g_pert = {g_pert:.5f} ...")
    x2, y2 = run_simulation(config, g_pert, dy_q, dx_q, dt_q, dip_t)

    # 2. TRUE K_x_arc (referans)
    K_x_true = calibrate_K_x_arc(config)
    print(f"\nK_x_arc (true, kalibre edilmiş) = {K_x_true:.6e} m⁻²")

    # 3. δ taraması
    print(f"\nδ aralığı: [{DELTA_MIN*100:.1f}%, {DELTA_MAX*100:.1f}%], "
          f"n={DELTA_N}  (test_params.json)")
    deltas = np.linspace(DELTA_MIN, DELTA_MAX, DELTA_N)
    rms_x_v1   = []; rms_x_v2   = []; rms_x_avg  = []
    rms_y_v1   = []; rms_y_v2   = []; rms_y_avg  = []

    for d in deltas:
        K_x_pert = K_x_true * (1.0 + d)

        # x düzlemi: perturbed K_x_arc ile inverse
        R1x = build_R(config, g_nom,  'x', K_x_arc=K_x_pert)
        R2x = build_R(config, g_pert, 'x', K_x_arc=K_x_pert)
        v1x = direct_invert(R1x, x1)
        v2x = direct_invert(R2x, x2)
        avgx = 0.5 * (v1x + v2x)
        rms_x_v1.append( rms(v1x - dx_q) )
        rms_x_v2.append( rms(v2x - dx_q) )
        rms_x_avg.append(rms(avgx - dx_q))

        # y düzlemi: K_x_arc'tan etkilenmez (kontrol). Aynı işlem ama
        # K_x_arc=None (y için kullanılmıyor)
        R1y = build_R(config, g_nom,  'y')
        R2y = build_R(config, g_pert, 'y')
        v1y = direct_invert(R1y, y1)
        v2y = direct_invert(R2y, y2)
        avgy = 0.5 * (v1y + v2y)
        rms_y_v1.append( rms(v1y - dy_q) )
        rms_y_v2.append( rms(v2y - dy_q) )
        rms_y_avg.append(rms(avgy - dy_q))

    rms_x_v1  = np.array(rms_x_v1);  rms_x_v2  = np.array(rms_x_v2)
    rms_x_avg = np.array(rms_x_avg); rms_y_avg = np.array(rms_y_avg)

    print(f"\n  {'δK/K [%]':>9s}  {'x avg RMS [μm]':>16s}  "
          f"{'y avg RMS [μm]':>16s}  {'x v2 RMS [μm]':>15s}")
    print("  " + "-" * 65)
    for i, d in enumerate(deltas):
        print(f"  {d*100:>9.1f}  {rms_x_avg[i]*1e6:>14.3f}  "
              f"{rms_y_avg[i]*1e6:>14.3f}  {rms_x_v2[i]*1e6:>13.3f}")

    # En düşük noktayı bul (optimal δ)
    i_min_x = int(np.argmin(rms_x_avg))
    i_min_v2 = int(np.argmin(rms_x_v2))
    print(f"\n  Optimal δ (x avg)  : {deltas[i_min_x]*100:+.2f}%  "
          f"→ RMS = {rms_x_avg[i_min_x]*1e6:.3f} μm")
    print(f"  Optimal δ (x v2)   : {deltas[i_min_v2]*100:+.2f}%  "
          f"→ RMS = {rms_x_v2[i_min_v2]*1e6:.3f} μm")

    # Grafik
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.plot(deltas * 100, rms_x_v1  * 1e6, '-o', label=r"$x$: $v_1$", color='C0')
    ax.plot(deltas * 100, rms_x_v2  * 1e6, '-s', label=r"$x$: $v_2$", color='C1')
    ax.plot(deltas * 100, rms_x_avg * 1e6, '-^', label=r"$x$: $(v_1+v_2)/2$",
            color='C2', lw=2)
    ax.plot(deltas * 100, rms_y_avg * 1e6, '--D', label=r"$y$: kontrol "
            r"(Maxwell, $K_{y,arc}=0$)", color='C3', markersize=4)
    ax.axhline(5.0, color='gray', ls=':', alpha=0.6, label="5 μm hedef")
    ax.axhline(10.0, color='red', ls=':', alpha=0.6, label="10 μm tavan")
    ax.axvline(0.0, color='black', ls='-', alpha=0.3)
    ax.set_xlabel(r"$K_{x,arc}$ pertürbasyonu $\delta$ [%]")
    ax.set_ylabel("Reconstruction RMS hatası [μm]")
    ax.set_title(r"$K_{x,arc}$ modeli hassasiyeti")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    # Yakın bölge (zoom)
    ax = axes[1]
    ax.plot(deltas * 100, rms_x_avg * 1e6, '-^', label=r"$x$: $(v_1+v_2)/2$",
            color='C2', lw=2)
    ax.plot(deltas * 100, rms_x_v2 * 1e6, '-s', label=r"$x$: $v_2$ (best)",
            color='C1')
    ax.axhline(5.0, color='gray', ls=':', alpha=0.6, label="5 μm hedef")
    ax.axhline(10.0, color='red', ls=':', alpha=0.6, label="10 μm tavan")
    ax.axvline(0.0, color='black', ls='-', alpha=0.3)
    ax.set_xlabel(r"$K_{x,arc}$ pertürbasyonu $\delta$ [%]")
    ax.set_ylabel("Reconstruction RMS hatası [μm]")
    ax.set_title("Yakın bölge")
    ax.set_ylim(0, 30)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle(r"Test 3: $K_{x,arc}$ inverse-crime sensitivity",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig("test3_kxarc_sensitivity.png", dpi=140)
    print("\nKaydedildi: test3_kxarc_sensitivity.png")


if __name__ == "__main__":
    main()
