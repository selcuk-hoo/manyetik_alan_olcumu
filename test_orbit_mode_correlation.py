#!/usr/bin/env python3
"""test_orbit_mode_correlation.py — Yörünge uzayında mod korelasyon yapısı.

SORU (kullanıcı): "k modları ile sahte EDM ilişkisini gösteren grafik
modların korelasyonunu göstermiyordu. Korelasyon bilgisini elde etmenin
bir yolu var mı? Belki o zaman fit'i neden 4'te kesmemiz gerektiğini
anlarız."

YANITLANAN ÜÇ ŞEY:
  1. GRAM MATRİSİ: 24 mod düğmesinin (k=1..12 × cos/sin) yörünge parmak
     izleri arasındaki korelasyon C_ij = ô_i·ô_j. Köşegene yakınsa LSQ
     modları temiz ayırır → fit kesiminin nedeni korelasyon DEĞİLDİR.
  2. SIZINTI: fit dışı modların gerçek içeriği, fit edilen kestirimlere
     ne kadar sızar? L = O_fit⁺ · O_dışı  (kestirim hatasının ofset-dışı
     bileşeni).
  3. EŞİK: kazanç yasası G_k = C/|Q²−k²| k=7..12'de de geçerli mi?
     (§12.12'deki yasa k≤6'ya oturtulmuştu — burada ÖNGÖRÜ test edilir.)
     Fit kesimi G_k > σ_b/σ_q eşiğinden gelir; korelasyondan değil.

Ek simülasyon: ref + k=7..12 × cos/sin = 13 yörünge koşumu (t2=1ms).
k=1..6 kalibrasyonu test_orbit_trim.json'dan yeniden kullanılır.

Çıktı: test_orbit_mode_correlation.json, test_orbit_mode_correlation.png
"""

import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
sys.path.insert(0, BASE)

from test_orbit_trim import (_worker, mode_vec_pair, spektrum,
                             T2, PATTERN_RMS, OFFSET_RMS,
                             PATTERN_SEED, OFFSET_SEED, A_CAL)

K_EXT = [7, 8, 9, 10, 11, 12]    # kalibrasyona eklenen modlar
# §12.12'de k=1..6'ya oturtulan yasa — k=7..12 için saf öngörü
LAW_C, LAW_Q2 = 24.8, 5.03


def main():
    t0 = time.time()
    with open("params.json") as fh:
        config = json.load(fh)
    from false_edm_mode_scan import setup_fields
    fields, *_ = setup_fields(config)
    n_q     = 2 * int(fields.nFODO)
    antisym = config.get("smooth_antisym_fodo", True)
    ctx     = mp.get_context("spawn")
    nw      = mp.cpu_count()

    # Mevcut kalibrasyon (k=1..6)
    with open("test_orbit_trim.json") as fh:
        prev = json.load(fh)
    O_low = np.array(prev["O_matrisi"])              # [48×12]
    names_low = list(prev["kazanclar"].keys())

    # ── k=7..12 kalibrasyon koşumları ───────────────────────────────────────
    tasks = [("ref", np.zeros(n_q).tolist(), "orbit", T2, 10)]
    for k in K_EXT:
        c, s = mode_vec_pair(n_q, k, antisym)
        tasks.append((f"o{k}c", (A_CAL*c).tolist(), "orbit", T2, 10))
        tasks.append((f"o{k}s", (A_CAL*s).tolist(), "orbit", T2, 10))
    print(f"{len(tasks)} yörünge simülasyonu ({nw} işçi)...")
    with ctx.Pool(processes=min(nw, len(tasks))) as pool:
        res = dict(pool.map(_worker, tasks))

    y_ref = np.asarray(res["ref"])
    O_cols, names_ext = [], []
    for k in K_EXT:
        for ph, tag in (("cos", "c"), ("sin", "s")):
            O_cols.append((np.asarray(res[f"o{k}{tag}"]) - y_ref) / A_CAL)
            names_ext.append(f"k{k} {ph}")
    O_ext = np.column_stack(O_cols)                  # [48×12]
    O_all = np.hstack([O_low, O_ext])                # [48×24]
    names = names_low + names_ext

    # ── 1. Kazanç yasası testi (k=7..12 öngörüsü) ───────────────────────────
    gains = np.sqrt(np.mean(O_all**2, axis=0))
    print(f"\n{'─'*56}")
    print("Kazanç yasası G_k = 24.8/|5.03−k²| — k=7..12 ÖNGÖRÜ testi")
    print(f"{'─'*56}")
    print(f"{'k':>4} {'ölçülen':>9} {'yasa':>9} {'sapma %':>8}")
    gain_by_k = {}
    for k in range(1, 13):
        i = names.index(f"k{k} cos")
        g_meas = 0.5*(gains[i] + gains[i+1])
        g_law  = LAW_C / abs(LAW_Q2 - k**2)
        gain_by_k[k] = (g_meas, g_law)
        print(f"{k:>4} {g_meas:>9.3f} {g_law:>9.3f} {100*(g_law/g_meas-1):>+8.1f}")

    # ── 2. Gram (korelasyon) matrisi ────────────────────────────────────────
    O_hat = O_all / np.linalg.norm(O_all, axis=0)
    G = O_hat.T @ O_hat                              # [24×24]
    off = G - np.diag(np.diag(G))
    print(f"\nGram matrisi: max |köşegen-dışı| = {np.abs(off).max():.3f}")
    # k≤6 (fit adayları) bloğu içinde:
    off_low = np.abs(off[:12, :12]).max()
    print(f"  k≤6 bloğu içinde max: {off_low:.3f}")

    # ── 3. Sızıntı: fit dışı içerik → fit kestirimi ─────────────────────────
    # C fiti (k=1..4 → ilk 8 sütun)
    O_C = O_all[:, :8]
    O_rest = O_all[:, 8:]
    L = np.linalg.pinv(O_C) @ O_rest                 # [8×16]
    print(f"\nSızıntı matrisi L = O_C⁺·O_dışı: max |L| = {np.abs(L).max():.3f}")

    # Seed=321 deseni için hata muhasebesi: kestirim hatası ≈ O⁺b + L·a_dışı
    P = np.random.default_rng(PATTERN_SEED).standard_normal(n_q) * PATTERN_RMS
    b = np.random.default_rng(OFFSET_SEED).standard_normal(n_q) * OFFSET_RMS
    spec, by_k = spektrum(P, n_q, antisym)
    a_rest = []
    for k in range(5, 13):
        a_rest.append(by_k[k].get('cos', 0.0))
        a_rest.append(by_k[k].get('sin', 0.0))
    a_rest = np.array(a_rest)

    eps_ofs  = np.linalg.pinv(O_C) @ b               # ofset yanlılığı
    eps_leak = L @ a_rest                            # mod sızıntısı
    eps_meas = np.array([prev["varyantlar"]["C"]["kestirim_hata_um"][nm]
                         for nm in names[:8]]) * 1e-6
    print(f"\n{'─'*72}")
    print("C-fit kestirim hatası muhasebesi [μm] (seed 321/777)")
    print(f"{'─'*72}")
    print(f"{'düğme':>9} {'ölçülen':>9} {'ofset O⁺b':>10} {'sızıntı L·a':>12} {'toplam':>9}")
    for i, nm in enumerate(names[:8]):
        print(f"{nm:>9} {eps_meas[i]*1e6:>9.2f} {eps_ofs[i]*1e6:>10.2f} "
              f"{eps_leak[i]*1e6:>12.2f} {(eps_ofs[i]+eps_leak[i])*1e6:>9.2f}")

    out = {"_aciklama": "Yörünge uzayında mod korelasyonu, sızıntı ve "
                        "kazanç yasası k=7..12 doğrulaması",
           "kazanc_olculen": {k: v[0] for k, v in gain_by_k.items()},
           "kazanc_yasa":    {k: v[1] for k, v in gain_by_k.items()},
           "gram_max_offdiag": float(np.abs(off).max()),
           "gram_max_offdiag_k6blok": float(off_low),
           "sizinti_max": float(np.abs(L).max()),
           "muhasebe_um": {names[i]: {"olculen": float(eps_meas[i]*1e6),
                                      "ofset": float(eps_ofs[i]*1e6),
                                      "sizinti": float(eps_leak[i]*1e6)}
                           for i in range(8)},
           "O_ext": O_ext.tolist(), "names": names}
    with open("test_orbit_mode_correlation.json", "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("\nKaydedildi: test_orbit_mode_correlation.json")

    # ── Figür: 3 panel ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 4.6))

    ax1 = fig.add_subplot(1, 3, 1)
    im = ax1.imshow(np.abs(G), cmap="viridis", vmin=0, vmax=1)
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_xticklabels([f"k{k}" for k in range(1, 13)], fontsize=7)
    ax1.set_yticks(range(0, 24, 2))
    ax1.set_yticklabels([f"k{k}" for k in range(1, 13)], fontsize=7)
    ax1.set_title(f"Yörünge parmak izi Gram matrisi |ô_i·ô_j|\n"
                  f"max köşegen-dışı = {np.abs(off).max():.2f}")
    plt.colorbar(im, ax=ax1, fraction=0.046)

    ax2 = fig.add_subplot(1, 3, 2)
    ks = np.arange(1, 13)
    g_meas = [gain_by_k[k][0] for k in ks]
    g_law  = [gain_by_k[k][1] for k in ks]
    ax2.semilogy(ks, g_meas, "o", label="ölçülen", ms=7)
    ax2.semilogy(ks, g_law, "-", label="yasa 24.8/|5.03−k²|")
    for ratio, ls in ((1.0, "--"), (0.5, ":")):
        ax2.axhline(ratio, color="r", ls=ls, lw=1,
                    label=f"eşik σ_b/σ_q={ratio}")
    ax2.axvspan(0.5, 4.5, alpha=0.12, color="g")
    ax2.set_xlabel("mod k")
    ax2.set_ylabel("yörünge kazancı G_k")
    ax2.set_title("Kazanç yasası ve fit eşiği\n(yeşil bant: σ_b=σ_q'da güvenli bölge)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(1, 3, 3)
    xi = np.arange(8)
    w = 0.27
    ax3.bar(xi - w, eps_meas*1e6, w, label="ölçülen hata")
    ax3.bar(xi,      eps_ofs*1e6,  w, label="ofset O⁺b")
    ax3.bar(xi + w,  eps_leak*1e6, w, label="sızıntı L·a")
    ax3.set_xticks(xi)
    ax3.set_xticklabels(names[:8], rotation=45, fontsize=8)
    ax3.set_ylabel("kestirim hatası [μm]")
    ax3.set_title("C-fit hata muhasebesi (seed 321)\nölçülen ≈ ofset + sızıntı")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("test_orbit_mode_correlation.png", dpi=150)
    print("Figür: test_orbit_mode_correlation.png")
    print(f"Toplam süre: {(time.time()-t0)/60:.1f} dk")


if __name__ == "__main__":
    main()
