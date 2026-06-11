#!/usr/bin/env python3
"""test_orbit_trim_seeds.py — Yörünge-trim genişlik haritasının evrenselliği.

SORU: "C (k=1..4) en iyi, D/B (k≥5 fit) zararlı" bulgusu seed=321/777'ye
özgü bir rastlantı mı, yoksa evrensel mi?

YÖNTEM: Kalibrasyon (O matrisi) kafese aittir, desene değil —
test_orbit_trim.json'dan yeniden kullanılır. 4 yeni (desen, ofset) seed
çifti üretilir; her biri için:
  1. Desenin yörüngesi ölçülür (1 orbit koşumu), ofsetler eklenir,
  2. Dört genişlik varyantı (A: k≤3, C: k≤4, D: k≤5, B: k≤6) ile LSQ
     kestirim + trim hesaplanır,
  3. Trim öncesi/sonrası spin takibi → bastırma oranları.

BEKLENTİ (evrensellik hipotezi): her seed'de C ≥ A ve C > D, C > B.
Tek tek bastırma oranları seed'e göre değişebilir (desenin k≥5 içeriği
ve ofsetlerin izdüşümü rastgele), ama SIRALAMA korunmalıdır.

Çıktı: test_orbit_trim_seeds.json, test_orbit_trim_seeds.png
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

from test_orbit_trim import (_worker, knob_matrix, spektrum,
                             FITS, K_CAL, T2, RETURN_STEPS,
                             PATTERN_RMS, OFFSET_RMS)

# Yeni seed çiftleri (desen, ofset) — orijinal (321, 777) idi
SEED_PAIRS = [(101, 201), (102, 202), (103, 203), (104, 204)]


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

    def run(tasks):
        with ctx.Pool(processes=min(nw, len(tasks))) as pool:
            return dict(pool.map(_worker, tasks))

    # Kalibrasyon bazını önceki koşumdan al (kafese ait, desene değil)
    with open("test_orbit_trim.json") as fh:
        prev = json.load(fh)
    O_full = np.array(prev["O_matrisi"])          # [48 × 12]
    gains  = np.sqrt(np.mean(O_full**2, axis=0))
    knob_names = list(prev["kazanclar"].keys())
    print(f"Kalibrasyon bazı yüklendi: O {O_full.shape}, "
          f"kazançlar k2={gains[2]:.1f} k4={gains[6]:.2f} k6={gains[10]:.2f}")

    # Desenler ve ofsetler
    patterns, offsets = {}, {}
    for ps, bs in SEED_PAIRS:
        patterns[ps] = np.random.default_rng(ps).standard_normal(n_q) * PATTERN_RMS
        offsets[ps]  = np.random.default_rng(bs).standard_normal(n_q) * OFFSET_RMS

    # ── Tur 1: desen yörüngeleri + trim-öncesi spin (8 görev) ──────────────
    tasks = []
    for ps, _ in SEED_PAIRS:
        tasks.append((f"orb_{ps}", patterns[ps].tolist(), "orbit", T2, 10))
        tasks.append((f"f0_{ps}",  patterns[ps].tolist(), "spin",  T2, RETURN_STEPS))
    print(f"\nTur 1: {len(tasks)} simülasyon ({nw} işçi)...")
    res1 = run(tasks)

    # ── Kestirimler ve trim desenleri ───────────────────────────────────────
    trims = {}     # (ps, varyant) → P_trim
    a_hats = {}
    for ps, _ in SEED_PAIRS:
        y_meas = np.asarray(res1[f"orb_{ps}"]) + offsets[ps]
        for nm, ks in FITS.items():
            ncol = 2 * len(ks)
            a_hat, *_ = np.linalg.lstsq(O_full[:, :ncol], y_meas, rcond=None)
            a_hats[(ps, nm)] = a_hat
            trims[(ps, nm)] = patterns[ps] - knob_matrix(n_q, ks, antisym) @ a_hat

    # ── Tur 2: trim-sonrası spin (16 görev) ────────────────────────────────
    tasks = [(f"f_{ps}_{nm}", trims[(ps, nm)].tolist(), "spin", T2, RETURN_STEPS)
             for ps, _ in SEED_PAIRS for nm in FITS]
    print(f"Tur 2: {len(tasks)} simülasyon...")
    res2 = run(tasks)

    # ── Sonuç tablosu ───────────────────────────────────────────────────────
    out = {"_aciklama": "Genişlik haritasının seed evrenselliği "
                        "(kalibrasyon test_orbit_trim.json'dan)",
           "seed_pairs": SEED_PAIRS, "varyantlar": list(FITS),
           "sonuclar": {}}

    print(f"\n{'─'*78}")
    print(f"{'seed':>10} {'f0 [rad/s]':>13}" +
          "".join(f"  {nm+' bastırma':>12}" for nm in FITS))
    print(f"{'─'*78}")
    ratios = {nm: [] for nm in FITS}
    for ps, bs in SEED_PAIRS:
        f0 = res1[f"f0_{ps}"]
        row = {"f0": f0, "bastirma": {}, "f_sonra": {}}
        line = f"{ps}/{bs:>5} {f0:>+13.3e}"
        for nm in FITS:
            f_after = res2[f"f_{ps}_{nm}"]
            ratio = abs(f0 / f_after) if f_after != 0 else np.inf
            row["f_sonra"][nm] = f_after
            row["bastirma"][nm] = ratio
            ratios[nm].append(ratio)
            line += f"  {ratio:>11.1f}×"
        out["sonuclar"][str(ps)] = row
        print(line)

    print(f"{'─'*78}")
    line = f"{'geo-ort':>10} {'':>13}"
    for nm in FITS:
        gm = float(np.exp(np.mean(np.log(ratios[nm]))))
        line += f"  {gm:>11.1f}×"
        out.setdefault("geo_ortalama", {})[nm] = gm
    print(line)

    # Sıralama kontrolü: her seed'de C, D ve B'den iyi mi?
    n_c_best = sum(1 for ps, _ in SEED_PAIRS
                   if out["sonuclar"][str(ps)]["bastirma"]["C"] >=
                      max(out["sonuclar"][str(ps)]["bastirma"][v]
                          for v in ("D", "B")))
    n_c_vs_a = sum(1 for ps, _ in SEED_PAIRS
                   if out["sonuclar"][str(ps)]["bastirma"]["C"] >=
                      out["sonuclar"][str(ps)]["bastirma"]["A"])
    out["C_DB_den_iyi"] = f"{n_c_best}/{len(SEED_PAIRS)}"
    out["C_A_dan_iyi"]  = f"{n_c_vs_a}/{len(SEED_PAIRS)}"
    print(f"\nC ≥ max(D,B): {n_c_best}/{len(SEED_PAIRS)} seed'de")
    print(f"C ≥ A       : {n_c_vs_a}/{len(SEED_PAIRS)} seed'de")

    with open("test_orbit_trim_seeds.json", "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("\nKaydedildi: test_orbit_trim_seeds.json")

    # ── Figür ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(FITS))
    width = 0.18
    for i, (ps, bs) in enumerate(SEED_PAIRS):
        vals = [out["sonuclar"][str(ps)]["bastirma"][nm] for nm in FITS]
        ax.bar(x + (i - 1.5) * width, vals, width, label=f"seed {ps}/{bs}")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{nm}\n(k≤{max(ks)})" for nm, ks in FITS.items()])
    ax.set_ylabel("Bastırma oranı |f₀/f|")
    ax.set_title("Genişlik haritasının seed evrenselliği "
                 f"(desen {PATTERN_RMS*1e6:.0f}μm, ofset {OFFSET_RMS*1e6:.0f}μm RMS)")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("test_orbit_trim_seeds.png", dpi=150)
    print("Figür: test_orbit_trim_seeds.png")
    print(f"Toplam süre: {(time.time()-t0)/60:.1f} dk")


if __name__ == "__main__":
    main()
