#!/usr/bin/env python3
"""test_orbit_trim.py — Yörünge-sürülü (BPM-referanslı) trim: uçtan uca zincir.

Senaryo (k-mod YOK, tek optik konfigürasyon):
  48 kuadrupolde rastgele dikey hizalama hatası, RMS = 100 μm.
  48 BPM'de statik ofset, RMS = 100 μm (her BPM bir quad girişinde).
  Hedef: yörünge ölçümünden hizalama modlarını kestir, trimle,
  etkin mod içeriğini birkaç mikrona indir. Doğrulama SPİN takibiyle.

Yöntemin üç adımı:
  1. KALİBRASYON: her mod-kuadratür düğmesi (k=1..6 × cos/sin) için
     bilinen genlikte desen uygula, tur-ortalamalı yörüngeyi oku
     → yörünge tepki bazı O [48 × 12]. (Gerçekte de diferansiyeldir:
     bilinen trim değişimine yörünge DEĞİŞİMİ ölçülür, statik ofset
     farkta iptal olur — kalibrasyon ofsetten etkilenmez.)
  2. ÖLÇÜM+KESTİRİM: desenin yörüngesi ölçülür (y_meas = y_true + b),
     LSQ ile O bazına oturtulur → mod genlik kestirimleri â.
     Statik ofsetin kaçınılmaz yanlılığı: ε = O⁺·b (mod başına ~σ_b/kazanç).
  3. TRİM+DOĞRULAMA: P_trim = P − Σ â·mod. Spin takibiyle sahte EDM
     öncesi/sonrası ölçülür. İkinci iterasyonla statik-ofset TABANI
     görünür kılınır (aynı ofset → aynı yanlış hedef → iyileşme durur).

İki kestirim genişliği yarıştırılır:
  Fit-A: k=1..3 (makaledeki güvenli bölge)
  Fit-B: k=1..6 (kazanç düştükçe ofset yanlılığı büyür — sınır testi)

Tüm koşullar CO=False (eksen fırlatma, kapalı yörünge aranmaz);
yörünge = tur-ortalaması (C++ cod_data.txt), spin = Poincaré Sy eğimi.
"""

import json
import os
import shutil
import sys
import tempfile
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
sys.path.insert(0, BASE)

from fourier_reconstruct import fodo_basis

# ── Parametreler ─────────────────────────────────────────────────────────────
PATTERN_RMS  = 1e-4      # rastgele hizalama hatası RMS [m] = 100 μm
PATTERN_SEED = 321
OFFSET_RMS   = 1e-4      # statik BPM ofseti RMS [m] = 100 μm
OFFSET_SEED  = 777
A_CAL        = 5e-5      # kalibrasyon mod genliği [m] = 50 μm
K_FIT_A      = [1, 2, 3]             # dar kestirim (makale bölgesi)
K_FIT_B      = [1, 2, 3, 4, 5, 6]    # geniş kestirim
T2           = 1e-3
RETURN_STEPS = 6000      # spin koşumları için


def _suppress_stdout():
    fd = os.dup(1)
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1)
    os.close(null)
    return fd


def _restore_stdout(fd):
    os.dup2(fd, 1)
    os.close(fd)


def _worker(task):
    """Tek koşum. mode='orbit': 48 BPM tur-ortalamalı y; mode='spin': dSy/dt.

    Her görev kendi geçici dizininde çalışır (cod_data.txt çakışmasın).
    """
    label, dy_list, mode, t2, return_steps = task

    import os, sys, json, tempfile, shutil
    import numpy as np
    sys.path.insert(0, BASE)
    tmp = tempfile.mkdtemp(prefix=f"otrim_{os.getpid()}_")
    os.chdir(tmp)

    from false_edm_mode_scan import setup_fields
    from integrator import integrate_particle

    with open(os.path.join(BASE, "params.json")) as fh:
        config = json.load(fh)
    fields, y0, beta0, R0, p_mag, direction = setup_fields(config)
    dt = float(config.get("dt", 1e-11))
    dy = np.asarray(dy_list, dtype=float)

    saved = _suppress_stdout()
    try:
        if mode == "orbit":
            fields.poincare_quad_index = 999.0
            integrate_particle(y0, 0.0, t2, dt, fields=fields,
                               return_steps=10, quad_dy=dy)
            cd = np.loadtxt("cod_data.txt", skiprows=1)
            cd[:, 1:3] *= 1e-3            # mm → m
            n = int(fields.nFODO)
            y_bpm = np.empty(2 * n)
            for k in range(n):
                y_bpm[2*k]     = cd[k*8 + 2, 2]   # QF girişi
                y_bpm[2*k + 1] = cd[k*8 + 6, 2]   # QD girişi
            result = y_bpm.tolist()
        else:
            fields.poincare_quad_index = 0.0
            _, poin, poin_t = integrate_particle(
                y0, 0.0, t2, dt, fields=fields,
                return_steps=return_steps, quad_dy=dy)
            result = float(np.polyfit(np.asarray(poin_t, float),
                                      np.asarray(poin[:, 7], float), 1)[0])
    finally:
        _restore_stdout(saved)
        os.chdir(BASE)
        shutil.rmtree(tmp, ignore_errors=True)

    return label, result


def mode_vec_pair(n_q, k, antisym):
    """k modunun (cos, sin) birim desen çifti."""
    Fk, _ = fodo_basis(n_q, [k], antisym)
    return Fk[:, 0], Fk[:, 1]


def knob_matrix(n_q, k_list, antisym):
    """Kaçıklık-uzayı düğme matrisi [48 × 2·len(k_list)]; sütun sırası
    (k1,cos),(k1,sin),(k2,cos),... — kalibrasyon etiketleriyle aynı."""
    cols = []
    for k in k_list:
        c, s = mode_vec_pair(n_q, k, antisym)
        cols.append(c)
        cols.append(s)
    return np.column_stack(cols)


def spektrum(P, n_q, antisym, kmax=12):
    """Desenin gerçek mod spektrumu: k → (A_k μm, φ_k °)."""
    F_full, meta = fodo_basis(n_q, list(range(0, kmax + 1)), antisym)
    coef, *_ = np.linalg.lstsq(F_full, P, rcond=None)
    out = {}
    by_k = {}
    for c, (k, kind) in zip(coef, meta):
        by_k.setdefault(k, {})[kind] = c
    for k, d in by_k.items():
        if 'dc' in d:
            out[k] = (abs(d['dc']), 0.0)
        else:
            ac, as_ = d.get('cos', 0.0), d.get('sin', 0.0)
            out[k] = (float(np.hypot(ac, as_)),
                      float(np.degrees(np.arctan2(as_, ac))))
    return out, by_k


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

    # Desen ve BPM ofsetleri
    rng_p = np.random.default_rng(PATTERN_SEED)
    P = rng_p.standard_normal(n_q) * PATTERN_RMS
    rng_b = np.random.default_rng(OFFSET_SEED)
    b = rng_b.standard_normal(n_q) * OFFSET_RMS

    spec_P, _ = spektrum(P, n_q, antisym)

    print(f"Desen: seed={PATTERN_SEED}, RMS={np.std(P)*1e6:.1f}μm | "
          f"BPM ofseti: seed={OFFSET_SEED}, RMS={np.std(b)*1e6:.1f}μm")
    print(f"Kestirimler: A=k{K_FIT_A}, B=k{K_FIT_B} | kalibrasyon {A_CAL*1e6:.0f}μm")

    # ══ BÖLÜM 1+2: kalibrasyon + desen yörüngesi + f0 (tek havuz) ═══════════
    tasks = [("ref", np.zeros(n_q).tolist(), "orbit", T2, 10)]
    for k in K_FIT_B:
        c, s = mode_vec_pair(n_q, k, antisym)
        tasks.append((f"o{k}c", (A_CAL*c).tolist(), "orbit", T2, 10))
        tasks.append((f"o{k}s", (A_CAL*s).tolist(), "orbit", T2, 10))
    tasks.append(("oP", P.tolist(), "orbit", T2, 10))
    tasks.append(("f0", P.tolist(), "spin", T2, RETURN_STEPS))

    print(f"\nBölüm 1+2: {len(tasks)} simülasyon ({nw} işçi)...")
    res = run(tasks)

    y_ref = np.asarray(res["ref"])
    O_cols, knob_names = [], []
    for k in K_FIT_B:
        for ph, tag in (("cos", "c"), ("sin", "s")):
            col = (np.asarray(res[f"o{k}{tag}"]) - y_ref) / A_CAL
            O_cols.append(col)
            knob_names.append(f"k{k} {ph}")
    O_B = np.column_stack(O_cols)            # [48 × 12]
    nA  = 2 * len(K_FIT_A)
    O_A = O_B[:, :nA]                        # k=1..3 alt bloğu

    # Yörünge kazançları
    gains = np.sqrt(np.mean(O_B**2, axis=0))   # RMS kazanç [m yörünge / m kaçıklık]
    print(f"\n{'─'*60}")
    print("Bölüm 1: Yörünge kazançları (RMS, boyutsuz)")
    print(f"{'─'*60}")
    for i, name in enumerate(knob_names):
        print(f"  {name:>8}: {gains[i]:>8.2f}")

    # Desen yörüngesi + ofsetler
    y_true = np.asarray(res["oP"])
    y_meas = y_true + b
    f0 = res["f0"]
    print(f"\nDesen yörüngesi RMS: gerçek {np.std(y_true)*1e3:.3f}mm, "
          f"ölçülen (ofsetli) {np.std(y_meas)*1e3:.3f}mm")
    print(f"Sahte EDM (trim öncesi): f0 = {f0:+.4e} rad/s")

    # ══ BÖLÜM 3: kestirim ═══════════════════════════════════════════════════
    a_hat_A, *_ = np.linalg.lstsq(O_A, y_meas, rcond=None)
    a_hat_B, *_ = np.linalg.lstsq(O_B, y_meas, rcond=None)

    # Gerçek mod içerikleri (kıyas için)
    _, by_k = spektrum(P, n_q, antisym)
    a_true_B = []
    for k in K_FIT_B:
        a_true_B.append(by_k[k].get('cos', 0.0))
        a_true_B.append(by_k[k].get('sin', 0.0))
    a_true_B = np.array(a_true_B)

    # Ofset yanlılığı öngörüsü: ε = O⁺ b
    eps_pred_A = np.linalg.pinv(O_A) @ b
    eps_pred_B = np.linalg.pinv(O_B) @ b

    print(f"\n{'─'*78}")
    print("Bölüm 3: Kestirim — gerçek içerik vs LSQ kestirimi [μm]")
    print(f"{'─'*78}")
    print(f"{'düğme':>9}  {'gerçek':>9}  {'kest-A':>9}  {'kest-B':>9}  "
          f"{'hata-B':>9}  {'O⁺b öngörü':>11}")
    print('─'*78)
    for i, name in enumerate(knob_names):
        kA = f"{a_hat_A[i]*1e6:>9.2f}" if i < nA else f"{'—':>9}"
        print(f"{name:>9}  {a_true_B[i]*1e6:>9.2f}  {kA}  "
              f"{a_hat_B[i]*1e6:>9.2f}  "
              f"{(a_hat_B[i]-a_true_B[i])*1e6:>9.2f}  "
              f"{eps_pred_B[i]*1e6:>11.2f}")
    err_A = a_hat_A - a_true_B[:nA]
    err_B = a_hat_B - a_true_B
    print(f"\n  Kestirim hatası RMS: A {np.std(err_A)*1e6:.2f}μm, "
          f"B {np.std(err_B)*1e6:.2f}μm")
    print(f"  O⁺b öngörü RMS    : A {np.std(eps_pred_A)*1e6:.2f}μm, "
          f"B {np.std(eps_pred_B)*1e6:.2f}μm")

    # ══ BÖLÜM 4: trim + spin doğrulama ══════════════════════════════════════
    F_A = knob_matrix(n_q, K_FIT_A, antisym)
    F_B = knob_matrix(n_q, K_FIT_B, antisym)
    P_A = P - F_A @ a_hat_A
    P_B = P - F_B @ a_hat_B

    print(f"\nBölüm 4: trim uygulandı; spin doğrulaması + B'nin 2. iterasyon "
          f"yörüngesi (3 simülasyon)...")
    res2 = run([("fA", P_A.tolist(), "spin", T2, RETURN_STEPS),
                ("fB", P_B.tolist(), "spin", T2, RETURN_STEPS),
                ("oB", P_B.tolist(), "orbit", T2, 10)])
    fA, fB = res2["fA"], res2["fB"]

    # ── 2. iterasyon (statik taban göstergesi) ────────────────────────────
    y_meas2 = np.asarray(res2["oB"]) + b      # AYNI statik ofsetler
    a_hat_2, *_ = np.linalg.lstsq(O_B, y_meas2, rcond=None)
    P_B2 = P_B - F_B @ a_hat_2
    print(f"  2. iterasyon kestirimi RMS: {np.std(a_hat_2)*1e6:.2f}μm "
          f"(beklenti ~0: aynı ofset aynı yanlış hedef)")

    res3 = run([("fB2", P_B2.tolist(), "spin", T2, RETURN_STEPS)])
    fB2 = res3["fB2"]

    # ── Artık spektrum ve sahte EDM öngörüsü ──────────────────────────────
    spec_A, _ = spektrum(P_A, n_q, antisym)
    spec_B, _ = spektrum(P_B, n_q, antisym)
    spec_B2, _ = spektrum(P_B2, n_q, antisym)

    # Spin katsayılarıyla artık öngörüsü (k=1..6, çift kuadratür)
    f_pred_B = None
    ck_path = os.path.join(BASE, "test_b_random_trim.json")
    if os.path.exists(ck_path):
        with open(ck_path) as fh:
            ckd = json.load(fh)
        _, by_k_B = spektrum(P_B, n_q, antisym)
        f_pred_B = sum(ckd["c_cos"][str(k)] * by_k_B[k].get('cos', 0.0)
                       + ckd["c_sin"][str(k)] * by_k_B[k].get('sin', 0.0)
                       for k in K_FIT_B)

    print(f"\n{'─'*70}")
    print("Sonuçlar")
    print(f"{'─'*70}")
    print(f"{'durum':>22}  {'dSy/dt [rad/s]':>15}  {'bastırma':>9}")
    print('─'*70)
    print(f"{'trim öncesi':>22}  {f0:>15.4e}  {'—':>9}")
    print(f"{'A: k=1..3 trim':>22}  {fA:>15.4e}  {abs(f0/fA):>8.1f}×")
    print(f"{'B: k=1..6 trim':>22}  {fB:>15.4e}  {abs(f0/fB):>8.1f}×")
    print(f"{'B + 2. iterasyon':>22}  {fB2:>15.4e}  {abs(f0/fB2):>8.1f}×")
    if f_pred_B is not None:
        print(f"\n  B artık öngörüsü (artık spektrum × c_k, k=1..6): "
              f"{f_pred_B:+.4e} rad/s")

    print(f"\n{'─'*70}")
    print("Mod içeriği [μm]: önce → A-trim → B-trim → B-iter2")
    print(f"{'─'*70}")
    for k in range(1, 13):
        a0 = spec_P[k][0]*1e6
        aA = spec_A[k][0]*1e6
        aB = spec_B[k][0]*1e6
        a2 = spec_B2[k][0]*1e6
        tag = " ←fit-B" if k in K_FIT_B else ""
        tagA = "/A" if k in K_FIT_A else ""
        print(f"  k={k:>2}: {a0:>7.2f} → {aA:>7.2f} → {aB:>7.2f} → "
              f"{a2:>7.2f}{tag}{tagA}")
    rms_fit_B  = np.sqrt(np.mean([spec_B[k][0]**2 for k in K_FIT_B]))
    rms_fit_B0 = np.sqrt(np.mean([spec_P[k][0]**2 for k in K_FIT_B]))
    print(f"\n  Fit-B modlarının RMS içeriği: {rms_fit_B0*1e6:.1f}μm → "
          f"{rms_fit_B*1e6:.2f}μm")

    # ── JSON ───────────────────────────────────────────────────────────────
    out = {
        "_aciklama": "Yörünge-sürülü trim zinciri (k-mod yok, CO=False)",
        "pattern_rms_um": PATTERN_RMS*1e6, "offset_rms_um": OFFSET_RMS*1e6,
        "pattern_seed": PATTERN_SEED, "offset_seed": OFFSET_SEED,
        "kazanclar": {knob_names[i]: float(gains[i])
                      for i in range(len(knob_names))},
        "kestirim_hata_um_B": {knob_names[i]: float(err_B[i]*1e6)
                               for i in range(len(knob_names))},
        "ofset_onggoru_um_B": {knob_names[i]: float(eps_pred_B[i]*1e6)
                               for i in range(len(knob_names))},
        "f0": f0, "fA": fA, "fB": fB, "fB_iter2": fB2,
        "f_pred_B": f_pred_B,
        "bastirma_A": abs(f0/fA), "bastirma_B": abs(f0/fB),
        "bastirma_B2": abs(f0/fB2),
        "mod_icerik_um": {str(k): {"once": spec_P[k][0]*1e6,
                                   "A": spec_A[k][0]*1e6,
                                   "B": spec_B[k][0]*1e6,
                                   "B2": spec_B2[k][0]*1e6}
                          for k in range(1, 13)},
    }
    with open("test_orbit_trim.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nKaydedildi: test_orbit_trim.json")
    print(f"Toplam süre: {time.time()-t0:.1f} s")

    # ══ Figür ════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Yörünge-sürülü trim: desen RMS={PATTERN_RMS*1e6:.0f}μm, "
                 f"BPM ofseti={OFFSET_RMS*1e6:.0f}μm (k-mod YOK)", fontsize=13)

    # Panel 1: yörünge kazançları
    ax = axes[0, 0]
    idx = np.arange(len(K_FIT_B))
    ax.bar(idx - 0.2, gains[0::2], 0.4, label='cos', color='tab:red', alpha=0.85)
    ax.bar(idx + 0.2, gains[1::2], 0.4, label='sin', color='tab:blue', alpha=0.85)
    ax.set_xticks(idx); ax.set_xticklabels([f"k={k}" for k in K_FIT_B])
    ax.set_ylabel("yörünge kazancı (RMS) [m/m]")
    ax.set_title("Mod-yörünge kazançları")
    ax.legend(); ax.grid(True, axis='y', alpha=0.3)

    # Panel 2: kestirim hatası vs ofset öngörüsü
    ax = axes[0, 1]
    xi = np.arange(len(knob_names))
    ax.bar(xi - 0.2, err_B*1e6, 0.4, label='gerçek hata', color='tab:purple',
           alpha=0.85)
    ax.bar(xi + 0.2, eps_pred_B*1e6, 0.4, label='O⁺b öngörüsü',
           color='tab:gray', alpha=0.85)
    ax.set_xticks(xi); ax.set_xticklabels(knob_names, rotation=45, fontsize=8)
    ax.set_ylabel("kestirim hatası [μm]")
    ax.set_title("Kestirim hatası kaynağı: statik BPM ofseti")
    ax.legend(); ax.grid(True, axis='y', alpha=0.3)

    # Panel 3: mod içerik öncesi/sonrası
    ax = axes[1, 0]
    kk = np.arange(1, 13)
    ax.bar(kk - 0.3, [spec_P[k][0]*1e6 for k in kk], 0.3, label='önce',
           color='tab:gray', alpha=0.85)
    ax.bar(kk,       [spec_A[k][0]*1e6 for k in kk], 0.3, label='A: k=1..3',
           color='tab:orange', alpha=0.85)
    ax.bar(kk + 0.3, [spec_B[k][0]*1e6 for k in kk], 0.3, label='B: k=1..6',
           color='tab:green', alpha=0.85)
    ax.set_xticks(kk)
    ax.set_xlabel("Fourier modu k"); ax.set_ylabel("A_k [μm]")
    ax.set_title("Mod içeriği: trim öncesi/sonrası")
    ax.legend(); ax.grid(True, axis='y', alpha=0.3)

    # Panel 4: sahte EDM
    ax = axes[1, 1]
    labels = ["önce", "A: k=1..3", "B: k=1..6", "B iter-2"]
    vals = [abs(f0), abs(fA), abs(fB), abs(fB2)]
    cols = ['tab:gray', 'tab:orange', 'tab:green', 'tab:olive']
    ax.bar(labels, vals, color=cols, alpha=0.85)
    ax.set_yscale('log')
    if f_pred_B is not None:
        ax.axhline(abs(f_pred_B), color='k', ls='--', lw=1.2,
                   label=f'B öngörüsü (artık spektrum × c_k)')
        ax.legend(fontsize=9)
    for i, v in enumerate(vals):
        sup = abs(f0)/v if v > 0 else float('inf')
        txt = "—" if i == 0 else f"{sup:.0f}×"
        ax.annotate(txt, (i, v), ha='center', va='bottom', fontsize=10)
    ax.set_ylabel("|dSy/dt| [rad/s]")
    ax.set_title("Sahte EDM: spin doğrulaması")
    ax.grid(True, which='both', axis='y', alpha=0.3)

    plt.tight_layout()
    fout = "test_orbit_trim.png"
    plt.savefig(fout, dpi=150)
    print(f"Figür kaydedildi: {fout}")


if __name__ == "__main__":
    main()
