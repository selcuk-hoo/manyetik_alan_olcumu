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
     görünür kılınır (aynı ofset → aynı yanlış hedef → güncelleme = 0).

Kestirim genişliği taraması (optimum arama):
  A: k=1..3   (makaledeki güvenli bölge)
  C: k=1..4   (k4 kazancı 2.3, ofset yanlılığı ~1μm — aday tatlı nokta)
  D: k=1..5
  B: k=1..6   (k5,k6 kazancı <1.3 — ofset yanlılığı enjeksiyonu beklenir)

İlk koşumun bulgusu: B (k=1..6) A'dan KÖTÜ (15× vs 24×) — düşük kazançlı
modların kestirimi ofsetle kirlenip hata enjekte ediyor; kazanç
hiyerarşisi doğal düzenleyicidir (regularizer). A'nın tabanı ise
neredeyse tamamen trimlenmeyen k=4 içeriğinden geliyordu → C varyantı.

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
K_CAL        = [1, 2, 3, 4, 5, 6]    # kalibre edilen düğmeler
FITS         = {"A": [1, 2, 3],
                "C": [1, 2, 3, 4],
                "D": [1, 2, 3, 4, 5],
                "B": [1, 2, 3, 4, 5, 6]}
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

    spec_P, by_k_P = spektrum(P, n_q, antisym)

    print(f"Desen: seed={PATTERN_SEED}, RMS={np.std(P)*1e6:.1f}μm | "
          f"BPM ofseti: seed={OFFSET_SEED}, RMS={np.std(b)*1e6:.1f}μm")
    print("Kestirim genişlikleri: " +
          ", ".join(f"{nm}=k1..{max(ks)}" for nm, ks in FITS.items()))

    # ══ BÖLÜM 1+2: kalibrasyon + desen yörüngesi + f0 (tek havuz) ═══════════
    tasks = [("ref", np.zeros(n_q).tolist(), "orbit", T2, 10)]
    for k in K_CAL:
        c, s = mode_vec_pair(n_q, k, antisym)
        tasks.append((f"o{k}c", (A_CAL*c).tolist(), "orbit", T2, 10))
        tasks.append((f"o{k}s", (A_CAL*s).tolist(), "orbit", T2, 10))
    tasks.append(("oP", P.tolist(), "orbit", T2, 10))
    tasks.append(("f0", P.tolist(), "spin", T2, RETURN_STEPS))

    print(f"\nBölüm 1+2: {len(tasks)} simülasyon ({nw} işçi)...")
    res = run(tasks)

    y_ref = np.asarray(res["ref"])
    O_cols, knob_names = [], []
    for k in K_CAL:
        for ph, tag in (("cos", "c"), ("sin", "s")):
            col = (np.asarray(res[f"o{k}{tag}"]) - y_ref) / A_CAL
            O_cols.append(col)
            knob_names.append(f"k{k} {ph}")
    O_full = np.column_stack(O_cols)            # [48 × 12]

    # Yörünge kazançları
    gains = np.sqrt(np.mean(O_full**2, axis=0))
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

    # Gerçek düğme içerikleri
    a_true = []
    for k in K_CAL:
        a_true.append(by_k_P[k].get('cos', 0.0))
        a_true.append(by_k_P[k].get('sin', 0.0))
    a_true = np.array(a_true)

    # ══ BÖLÜM 3: kestirimler (genişlik taraması) ════════════════════════════
    a_hats, eps_preds, P_trims = {}, {}, {}
    for nm, ks in FITS.items():
        ncol = 2 * len(ks)
        O_fit = O_full[:, :ncol]
        a_hat, *_ = np.linalg.lstsq(O_fit, y_meas, rcond=None)
        a_hats[nm] = a_hat
        eps_preds[nm] = np.linalg.pinv(O_fit) @ b
        P_trims[nm] = P - knob_matrix(n_q, ks, antisym) @ a_hat

    print(f"\n{'─'*94}")
    print("Bölüm 3: Kestirim [μm] — gerçek vs varyantlar (parantez: hata)")
    print(f"{'─'*94}")
    hdr = f"{'düğme':>9}  {'gerçek':>8}"
    for nm in FITS:
        hdr += f"  {('kest-'+nm):>16}"
    print(hdr)
    print('─'*94)
    for i, name in enumerate(knob_names):
        row = f"{name:>9}  {a_true[i]*1e6:>8.2f}"
        for nm, ks in FITS.items():
            if i < 2*len(ks):
                err = (a_hats[nm][i] - a_true[i])*1e6
                row += f"  {a_hats[nm][i]*1e6:>8.2f} ({err:>+6.2f})"
            else:
                row += f"  {'—':>16}"
        print(row)
    print()
    for nm, ks in FITS.items():
        err = a_hats[nm] - a_true[:2*len(ks)]
        print(f"  {nm} (k=1..{max(ks)}): kestirim hatası RMS "
              f"{np.std(err)*1e6:>6.2f}μm | O⁺b öngörü RMS "
              f"{np.std(eps_preds[nm])*1e6:>6.2f}μm")

    # ══ BÖLÜM 4: trim + spin doğrulama (tüm varyantlar) ═════════════════════
    print(f"\nBölüm 4: 4 trim varyantı spin doğrulaması + en iyinin "
          f"2. iterasyon yörüngesi...")
    spin_tasks = [(nm, P_trims[nm].tolist(), "spin", T2, RETURN_STEPS)
                  for nm in FITS]
    res2 = run(spin_tasks)
    f_res = {nm: res2[nm] for nm in FITS}

    best = min(FITS, key=lambda nm: abs(f_res[nm]))
    print(f"  En iyi varyant: {best} (k=1..{max(FITS[best])})")

    # 2. iterasyon (statik taban göstergesi) — en iyi varyantta
    res3 = run([("oBest", P_trims[best].tolist(), "orbit", T2, 10)])
    y_meas2 = np.asarray(res3["oBest"]) + b      # AYNI statik ofsetler
    ncol_b = 2 * len(FITS[best])
    a_hat_2, *_ = np.linalg.lstsq(O_full[:, :ncol_b], y_meas2, rcond=None)
    print(f"  2. iterasyon güncellemesi RMS: {np.std(a_hat_2)*1e6:.4f}μm "
          f"(beklenti 0: aynı ofset → aynı yanlış hedef → taban)")

    # Artık spektrumlar
    specs = {nm: spektrum(P_trims[nm], n_q, antisym)[0] for nm in FITS}

    # Spin katsayılarıyla artık öngörüsü (kalibre k=1..6 kısmı)
    f_preds = {}
    ck_path = os.path.join(BASE, "test_b_random_trim.json")
    if os.path.exists(ck_path):
        with open(ck_path) as fh:
            ckd = json.load(fh)
        for nm in FITS:
            _, by_k_t = spektrum(P_trims[nm], n_q, antisym)
            f_preds[nm] = sum(
                ckd["c_cos"][str(k)] * by_k_t[k].get('cos', 0.0)
                + ckd["c_sin"][str(k)] * by_k_t[k].get('sin', 0.0)
                for k in K_CAL)

    print(f"\n{'─'*74}")
    print("Sonuçlar — kestirim genişliği taraması")
    print(f"{'─'*74}")
    print(f"{'varyant':>14}  {'dSy/dt [rad/s]':>15}  {'bastırma':>9}  "
          f"{'k1..6 artık öngörüsü':>21}")
    print('─'*74)
    print(f"{'trim öncesi':>14}  {f0:>15.4e}  {'—':>9}  {'—':>21}")
    for nm, ks in FITS.items():
        pred = f"{f_preds[nm]:+.3e}" if nm in f_preds else "—"
        star = " ★" if nm == best else ""
        print(f"{nm+': k=1..'+str(max(ks)):>14}  {f_res[nm]:>15.4e}  "
              f"{abs(f0/f_res[nm]):>8.1f}×  {pred:>21}{star}")

    print(f"\n{'─'*86}")
    print("Mod içeriği [μm]: önce → A → C → D → B   (←f: fit kapsamında)")
    print(f"{'─'*86}")
    for k in range(1, 13):
        row = f"  k={k:>2}: {spec_P[k][0]*1e6:>7.2f}"
        for nm in FITS:
            row += f" → {specs[nm][k][0]*1e6:>7.2f}"
        tags = "".join(nm for nm, ks in FITS.items() if k in ks)
        print(row + (f"   ←{tags}" if tags else ""))

    # ── JSON ───────────────────────────────────────────────────────────────
    out = {
        "_aciklama": "Yörünge-sürülü trim zinciri, genişlik taraması "
                     "(k-mod yok, CO=False)",
        "pattern_rms_um": PATTERN_RMS*1e6, "offset_rms_um": OFFSET_RMS*1e6,
        "pattern_seed": PATTERN_SEED, "offset_seed": OFFSET_SEED,
        "kazanclar": {knob_names[i]: float(gains[i])
                      for i in range(len(knob_names))},
        "f0": f0,
        "varyantlar": {nm: {
            "k_list": FITS[nm],
            "f": f_res[nm],
            "bastirma": abs(f0/f_res[nm]),
            "f_pred_k16": f_preds.get(nm),
            "kestirim_hata_um": {
                knob_names[i]: float((a_hats[nm][i]-a_true[i])*1e6)
                for i in range(2*len(FITS[nm]))},
            "ofset_onggoru_um": {
                knob_names[i]: float(eps_preds[nm][i]*1e6)
                for i in range(2*len(FITS[nm]))},
        } for nm in FITS},
        "en_iyi": best,
        "iter2_guncelleme_rms_um": float(np.std(a_hat_2)*1e6),
        "mod_icerik_um": {str(k): dict(
            {"once": spec_P[k][0]*1e6},
            **{nm: specs[nm][k][0]*1e6 for nm in FITS})
            for k in range(1, 13)},
        "y_true_bpm": y_true.tolist(),
        "bpm_ofsetleri": b.tolist(),
        "O_matrisi": O_full.tolist(),
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
    idx = np.arange(len(K_CAL))
    ax.bar(idx - 0.2, gains[0::2], 0.4, label='cos', color='tab:red', alpha=0.85)
    ax.bar(idx + 0.2, gains[1::2], 0.4, label='sin', color='tab:blue', alpha=0.85)
    ax.set_xticks(idx); ax.set_xticklabels([f"k={k}" for k in K_CAL])
    ax.set_ylabel("yörünge kazancı (RMS) [m/m]")
    ax.set_title("Mod-yörünge kazançları (rezonans hiyerarşisi)")
    ax.set_yscale('log')
    ax.legend(); ax.grid(True, which='both', axis='y', alpha=0.3)

    # Panel 2: kestirim hatası (B varyantı) vs ofset öngörüsü
    ax = axes[0, 1]
    err_B = a_hats["B"] - a_true
    xi = np.arange(len(knob_names))
    ax.bar(xi - 0.2, err_B*1e6, 0.4, label='gerçek hata (B)',
           color='tab:purple', alpha=0.85)
    ax.bar(xi + 0.2, eps_preds["B"]*1e6, 0.4, label='O⁺b öngörüsü',
           color='tab:gray', alpha=0.85)
    ax.set_xticks(xi); ax.set_xticklabels(knob_names, rotation=45, fontsize=8)
    ax.set_ylabel("kestirim hatası [μm]")
    ax.set_title("Geniş fit (k=1..6): düşük kazançta ofset patlaması")
    ax.legend(); ax.grid(True, axis='y', alpha=0.3)

    # Panel 3: mod içerik öncesi/sonrası (önce, A, C, B)
    ax = axes[1, 0]
    kk = np.arange(1, 13)
    ax.bar(kk - 0.3, [spec_P[k][0]*1e6 for k in kk], 0.2, label='önce',
           color='tab:gray', alpha=0.85)
    ax.bar(kk - 0.1, [specs["A"][k][0]*1e6 for k in kk], 0.2, label='A: k=1..3',
           color='tab:orange', alpha=0.85)
    ax.bar(kk + 0.1, [specs["C"][k][0]*1e6 for k in kk], 0.2, label='C: k=1..4',
           color='tab:green', alpha=0.85)
    ax.bar(kk + 0.3, [specs["B"][k][0]*1e6 for k in kk], 0.2, label='B: k=1..6',
           color='tab:red', alpha=0.85)
    ax.set_xticks(kk)
    ax.set_xlabel("Fourier modu k"); ax.set_ylabel("A_k [μm]")
    ax.set_title("Mod içeriği: trim öncesi/sonrası")
    ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.3)

    # Panel 4: sahte EDM genişlik taraması
    ax = axes[1, 1]
    labels = ["önce"] + [f"{nm}: k≤{max(ks)}" for nm, ks in FITS.items()]
    vals = [abs(f0)] + [abs(f_res[nm]) for nm in FITS]
    cols = ['tab:gray', 'tab:orange', 'tab:green', 'tab:cyan', 'tab:red']
    bars = ax.bar(labels, vals, color=cols, alpha=0.85)
    ax.set_yscale('log')
    for i, v in enumerate(vals):
        txt = "—" if i == 0 else f"{abs(f0)/v:.0f}×"
        ax.annotate(txt, (i, v), ha='center', va='bottom', fontsize=10)
    ax.set_ylabel("|dSy/dt| [rad/s]")
    ax.set_title("Sahte EDM vs kestirim genişliği (spin doğrulaması)")
    ax.grid(True, which='both', axis='y', alpha=0.3)

    plt.tight_layout()
    fout = "test_orbit_trim.png"
    plt.savefig(fout, dpi=150)
    print(f"Figür kaydedildi: {fout}")


if __name__ == "__main__":
    main()
