#!/usr/bin/env python3
"""test_symm_basis_fit.py — Genişletilmiş baz (antisim + simetrik mod) ile trim testi.

ARKA PLAN:
  test_orbit_trim.py'nin C varyantı (k=1..4 antisimetrik) sahte EDM'yi
  ~100× bastırır (f0=-1.62e-3 → f_C=+1.6e-5 rad/s). Ancak simetrik
  hizalama bileşenleri (23 boyutlu tümleme) yörüngede ~165 μm COD_rms
  üretiyor (kazanç ~3.0) — antisimetrik baz bunları göremediği için
  ~1e-4 rad/s taban kalıyor.

HEDEF:
  Antisimetrik k=1..4 modlarına EK OLARAK simetrik k=1..4 modlarını
  kalibrasyon bazına ekle → 16 boyutlu genişletilmiş O matrisi →
  simetrik içeriği de trimle → tabanın 10^-5 seviyesine düşüp düşmediğini
  sına.

SİMETRİK MOD TANIMI:
  fodo_basis(n_q, k, antisym=False) → QF ve QD aynı işaret, aynı faz
  (antisym=True'da (-1)^j ile işaret değiştiriyordu).

YÖNTEM:
  1. KALİBRASYON: antisim k=1..4 → test_orbit_trim.json'dan yeniden kullan
                  simetrik k=1..4 → 9 yeni simülasyon (ref + 2×4 mod)
  2. O_ext = [O_antisim | O_symm]  (48×16)
  3. ÖLÇÜM: seed=321 deseni + BPM ofseti (test_orbit_trim ile aynı)
  4. FİT: â = pinv(O_ext) @ (y_meas - y_ref)
  5. TRİM: dy_trim = P - knob_ext @ â
  6. DOĞRULAMA: spin takibiyle dSy/dt

KARŞILAŞTIRMA:
  trim öncesi | antisim-C (test_orbit_trim.json) | genişletilmiş baz

Çıktı: test_symm_basis_fit.json, test_symm_basis_fit.png
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

# ── Parametreler (test_orbit_trim.py ile aynı) ────────────────────────────────
PATTERN_RMS  = 1e-4
PATTERN_SEED = 321
OFFSET_RMS   = 1e-4
OFFSET_SEED  = 777
A_CAL        = 5e-5      # kalibrasyon genliği [m] = 50 μm
K_FIT        = [1, 2, 3, 4]  # antisim ve simetrik için aynı k listesi
T2           = 1e-3
RETURN_STEPS = 6000

# Ek seed'ler için çift: (PATTERN_SEED, OFFSET_SEED)
EXTRA_SEEDS  = [(101, 201), (102, 202), (103, 203), (104, 204)]


# ── Stdout bastırıcı ─────────────────────────────────────────────────────────
def _suppress_stdout():
    fd = os.dup(1)
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1)
    os.close(null)
    return fd


def _restore_stdout(fd):
    os.dup2(fd, 1)
    os.close(fd)


# ── İşçi fonksiyon ───────────────────────────────────────────────────────────
def _worker(task):
    """Tek koşum: mode='orbit' → 48 BPM y; mode='spin' → dSy/dt."""
    label, dy_list, mode, t2, return_steps = task

    import os, sys, json, tempfile, shutil
    import numpy as np
    sys.path.insert(0, BASE)
    tmp = tempfile.mkdtemp(prefix=f"sfit_{os.getpid()}_")
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
            fields.EDMSwitch = 0.0
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


# ── Mod vektörü çiftleri ─────────────────────────────────────────────────────
def mode_vec_pair_antisym(n_q, k):
    """Antisimetrik k-modu: QF ve QD zıt işaretli (klasik FODO)."""
    Fk, _ = fodo_basis(n_q, [k], antisym=True)
    return Fk[:, 0], Fk[:, 1]


def mode_vec_pair_symm(n_q, k):
    """Simetrik k-modu: QF ve QD aynı işaretli.

    fodo_basis(n_q, [k], antisym=False) kullanır.
    antisym=False → (-1)^j faktörü YOK → her FODO çiftinde QF ve QD
    aynı cos/sin değeri alır.
    """
    Fk, _ = fodo_basis(n_q, [k], antisym=False)
    return Fk[:, 0], Fk[:, 1]


def knob_matrix(n_q, k_list, antisym):
    """Hata-uzayı düğme matrisi [48 × 2·len(k_list)]."""
    cols = []
    for k in k_list:
        if antisym:
            c, s = mode_vec_pair_antisym(n_q, k)
        else:
            c, s = mode_vec_pair_symm(n_q, k)
        cols.append(c)
        cols.append(s)
    return np.column_stack(cols)


def knob_matrix_extended(n_q, k_list):
    """Genişletilmiş [antisim | simetrik] düğme matrisi [48 × 4·len(k_list)]."""
    cols = []
    for k in k_list:
        c, s = mode_vec_pair_antisym(n_q, k)
        cols.append(c); cols.append(s)
    for k in k_list:
        c, s = mode_vec_pair_symm(n_q, k)
        cols.append(c); cols.append(s)
    return np.column_stack(cols)


# ── Tek-seed analizi ──────────────────────────────────────────────────────────
def run_one_seed(pattern_seed, offset_seed, n_q, ctx, nw,
                 y_ref, O_antisim, f0_override=None,
                 f_antisim_C_override=None,
                 verbose=True):
    """Tek seed için:
      1. Simetrik kalibrasyon simülasyonları (ref yeniden kullanılır)
      2. Desen yörüngesi + f0 spin ölçümü
      3. Antisim-C trim (test_orbit_trim.json) referansı
      4. Genişletilmiş baz trim + spin doğrulaması
    Döndürür: dict(f0, f_antisim_C, f_ext)
    """
    rng_p = np.random.default_rng(pattern_seed)
    P = rng_p.standard_normal(n_q) * PATTERN_RMS
    rng_b = np.random.default_rng(offset_seed)
    b = rng_b.standard_normal(n_q) * OFFSET_RMS

    def run(tasks):
        with ctx.Pool(processes=min(nw, len(tasks))) as pool:
            return dict(pool.map(_worker, tasks))

    # ── Simetrik kalibrasyon simülasyonları ──────────────────────────────────
    symm_tasks = [("sref", np.zeros(n_q).tolist(), "orbit", T2, 10)]
    for k in K_FIT:
        c, s = mode_vec_pair_symm(n_q, k)
        symm_tasks.append((f"sc{k}c", (A_CAL * c).tolist(), "orbit", T2, 10))
        symm_tasks.append((f"sc{k}s", (A_CAL * s).tolist(), "orbit", T2, 10))

    # Desen yörüngesi + f0 spin (sadece ref olmayan seed'ler için gerekli)
    need_f0 = (f0_override is None)
    symm_tasks.append(("oP", P.tolist(), "orbit", T2, 10))
    if need_f0:
        symm_tasks.append(("f0", P.tolist(), "spin", T2, RETURN_STEPS))

    if verbose:
        print(f"\nSimetrik kalibrasyon + desen ölçümü: {len(symm_tasks)} sim...")

    sres = run(symm_tasks)

    # Simetrik O sütunları
    y_sref = np.asarray(sres["sref"])
    O_symm_cols = []
    for k in K_FIT:
        for tag in ("c", "s"):
            col = (np.asarray(sres[f"sc{k}{tag}"]) - y_sref) / A_CAL
            O_symm_cols.append(col)
    O_symm = np.column_stack(O_symm_cols)   # [48 × 8]

    # Genişletilmiş O matrisi: [antisim k1..4 | simetrik k1..4]
    # O_antisim gelirken 48×12 (k=1..6); biz k=1..4 → ilk 8 sütun
    O_antisim_4 = O_antisim[:, :8]          # k=1..4 antisim [48 × 8]
    O_ext = np.hstack([O_antisim_4, O_symm])  # [48 × 16]

    # Simetrik kazançlar
    gains_symm = np.sqrt(np.mean(O_symm**2, axis=0))

    # Desen yörüngesi
    y_true = np.asarray(sres["oP"])
    y_meas = y_true + b

    f0 = f0_override if f0_override is not None else sres["f0"]

    if verbose:
        print(f"f0 (trim öncesi): {f0:+.4e} rad/s")
        print("\nSimetrik mod kazançları (RMS, boyutsuz):")
        symm_names = [f"sk{k} {'cos' if i%2==0 else 'sin'}"
                      for k in K_FIT for i in range(2)]
        for nm, g in zip(symm_names, gains_symm):
            print(f"  {nm:>9}: {g:>8.4f}")

    # ── Antisim-C trim (sadece antisim k=1..4) ────────────────────────────────
    if f_antisim_C_override is not None:
        f_C = f_antisim_C_override
    else:
        # O_antisim ilk 8 sütun (k=1..4)
        a_hat_C, *_ = np.linalg.lstsq(O_antisim_4, y_meas, rcond=None)
        KM_antisim = knob_matrix(n_q, K_FIT, antisym=True)
        P_trim_C = P - KM_antisim @ a_hat_C
        if verbose:
            print("\nAntisim-C trim spin simülasyonu...")
        r_C = run([("fC", P_trim_C.tolist(), "spin", T2, RETURN_STEPS)])
        f_C = r_C["fC"]

    if verbose:
        print(f"f_antisim_C: {f_C:+.4e} rad/s")

    # ── Genişletilmiş baz trim ────────────────────────────────────────────────
    a_hat_ext, *_ = np.linalg.lstsq(O_ext, y_meas, rcond=None)

    # Trim deseni: antisim kısmı + simetrik kısmı
    KM_antisim = knob_matrix(n_q, K_FIT, antisym=True)   # [48 × 8]
    KM_symm    = knob_matrix(n_q, K_FIT, antisym=False)  # [48 × 8]
    KM_ext = np.hstack([KM_antisim, KM_symm])            # [48 × 16]
    P_trim_ext = P - KM_ext @ a_hat_ext

    if verbose:
        a_antisim_part = a_hat_ext[:8]
        a_symm_part    = a_hat_ext[8:]
        print(f"\nGenişletilmiş fit: antisim RMS = {np.std(a_antisim_part)*1e6:.2f}μm, "
              f"simetrik RMS = {np.std(a_symm_part)*1e6:.2f}μm")
        print("Genişletilmiş baz trim spin simülasyonu...")

    r_ext = run([("fext", P_trim_ext.tolist(), "spin", T2, RETURN_STEPS)])
    f_ext = r_ext["fext"]

    if verbose:
        print(f"f_ext: {f_ext:+.4e} rad/s")

    return {
        "f0": f0,
        "f_antisim_C": f_C,
        "f_ext": f_ext,
        "gains_symm": gains_symm.tolist(),
        "a_hat_ext": a_hat_ext.tolist(),
    }


# ── Ana program ──────────────────────────────────────────────────────────────
def main():
    t0 = time.time()

    with open("params.json") as fh:
        config = json.load(fh)

    from false_edm_mode_scan import setup_fields
    fields, *_ = setup_fields(config)
    n_q = 2 * int(fields.nFODO)   # = 48

    ctx = mp.get_context("spawn")
    nw  = mp.cpu_count()

    # ── test_orbit_trim.json'dan mevcut verileri yükle ─────────────────────────
    trim_json_path = os.path.join(BASE, "test_orbit_trim.json")
    if not os.path.exists(trim_json_path):
        print("HATA: test_orbit_trim.json bulunamadı. Önce test_orbit_trim.py çalıştırın.")
        sys.exit(1)

    with open(trim_json_path) as fh:
        trim_data = json.load(fh)

    O_antisim = np.asarray(trim_data["O_matrisi"])    # [48 × 12]
    y_ref     = np.asarray(trim_data["y_true_bpm"]) - np.asarray(trim_data["bpm_ofsetleri"])
    # y_true_bpm = gerçek yörünge (ofsetsiz) → ref olarak yeniden kullanmaya gerek yok
    # Kalibrasyon ref'i test_orbit_trim'deki sref (sıfır desen → y=0'a yakın)
    # Ama burada y_ref sadece bilgi amaçlı; O_antisim zaten diferansiyel hesaplanmış.

    f0_main     = trim_data["f0"]
    f_antisim_C = trim_data["varyantlar"]["C"]["f"]

    print("=" * 70)
    print("test_symm_basis_fit.py — Genişletilmiş baz (antisim + simetrik) trim")
    print("=" * 70)
    print(f"n_q={n_q}, K_FIT={K_FIT}, A_CAL={A_CAL*1e6:.0f}μm")
    print(f"test_orbit_trim.json yüklendi:")
    print(f"  f0 (trim öncesi)   = {f0_main:+.4e} rad/s")
    print(f"  f_antisim_C (k1..4) = {f_antisim_C:+.4e} rad/s  "
          f"(bastırma {abs(f0_main/f_antisim_C):.1f}×)")

    # Antisimetrik kazançları göster (karşılaştırma için)
    print("\nAntisimetrik mod kazançları (test_orbit_trim.json'dan, k=1..4):")
    kazanclar = trim_data["kazanclar"]
    antisim_names = [f"k{k} {'cos' if i%2==0 else 'sin'}"
                     for k in [1,2,3,4] for i in range(2)]
    for nm in antisim_names:
        g = kazanclar.get(nm, float('nan'))
        print(f"  {nm:>8}: {g:>8.4f}")

    # ── Ana seed (321) analizi ─────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print(f"Seed = {PATTERN_SEED} analizi başlatılıyor...")

    seed_result = run_one_seed(
        PATTERN_SEED, OFFSET_SEED,
        n_q, ctx, nw,
        y_ref=None,          # kullanılmıyor, O_antisim zaten kalibrasyon içeriyor
        O_antisim=O_antisim,
        f0_override=f0_main,
        f_antisim_C_override=f_antisim_C,
        verbose=True
    )

    f0     = seed_result["f0"]
    f_C    = seed_result["f_antisim_C"]
    f_ext  = seed_result["f_ext"]

    gains_symm = np.asarray(seed_result["gains_symm"])

    # ── Simetrik kazançları raporla ────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("Simetrik vs antisimetrik mod kazançları karşılaştırması:")
    print(f"{'mod':>12}  {'antisim kazanç':>16}  {'simetrik kazanç':>16}")
    print("─" * 50)
    antisim_gains_4 = [kazanclar.get(nm, float('nan')) for nm in antisim_names]
    symm_names = [f"k{k} {'cos' if i%2==0 else 'sin'}"
                  for k in K_FIT for i in range(2)]
    for i, (anm, snm) in enumerate(zip(antisim_names, symm_names)):
        ga = antisim_gains_4[i]
        gs = gains_symm[i]
        print(f"  {anm:>8}  {ga:>16.4f}  {gs:>16.4f}")

    # ── Özet tablo ────────────────────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("ÖZET TABLOSU — seed=321")
    print("=" * 74)
    print(f"{'seed':>6} | {'trim öncesi dSy/dt':>22} | {'antisim-C dSy/dt':>20} | "
          f"{'genisletilmis dSy/dt':>22}")
    print("─" * 74)

    def fmt(v):
        return f"{v:+.4e}"

    suppress_C   = abs(f0 / f_C)  if f_C   != 0 else float('inf')
    suppress_ext = abs(f0 / f_ext) if f_ext != 0 else float('inf')

    print(f"{PATTERN_SEED:>6} | {fmt(f0):>22} | "
          f"{fmt(f_C):>20} ({suppress_C:.0f}x) | "
          f"{fmt(f_ext):>22} ({suppress_ext:.0f}x)")
    print("─" * 74)

    # ── Ek seed'ler (koşullu) ─────────────────────────────────────────────────
    improvement = suppress_ext / suppress_C
    extra_results = []

    if improvement > 1.5:
        print(f"\nGenişletilmiş baz {improvement:.1f}× ek bastırma sağladı "
              f"→ ek seed'ler çalıştırılıyor...")
        for pseed, oseed in EXTRA_SEEDS:
            print(f"\n{'─'*70}")
            print(f"Ek seed: pattern_seed={pseed}, offset_seed={oseed}")
            res = run_one_seed(
                pseed, oseed,
                n_q, ctx, nw,
                y_ref=None,
                O_antisim=O_antisim,
                f0_override=None,
                f_antisim_C_override=None,
                verbose=False
            )
            extra_results.append({
                "pattern_seed": pseed,
                "offset_seed": oseed,
                **res
            })
            f0_s   = res["f0"]
            f_C_s  = res["f_antisim_C"]
            f_ext_s = res["f_ext"]
            sc = abs(f0_s/f_C_s)  if f_C_s  != 0 else float('inf')
            se = abs(f0_s/f_ext_s) if f_ext_s != 0 else float('inf')
            print(f"  f0={f0_s:+.4e}  f_C={f_C_s:+.4e} ({sc:.0f}x)  "
                  f"f_ext={f_ext_s:+.4e} ({se:.0f}x)")

        # Genişletilmiş tablo
        print("\n" + "=" * 74)
        print("TAM KARŞILAŞTIRMA TABLOSU — tüm seed'ler")
        print("=" * 74)
        print(f"{'pseed':>6} | {'f0':>14} | {'antisim-C':>14} | "
              f"{'genisletilmis':>14} | {'ek bastirma':>12}")
        print("─" * 74)
        # Başlangıç seed'i
        print(f"{PATTERN_SEED:>6} | {f0:>14.4e} | {f_C:>14.4e} | "
              f"{f_ext:>14.4e} | {improve_str(f_C, f_ext):>12}")
        for r in extra_results:
            f0_s = r["f0"]; f_C_s = r["f_antisim_C"]; f_ext_s = r["f_ext"]
            print(f"{r['pattern_seed']:>6} | {f0_s:>14.4e} | {f_C_s:>14.4e} | "
                  f"{f_ext_s:>14.4e} | {improve_str(f_C_s, f_ext_s):>12}")
        print("─" * 74)
    else:
        print(f"\nGenişletilmiş baz ek bastırmada anlamlı iyileşme yok "
              f"(improvement={improvement:.2f}×) → ek seed'ler atlandı.")

    # ── Taban değerlendirmesi ─────────────────────────────────────────────────
    target = 1e-5
    print(f"\nHedef taban: {target:.0e} rad/s")
    print(f"  trim öncesi  : {abs(f0):.2e} rad/s  "
          f"({'✓' if abs(f0) < target else '✗'} hedef altında)")
    print(f"  antisim-C    : {abs(f_C):.2e} rad/s  "
          f"({'✓' if abs(f_C) < target else '✗'} hedef altında)")
    print(f"  genisletilmis: {abs(f_ext):.2e} rad/s  "
          f"({'✓' if abs(f_ext) < target else '✗'} hedef altında)")

    # ── JSON kaydet ────────────────────────────────────────────────────────────
    out = {
        "_aciklama": ("Genişletilmiş baz (antisim + simetrik k=1..4) trim testi. "
                      "test_orbit_trim.json'dan antisim kalibrasyon reuse edildi."),
        "parametreler": {
            "PATTERN_RMS_um": PATTERN_RMS * 1e6,
            "OFFSET_RMS_um": OFFSET_RMS * 1e6,
            "PATTERN_SEED": PATTERN_SEED,
            "OFFSET_SEED": OFFSET_SEED,
            "A_CAL_um": A_CAL * 1e6,
            "K_FIT": K_FIT,
            "n_q": n_q,
        },
        "antisim_kazanclar_k1_4": {nm: kazanclar[nm] for nm in antisim_names},
        "simetrik_kazanclar_k1_4": {
            nm: float(g) for nm, g in zip(symm_names, gains_symm)
        },
        "seed_321": {
            "f0": f0,
            "f_antisim_C": f_C,
            "f_ext": f_ext,
            "bastirma_antisim_C": suppress_C,
            "bastirma_ext": suppress_ext,
            "a_hat_ext_um": [float(a * 1e6) for a in seed_result["a_hat_ext"]],
        },
        "hedef_1e5_altinda": {
            "antisim_C": bool(abs(f_C) < 1e-5),
            "genisletilmis": bool(abs(f_ext) < 1e-5),
        },
        "extra_seeds": extra_results,
    }

    with open("test_symm_basis_fit.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("\nKaydedildi: test_symm_basis_fit.json")
    print(f"Toplam süre: {time.time()-t0:.1f} s")

    # ── Figür ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Genişletilmiş Baz Trim (antisim + simetrik k=1..4)\n"
        f"seed={PATTERN_SEED}, RMS={PATTERN_RMS*1e6:.0f}μm, BPM ofseti={OFFSET_RMS*1e6:.0f}μm",
        fontsize=12
    )

    # Panel 1: Mod kazançları karşılaştırması
    ax = axes[0]
    k_labels = [f"k{k}" for k in K_FIT]
    x = np.arange(len(K_FIT))
    ag_cos = [kazanclar.get(f"k{k} cos", 0.0) for k in K_FIT]
    ag_sin = [kazanclar.get(f"k{k} sin", 0.0) for k in K_FIT]
    sg_cos = gains_symm[0::2]
    sg_sin = gains_symm[1::2]
    w = 0.2
    ax.bar(x - 1.5*w, ag_cos, w, label='antisim cos', color='tab:blue', alpha=0.85)
    ax.bar(x - 0.5*w, ag_sin, w, label='antisim sin', color='tab:cyan', alpha=0.85)
    ax.bar(x + 0.5*w, sg_cos, w, label='simetrik cos', color='tab:red', alpha=0.85)
    ax.bar(x + 1.5*w, sg_sin, w, label='simetrik sin', color='tab:orange', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(k_labels)
    ax.set_ylabel("Yörünge kazancı (RMS) [m/m]")
    ax.set_title("Mod kazançları: antisim vs simetrik")
    ax.set_yscale('log')
    ax.legend(fontsize=8); ax.grid(True, which='both', axis='y', alpha=0.3)

    # Panel 2: Sahte EDM üç koşul (ana seed)
    ax = axes[1]
    labels_plt = ["trim öncesi", "antisim-C\n(k=1..4)", "genişletilmiş\n(k=1..4 her iki)"]
    vals = [abs(f0), abs(f_C), abs(f_ext)]
    colors_plt = ['tab:gray', 'tab:blue', 'tab:green']
    bars = ax.bar(labels_plt, vals, color=colors_plt, alpha=0.85, width=0.5)
    ax.set_yscale('log')
    ax.axhline(1e-5, color='red', ls='--', lw=1.5, label='hedef 10⁻⁵')
    ax.axhline(1e-4, color='orange', ls=':', lw=1, label='mevcut taban')
    for bar, val in zip(bars, vals):
        ax.annotate(f"{val:.2e}",
                    (bar.get_x() + bar.get_width()/2, val * 1.5),
                    ha='center', va='bottom', fontsize=9)
    ax.set_ylabel("|dSy/dt| [rad/s]")
    ax.set_title(f"Sahte EDM (seed={PATTERN_SEED})")
    ax.legend(fontsize=9); ax.grid(True, which='both', axis='y', alpha=0.3)

    # Panel 3: Kestirilen mod genliği dağılımı (genişletilmiş baz)
    ax = axes[2]
    a_hat = np.asarray(seed_result["a_hat_ext"])
    n_modes = len(a_hat)
    x_a = np.arange(n_modes)
    mode_labels = (
        [f"A k{k}{'c' if i%2==0 else 's'}" for k in K_FIT for i in range(2)] +
        [f"S k{k}{'c' if i%2==0 else 's'}" for k in K_FIT for i in range(2)]
    )
    colors_ext = ['tab:blue'] * 8 + ['tab:red'] * 8
    ax.bar(x_a, np.abs(a_hat) * 1e6, color=colors_ext, alpha=0.85)
    ax.set_xticks(x_a); ax.set_xticklabels(mode_labels, rotation=45, fontsize=7)
    ax.set_ylabel("|â| [μm]")
    ax.set_title("Kestirilen mod genliği (A=antisim, S=simetrik)")
    ax.grid(True, axis='y', alpha=0.3)
    # Dikey çizgi antisim/simetrik sınırı
    ax.axvline(7.5, color='black', ls='--', lw=1, alpha=0.5)
    ax.annotate("antisim", (3.5, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1),
                ha='center', fontsize=8, color='tab:blue')
    ax.annotate("simetrik", (11.5, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1),
                ha='center', fontsize=8, color='tab:red')

    plt.tight_layout()
    plt.savefig("test_symm_basis_fit.png", dpi=150)
    print("Figür kaydedildi: test_symm_basis_fit.png")


def improve_str(f_C, f_ext):
    """Antisim-C → genişletilmiş ek bastırma."""
    if f_C == 0 or f_ext == 0:
        return "—"
    ratio = abs(f_C / f_ext)
    return f"{ratio:.1f}x"


if __name__ == "__main__":
    main()
