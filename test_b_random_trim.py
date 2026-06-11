#!/usr/bin/env python3
"""test_b_random_trim.py — Rastgele hizalama hatası + çok-modlu fazlı trim.

Senaryo:
  Gerçek deneye en yakın durum: 48 kuadrupolde RASTGELE hizalama hatası
  (RMS=10μm). Desende her k modunun İKİ kuadratürü var:
      Δy_j = Σ_k A_k · s_j · cos(2πk·n_j/N − φ_k)
  φ_k fazları rastgele. Şimdiye dek yalnız cos fazı (φ=0) kalibre edildi.

Faz problemi:
  Trim modu k'yi φ fazında uygularsak etkinliği
      c_k(φ) = c_k^cos·cosφ + c_k^sin·sinφ = |c_k|·cos(φ − ψ_k)
  olmalı (doğrusallık). ψ_k = "etkin faz": birim genlik başına en güçlü
  trim bu fazda. Yanlış fazda (φ = ψ_k ± 90°) trim hiç etki etmez!

Plan:
  Bölüm 1: Çift kuadratür kalibrasyon — c_k^cos, c_k^sin (k=1..6, A=10μm)
           → |c_k| ve ψ_k tablosu.
  Bölüm 2: Faz modeli doğrulaması — k=2 için φ=45°,135° ara ölçümler,
           sinüzoid kestirimiyle karşılaştırma.
  Bölüm 3: Rastgele desen (seed=123, RMS=10μm) → f₀ ölçümü;
           desenin Fourier spektrumu + kalibre c'lerle f₀ TAHMİNİ
           (doğrusallık/tamlık testi).
  Bölüm 4: Trim stratejileri (3'er iterasyon, hepsi ölç-trimle):
           A) k=2 @ ψ₂ (tek mod, optimal faz)
           B) k=2 @ ψ₂ + k=3 @ ψ₃ eşit bölüşmüş (bütçe küçülür)
           C) k=2 @ φ=0 (yalnız cos — faz-cahil kontrol)

Tüm koşullar CO=False (eksenden fırlatma), t2=1ms.
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

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fourier_reconstruct import fodo_basis

# ── Parametreler ─────────────────────────────────────────────────────────────
A_CAL        = 1e-5          # kalibrasyon genliği [m] = 10 μm
T2           = 1e-3
RETURN_STEPS = 6000
K_CAL        = [1, 2, 3, 4, 5, 6]   # çift kuadratür kalibre edilecek modlar
PHASE_CHECK  = [45.0, 135.0]        # k=2 faz modeli ara noktaları [derece]
PATTERN_SEED = 123
PATTERN_RMS  = 1e-5          # rastgele desen RMS [m] = 10 μm
N_ITER       = 3


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
    """CO=False (eksenden fırlatma) → dSy/dt eğimi."""
    label, dy_list, t2, return_steps = task

    import os, sys, json, numpy as np
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from false_edm_mode_scan import setup_fields
    from integrator import integrate_particle

    with open("params.json") as fh:
        config = json.load(fh)
    fields, y0, beta0, R0, p_mag, direction = setup_fields(config)
    dt = float(config.get("dt", 1e-11))
    dy = np.asarray(dy_list, dtype=float)

    saved = _suppress_stdout()
    try:
        fields.poincare_quad_index = 0.0
        _, poin, poin_t = integrate_particle(
            y0, 0.0, t2, dt,
            fields=fields, return_steps=return_steps, quad_dy=dy)
    finally:
        _restore_stdout(saved)

    slope = float(np.polyfit(np.asarray(poin_t, float),
                             np.asarray(poin[:, 7], float), 1)[0])
    return label, slope


def mode_vec_phase(n_q, k, amp, phase_rad, antisym):
    """Faz φ'li tek FODO modu: Δy = amp·s·cos(2πkn/N − φ)."""
    Fk, _ = fodo_basis(n_q, [k], antisym)
    return amp * (Fk[:, 0] * np.cos(phase_rad) + Fk[:, 1] * np.sin(phase_rad))


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

    # Rastgele desen
    rng = np.random.default_rng(PATTERN_SEED)
    P = rng.standard_normal(n_q) * PATTERN_RMS

    # ══ BÖLÜM 1+2+3a: kalibrasyon + faz ara noktaları + f0 (tek havuz) ══════
    tasks = []
    for k in K_CAL:
        tasks.append((f"c{k}_cos",
                      mode_vec_phase(n_q, k, A_CAL, 0.0, antisym).tolist(),
                      T2, RETURN_STEPS))
        tasks.append((f"c{k}_sin",
                      mode_vec_phase(n_q, k, A_CAL, np.pi/2, antisym).tolist(),
                      T2, RETURN_STEPS))
    for deg in PHASE_CHECK:
        tasks.append((f"ph{deg:.0f}",
                      mode_vec_phase(n_q, 2, A_CAL, np.radians(deg), antisym).tolist(),
                      T2, RETURN_STEPS))
    tasks.append(("f0", P.tolist(), T2, RETURN_STEPS))

    print(f"Bölüm 1+2+3a: {len(tasks)} simülasyon ({nw} işçi)...")
    res = run(tasks)

    # ── Bölüm 1: çift kuadratür tablo ──────────────────────────────────────
    c_cos = {k: res[f"c{k}_cos"] / A_CAL for k in K_CAL}
    c_sin = {k: res[f"c{k}_sin"] / A_CAL for k in K_CAL}
    c_abs = {k: float(np.hypot(c_cos[k], c_sin[k])) for k in K_CAL}
    psi   = {k: float(np.arctan2(c_sin[k], c_cos[k])) for k in K_CAL}

    print(f"\n{'─'*72}")
    print("Bölüm 1: Çift kuadratür kalibrasyon (CO=False, A=10μm)")
    print(f"{'─'*72}")
    print(f"{'k':>3}  {'c_k^cos':>12}  {'c_k^sin':>12}  {'|c_k|':>12}  "
          f"{'ψ_k [°]':>9}")
    print('─'*72)
    for k in K_CAL:
        print(f"{k:>3}  {c_cos[k]:>12.4e}  {c_sin[k]:>12.4e}  "
              f"{c_abs[k]:>12.4e}  {np.degrees(psi[k]):>9.2f}")

    # ── Bölüm 2: faz modeli doğrulaması (k=2) ──────────────────────────────
    print(f"\n{'─'*72}")
    print("Bölüm 2: Faz modeli doğrulaması — c_2(φ) = |c_2|·cos(φ−ψ_2) ?")
    print(f"{'─'*72}")
    print(f"{'φ [°]':>7}  {'ölçülen c_2(φ)':>16}  {'model':>14}  {'fark %':>8}")
    print('─'*72)
    check_degs  = [0.0, 45.0, 90.0, 135.0]
    check_meas  = [c_cos[2],
                   res["ph45"]/A_CAL,
                   c_sin[2],
                   res["ph135"]/A_CAL]
    model_vals, devs = [], []
    for deg, meas in zip(check_degs, check_meas):
        model = c_abs[2] * np.cos(np.radians(deg) - psi[2])
        model_vals.append(model)
        dev = (meas - model) / c_abs[2] * 100
        devs.append(dev)
        print(f"{deg:>7.0f}  {meas:>16.4e}  {model:>14.4e}  {dev:>8.3f}")
    max_dev = max(abs(d) for d in devs)
    print(f"\n  → Maks sapma: %{max_dev:.3f} / |c_2|  "
          f"(0 ve 90° tanım gereği tam)")

    # ── Bölüm 3: rastgele desen spektrumu + f0 tahmini ─────────────────────
    f0 = res["f0"]

    # Desenin Fourier ayrıştırması (k=0..12 tam taban, LSQ)
    F_full, meta = fodo_basis(n_q, list(range(0, 13)), antisym)
    coef, *_ = np.linalg.lstsq(F_full, P, rcond=None)
    spec = {}   # k → (A_k, φ_k)
    for c_val, (k, kind) in zip(coef, meta):
        spec.setdefault(k, {})[kind] = c_val
    spec_amp, spec_ph = {}, {}
    for k, d in spec.items():
        if 'dc' in d:
            spec_amp[k], spec_ph[k] = abs(d['dc']), 0.0
        else:
            ac, as_ = d.get('cos', 0.0), d.get('sin', 0.0)
            spec_amp[k] = float(np.hypot(ac, as_))
            spec_ph[k]  = float(np.arctan2(as_, ac))

    # f0 tahmini: kalibre modlardan (k=1..6) katkılar
    f_pred_parts = {}
    for k in K_CAL:
        ak = spec[k].get('cos', 0.0)
        bk = spec[k].get('sin', 0.0)
        f_pred_parts[k] = c_cos[k]*ak + c_sin[k]*bk
    f_pred = sum(f_pred_parts.values())

    print(f"\n{'─'*72}")
    print(f"Bölüm 3: Rastgele desen (seed={PATTERN_SEED}, RMS=10μm)")
    print(f"{'─'*72}")
    print(f"{'k':>3}  {'A_k [μm]':>10}  {'φ_k [°]':>9}  {'katkı tahmini [rad/s]':>22}")
    print('─'*72)
    for k in sorted(spec_amp):
        part = f_pred_parts.get(k, float('nan'))
        part_s = f"{part:+.3e}" if k in f_pred_parts else "(kalibre değil)"
        print(f"{k:>3}  {spec_amp[k]*1e6:>10.2f}  "
              f"{np.degrees(spec_ph[k]):>9.1f}  {part_s:>22}")
    print(f"\n  Ölçülen  f0 = {f0:+.4e} rad/s")
    print(f"  Tahmin (k=1..6) = {f_pred:+.4e} rad/s "
          f"(fark %{abs(f_pred-f0)/abs(f0)*100:.2f} — k≥7 katkısı + fit artığı)")

    # ── Bölüm 4: trim stratejileri ─────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("Bölüm 4: Trim stratejileri (3 iterasyon, ölç-trimle)")
    print(f"{'─'*72}")
    print(f"  A) k=2 @ ψ₂={np.degrees(psi[2]):.1f}° (optimal faz), "
          f"etkin |c₂|={c_abs[2]:.3e}")
    print(f"  B) k=2 @ ψ₂ + k=3 @ ψ₃={np.degrees(psi[3]):.1f}° eşit bölüşmüş, "
          f"etkin |c₂|+|c₃|={c_abs[2]+c_abs[3]:.3e}")
    print(f"  C) k=2 @ φ=0 (yalnız cos — faz-cahil), etkin c₂^cos={c_cos[2]:.3e}")

    trims = {"A": np.zeros(n_q), "B": np.zeros(n_q), "C": np.zeros(n_q)}
    hist  = {"A": [f0], "B": [f0], "C": [f0]}
    budget = {"A": 0.0, "B": 0.0, "C": 0.0}

    for it in range(1, N_ITER + 1):
        # trim genlikleri (son ölçümden)
        aA = -hist["A"][-1] / c_abs[2]
        xB = -hist["B"][-1] / (c_abs[2] + c_abs[3])
        aC = -hist["C"][-1] / c_cos[2]

        trims["A"] = trims["A"] + mode_vec_phase(n_q, 2, aA, psi[2], antisym)
        trims["B"] = (trims["B"]
                      + mode_vec_phase(n_q, 2, xB, psi[2], antisym)
                      + mode_vec_phase(n_q, 3, xB, psi[3], antisym))
        trims["C"] = trims["C"] + mode_vec_phase(n_q, 2, aC, 0.0, antisym)
        budget["A"] += abs(aA)
        budget["B"] += 2 * abs(xB)
        budget["C"] += abs(aC)

        print(f"\nAdım {it}: A({aA*1e6:+.2f}μm) "
              f"B({xB*1e6:+.2f}μm ×2) C({aC*1e6:+.2f}μm)...")
        r = run([(s, (P + trims[s]).tolist(), T2, RETURN_STEPS)
                 for s in ("A", "B", "C")])
        for s in ("A", "B", "C"):
            hist[s].append(r[s])
            sup = abs(f0 / r[s]) if r[s] != 0 else float('inf')
            print(f"  {s}: f{it} = {r[s]:+.3e}  (bastırma {sup:.1f}×)")

    print(f"\n{'─'*78}")
    print("Strateji özeti")
    print(f"{'─'*78}")
    print(f"{'adım':>5}  {'A: k=2@ψ₂':>13}  {'B: k=2+3@ψ':>13}  "
          f"{'C: k=2@cos':>13}")
    print('─'*78)
    for i in range(N_ITER + 1):
        print(f"{i:>5}  " + "  ".join(
            f"{hist[s][i]:>13.3e}" for s in ("A", "B", "C")))
    print('─'*78)
    print("bastırma: " + "   ".join(
        f"{s}: {abs(f0/hist[s][-1]):.1e}×" if hist[s][-1] != 0 else f"{s}: inf"
        for s in ("A", "B", "C")))
    print("bütçe   : " + "   ".join(
        f"{s}: {budget[s]*1e6:.2f}μm" for s in ("A", "B", "C")))

    # k=2 modunun gerçek içeriğiyle karşılaştırma
    A2_true, ph2_true = spec_amp[2], spec_ph[2]
    print(f"\n  Desendeki gerçek k=2 içeriği: A₂={A2_true*1e6:.2f}μm, "
          f"φ₂={np.degrees(ph2_true):.1f}°")
    print(f"  Strateji A toplam trimi     : {budget['A']*1e6:.2f}μm @ "
          f"ψ₂={np.degrees(psi[2]):.1f}°")

    # ── JSON çıktı ─────────────────────────────────────────────────────────
    out = {
        "_aciklama": "Rastgele desen + fazlı çok-modlu trim (CO=False)",
        "pattern_seed": PATTERN_SEED, "pattern_rms_m": PATTERN_RMS,
        "c_cos": {str(k): c_cos[k] for k in K_CAL},
        "c_sin": {str(k): c_sin[k] for k in K_CAL},
        "c_abs": {str(k): c_abs[k] for k in K_CAL},
        "psi_deg": {str(k): float(np.degrees(psi[k])) for k in K_CAL},
        "faz_modeli_max_sapma_yuzde": float(max_dev),
        "f0": f0, "f0_tahmin_k1_6": float(f_pred),
        "desen_spektrum_amp_um": {str(k): spec_amp[k]*1e6 for k in spec_amp},
        "desen_spektrum_faz_deg": {str(k): float(np.degrees(spec_ph[k]))
                                   for k in spec_ph},
        "stratejiler": {s: {"f_hist": hist[s], "butce_um": budget[s]*1e6,
                            "bastirma": (abs(f0/hist[s][-1])
                                         if hist[s][-1] != 0 else None)}
                        for s in ("A", "B", "C")},
    }
    with open("test_b_random_trim.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nKaydedildi: test_b_random_trim.json")

    elapsed = time.time() - t0
    print(f"Toplam süre: {elapsed:.1f} s")

    # ══ Figür ═══════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10))
    fig.suptitle("Rastgele desen + fazlı çok-modlu trim (CO=False, RMS=10μm)",
                 fontsize=13)

    # Panel 1: çift kuadratür c_k
    ax = axes[0, 0]
    w = 0.35
    kk = np.array(K_CAL)
    ax.bar(kk - w/2, [c_cos[k] for k in K_CAL], w, label='c_k^cos',
           color='tab:red', alpha=0.85)
    ax.bar(kk + w/2, [c_sin[k] for k in K_CAL], w, label='c_k^sin',
           color='tab:blue', alpha=0.85)
    ax.axhline(0, color='k', lw=0.8)
    for k in K_CAL:
        ax.annotate(f"ψ={np.degrees(psi[k]):.0f}°",
                    (k, max(c_cos[k], c_sin[k])),
                    ha='center', va='bottom', fontsize=8)
    ax.set_xlabel("Fourier modu k"); ax.set_ylabel("c_k [rad/s/m]")
    ax.set_title("Çift kuadratür kalibrasyon")
    ax.set_xticks(K_CAL); ax.legend(); ax.grid(True, axis='y', alpha=0.3)

    # Panel 2: faz modeli (k=2)
    ax = axes[0, 1]
    phis = np.linspace(0, 180, 200)
    model = c_abs[2] * np.cos(np.radians(phis) - psi[2])
    ax.plot(phis, model, '-', color='tab:gray', lw=2,
            label=f"|c₂|·cos(φ−ψ₂)  ψ₂={np.degrees(psi[2]):.1f}°")
    ax.plot(check_degs, check_meas, 'o', color='tab:red', ms=10,
            label='ölçümler')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel("trim fazı φ [°]"); ax.set_ylabel("c_2(φ) [rad/s/m]")
    ax.set_title(f"Faz modeli doğrulaması (maks sapma %{max_dev:.2f})")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Panel 3: desen spektrumu
    ax = axes[1, 0]
    ks = sorted(k for k in spec_amp if k > 0)
    ax.bar(ks, [spec_amp[k]*1e6 for k in ks], color='tab:purple', alpha=0.85)
    ax.set_xlabel("Fourier modu k"); ax.set_ylabel("A_k [μm]")
    ax.set_title(f"Rastgele desen spektrumu (seed={PATTERN_SEED})")
    ax.set_xticks(ks); ax.grid(True, axis='y', alpha=0.3)

    # Panel 4: strateji yakınsaması
    ax = axes[1, 1]
    steps = np.arange(N_ITER + 1)
    styles = {"A": ('o-', 'tab:green', f'A: k=2@ψ₂ ({budget["A"]*1e6:.1f}μm)'),
              "B": ('s--', 'tab:blue', f'B: k=2+3@ψ ({budget["B"]*1e6:.1f}μm)'),
              "C": ('^:', 'tab:orange', f'C: k=2@cos ({budget["C"]*1e6:.1f}μm)')}
    for s, (st, col, lab) in styles.items():
        ax.semilogy(steps, np.abs(hist[s]), st, color=col, ms=9, lw=2,
                    label=lab)
    ax.set_xticks(steps)
    ax.set_xlabel("iterasyon adımı"); ax.set_ylabel("|dSy/dt| [rad/s]")
    ax.set_title("Ölç-trimle yakınsaması (parantez: trim bütçesi)")
    ax.legend(fontsize=9); ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    fout = "test_b_random_trim.png"
    plt.savefig(fout, dpi=150)
    print(f"Figür kaydedildi: {fout}")


if __name__ == "__main__":
    main()
