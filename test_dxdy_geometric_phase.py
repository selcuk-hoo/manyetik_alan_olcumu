#!/usr/bin/env python3
"""test_dxdy_geometric_phase.py — dx·dy geometrik-faz sahte EDM kanalı (Omarov σ²).

AMAÇ (bkz. false_edm_harmonic_sinir.md §13):
  "Omarov makalesinde sahte EDM quad misalignment'ın KARESİyle değişir; bizde
  doğrusal terim baskın — kuadratik nerede?" sorusunun nihai cevabını üretir.

  Bulgu: kuadratik sinyal gizlenmemiştir; proje yalnız DİKEY (dy) misalignment'ı
  ölçüyordu — o, birinci-derece radyal-alan kanalıdır (doğrusal, ~1e-9). Omarov'un
  kuadratiği YATAY+DİKEY (dx+dy) misalignment'ın geometrik-faz çapraz kanalıdır:
    dy → B_x (spin x-ekseni etrafında döner),  dx → B_y (y-ekseni etrafında döner)
  İki komütatif-olmayan dönme → geometrik (Berry) faz → S_y ∝ dx·dy ∝ σ².

ÜÇ TEST (tek koşu):
  A) Kontrast (σ=10μm, tek seed): dy-only / dx-only / dx+dy.
     Beklenti: dx-only ≈ 0 (düzlemler ayrık), dx+dy ≫ dy-only (çapraz terim).
  B) dx+dy CO=True α taraması (tek seed): dSy/dt vs σ → log-log üs ≈ 2.
  C) Çok-seed RMS: N_SEED rastgele desen, dx+dy, CO=True 4D, RMS(σ) → σ² ve
     Omarov ~1e-5 (10μm) ile karşılaştırma.

YÖNTEM NOTLARI (neden böyle — §13.8):
  - Seküler eğim madde 2 model fitiyle (measure_dSy_dt_model) ölçülür: düz polyfit
    betatron salınımını seküler sanar.
  - CO=True (4D kapalı yörünge): betatronu kaynağında öldürür. dx+dy kanalı için
    demet=ideal olduğundan (4-katlı antitetik testle kanıtlandı, §13.6) tek ideal
    parçacık demeti temsil eder.
  - CO doğrusal ölçeklenir: sextupol yok → kapalı yörünge misalignment'ta tam
    doğrusal; CO bir kez bulunup σ ile ölçeklenir (yaklaşıklık değil).
  - Her yerde TAM dt: adım büyütmek GL4 truncation hatasını bozar.

Ortam değişkenleri: N_SEED (varsayılan 10), TAARZ (σ listesi μm, vir. ayrık).
Çıktı: konsol tabloları.
"""
import json
import os
import sys
import time
import multiprocessing as mp

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
os.chdir(BASE)

from false_edm_mode_scan import (setup_fields, _make_state,  # noqa: E402
                                 measure_dSy_dt_model, C)
from integrator import integrate_particle  # noqa: E402

T2  = 5e-4
DT  = 1e-11          # tam adım (CO arama + spin izi)
SIG0 = 10e-6         # CO referans ölçeği (doğrusal ölçekleme için)


def _T_rev(f, beta0, R0):
    circ = 2 * np.pi * R0 + 4 * f.nFODO * f.driftLen + 2 * f.nFODO * f.quadLen
    return circ / (beta0 * C)


def _find_co_plane(f, p_mag, direction, qd, qx, dt, T_rev, plane,
                   n_turns=40, n_iter=2):
    """Tek düzlemde (plane='x'|'y') kapalı yörünge fırlatma noktası (Newton).

    Sabit azimutta tur-başına konum varyansını minimize eder; lineer lattiste
    varyans, sapmanın kuadratik formudur → sonlu-fark Hessian ile tek-iki adım."""
    f.poincare_quad_index = 0.0
    col = 0 if plane == 'x' else 1

    def state(c, cp):
        v = [c, 0, cp, 0] if plane == 'x' else [0, c, 0, cp]
        return _make_state(v, p_mag, direction, [0.0, 0.0, direction])

    def var2(c, cp):
        _, poin, _ = integrate_particle(state(c, cp), 0.0, n_turns * T_rev, dt,
                                        fields=f, return_steps=10,
                                        quad_dy=qd, quad_dx=qx)
        if poin is None or len(poin) < 5:
            return 1e30
        return float(np.var(poin[:, col]))

    c, cp = 0.0, 0.0
    s1, s2 = 2e-4, 2e-5
    for _ in range(n_iter):
        f0 = var2(c, cp)
        fpy = var2(c + s1, cp); fmy = var2(c - s1, cp)
        fpp = var2(c, cp + s2); fmp = var2(c, cp - s2)
        fpp2 = var2(c + s1, cp + s2)
        gy = (fpy - fmy) / (2 * s1); gp = (fpp - fmp) / (2 * s2)
        Hyy = (fpy - 2 * f0 + fmy) / (s1 * s1)
        Hpp = (fpp - 2 * f0 + fmp) / (s2 * s2)
        Hyp = (fpp2 - fpy - fpp + f0) / (s1 * s2)
        det = Hyy * Hpp - Hyp * Hyp
        if det <= 0 or Hyy <= 0:
            break
        c += -(Hpp * gy - Hyp * gp) / det
        cp += -(-Hyp * gy + Hyy * gp) / det
        s1 *= 0.2; s2 *= 0.2
    f.poincare_quad_index = -1.0
    return c, cp


def _secular_co(qd, qx, co=None):
    """Verilen misalignment için CO=True seküler dSy/dt (işaretli, model fit).
    co verilirse (xc,xpc,yc,ypc) fırlatma noktası olarak kullanılır (ölçekleme)."""
    f, y0, beta0, R0, p_mag, direction = setup_fields(json.load(open("params.json")))
    T_rev = _T_rev(f, beta0, R0)
    fd = os.dup(1); nul = os.open(os.devnull, os.O_WRONLY)
    os.dup2(nul, 1); os.close(nul)
    try:
        if co is None:
            yc, ypc = _find_co_plane(f, p_mag, direction, qd, qx, DT, T_rev, 'y')
            xc, xpc = _find_co_plane(f, p_mag, direction, qd, qx, DT, T_rev, 'x')
        else:
            xc, xpc, yc, ypc = co
        yl = _make_state([xc, yc, xpc, ypc], p_mag, direction,
                         [0.0, 0.0, direction])
        f.poincare_quad_index = 0.0
        _, poin, pt = integrate_particle(yl, 0.0, T2, DT, fields=f,
                                         return_steps=5000, quad_dy=qd, quad_dx=qx)
    finally:
        os.dup2(fd, 1); os.close(fd)
    return float(measure_dSy_dt_model(np.asarray(poin[:, 7], float),
                                      np.asarray(pt, float)))


def _n_q():
    f, *_ = setup_fields(json.load(open("params.json")))
    return 2 * int(f.nFODO)


def _contrast(_):
    """Test A: dy / dx / dx+dy kontrastı (σ=10μm, seed 0)."""
    n_q = _n_q()
    rng = np.random.default_rng(1000)
    dy = rng.standard_normal(n_q) * 10e-6
    dx = rng.standard_normal(n_q) * 10e-6
    z = np.zeros(n_q)
    return {"dy": _secular_co(dy, z), "dx": _secular_co(z, dx),
            "both": _secular_co(dy, dx)}


def _alpha_seed(seed):
    """Test B+C: bir seed için dx+dy seküler eğimi tüm σ'larda (CO ölçekli)."""
    n_q = _n_q()
    rng = np.random.default_rng(1000 + seed)
    qd_u = rng.standard_normal(n_q); qx_u = rng.standard_normal(n_q)
    f, y0, beta0, R0, p_mag, direction = setup_fields(json.load(open("params.json")))
    T_rev = _T_rev(f, beta0, R0)
    fd = os.dup(1); nul = os.open(os.devnull, os.O_WRONLY)
    os.dup2(nul, 1); os.close(nul)
    try:
        yc0, ypc0 = _find_co_plane(f, p_mag, direction, qd_u * SIG0, qx_u * SIG0,
                                   DT, T_rev, 'y')
        xc0, xpc0 = _find_co_plane(f, p_mag, direction, qd_u * SIG0, qx_u * SIG0,
                                   DT, T_rev, 'x')
    finally:
        os.dup2(fd, 1); os.close(fd)
    out = {}
    for sig in SIGMAS:
        r = sig / SIG0
        co = (xc0 * r, xpc0 * r, yc0 * r, ypc0 * r)
        out[sig] = _secular_co(qd_u * sig, qx_u * sig, co=co)
    return seed, out


# σ listesi (μm) ortam değişkeninden
SIGMAS = [float(s) * 1e-6 for s in os.environ.get("TAARZ", "5,10,20").split(",")]
N_SEED = int(os.environ.get("N_SEED", "10"))


def main():
    t0 = time.time()
    ctx = mp.get_context("spawn")

    # ── Test A: kontrast ──
    print("=" * 64)
    print("  TEST A — dy / dx / dx+dy kontrastı (σ=10μm, seed 0)")
    print("=" * 64)
    with ctx.Pool(1) as p:
        c = p.map(_contrast, [0])[0]
    print(f"  dy-only : {c['dy']:+.3e} rad/s")
    print(f"  dx-only : {c['dx']:+.3e} rad/s   (~0 beklenir: düzlemler ayrık)")
    print(f"  dx+dy   : {c['both']:+.3e} rad/s   ({abs(c['both']/c['dy']):.0f}× dy-only)")
    print(f"  çapraz  : both-dy-dx = {c['both']-c['dy']-c['dx']:+.3e}\n")

    # ── Test B+C: çok-seed dx+dy α + RMS ──
    print("=" * 64)
    print(f"  TEST B+C — dx+dy CO=True, {N_SEED} seed, RMS(σ)")
    print("=" * 64)
    print("seed " + " ".join(f"{s*1e6:>9.0f}um" for s in SIGMAS))
    by = {s: [] for s in SIGMAS}
    with ctx.Pool(min(mp.cpu_count(), N_SEED)) as p:
        for seed, out in p.imap_unordered(_alpha_seed, range(N_SEED)):
            for s in SIGMAS:
                by[s].append(out[s])
            print(f"{seed:>4} " + " ".join(f"{out[s]:>+11.2e}" for s in SIGMAS),
                  flush=True)

    print(f"\n{'σ[μm]':>6} {'RMS':>11} {'medyan':>11} {'min':>10} {'maks':>10}")
    rms = []
    for s in SIGMAS:
        v = np.array(by[s]); av = np.abs(v)
        r = np.sqrt(np.mean(v ** 2)); rms.append(r)
        print(f"{s*1e6:>6.0f} {r:>11.3e} {np.median(av):>11.3e} "
              f"{av.min():>10.2e} {av.max():>10.2e}")
    A = np.array(SIGMAS)
    a = np.polyfit(np.log(A), np.log(rms), 1)[0]
    print(f"\n  RMS ~ σ^{a:.2f}   (Omarov: kuadratik = 2)")
    if 10e-6 in SIGMAS:
        r10 = rms[SIGMAS.index(10e-6)]
        print(f"  10μm RMS = {r10:.3e} rad/s  vs Omarov ~1e-5  → {1e-5/r10:.1f}×")
    print(f"\n  toplam süre: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
