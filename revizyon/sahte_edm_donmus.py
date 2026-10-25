#!/usr/bin/env python3
"""sahte_edm_donmus.py — donmuş spin noktasında sahte EDM ölçüm aracı.

NEDEN (revizyon/tests/teshis_enerji_ABC.py): eski prosedür (kapalı yörünge üzerinde, momentum
büyüklüğü sihirli) parçacığın enerjisini fırlatma noktasındaki kapalı yörünge kaymasına bağlı
olarak kaydırır. Sahte EDM enerji sapmasına doğrusal ve çok duyarlıdır (δ = 1e-7 → ~290 × hedef),
bu yüzden sonuç fırlatma azimutuna göre +2729 ile −456 × hedef arasında değişir. Yatay spin
presesyonunun sıfır olduğu (donmuş spin) enerjide ise dört azimut %0.2 içinde aynı sonucu verir.
Deney de enerjiyi donmuş spin geri beslemesiyle tuttuğu için doğru çalışma noktası budur.

PROSEDÜR:
  1. Kapalı yörünge üzerinde, toplam enerjisi tasarım değerinde (enerji korunumlu) fırlatma; δ = 0:
     f0 = dS_y/dt (model-fit), ω0 = yatay presesyon hızı d/dt atan2(S_x, S_z).
  2. δ1 = −ω0 / S (S = dω_h/dδ ≈ −5.24e6 rad/s, teşhisten); aynı ölçüm → f1, ω1.
  3. f* = f0 + (f1 − f0)(0 − ω0)/(ω1 − ω0)  (f ve ω_h δ ile doğrusal; teşhiste doğrulandı).
  Kapalı yörünge araması her δ için aynı fırlatma kuralıyla yapılır.

Kullanım (Python):  from sahte_edm_donmus import measure_frozen
  r = measure_frozen(dx, dy, direction=-1)   →  dict(f=..., f0, f1, w0, w1, delta_star, ...)
Saptırıcı hataları/düzelticiler: integrator_rev.DEFL_* global dizileri (çağıran ayarlar).
"""
import os, sys, json, tempfile
import numpy as np

REV = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(REV)
for p in (REV, os.path.join(REV, "tests"), BASE, os.path.join(BASE, "berry_data"),
          os.path.join(BASE, "kmod_drivers")):
    if p not in sys.path:
        sys.path.insert(0, p)
import integrator_rev
sys.modules["integrator"] = integrator_rev
from ortak import Q_E, M_P, C_LIGHT, E0_MAGIC, R0

S_W_DELTA = -5.238e6        # dω_h/dδ [rad/s] (teshis_enerji_ABC, seed 0; donmuş spin araması için başlangıç)
NQ = 48


def launcher(delta):
    """Enerji korunumlu fırlatma + ek bağıl momentum sapması δ."""
    def make_state(v, p_mag, direction, spin):
        x = v[0]
        W_ref = np.sqrt((p_mag * C_LIGHT) ** 2 + (M_P * C_LIGHT ** 2) ** 2)
        Wk = W_ref - Q_E * (-E0_MAGIC * R0 * np.log((R0 + x) / R0))
        p = np.sqrt(Wk ** 2 - (M_P * C_LIGHT ** 2) ** 2) / C_LIGHT * (1.0 + delta)
        return [v[0], v[1], 0.0, p * direction * v[2], p * direction * v[3], p * direction,
                spin[0], spin[1], spin[2]]
    return make_state


def _one(dx, dy, direction, delta, tilt, dG, n_turns, n_iter, t2):
    import false_edm_4d, false_edm_mode_scan
    ms = launcher(delta)
    false_edm_4d._make_state = ms; false_edm_mode_scan._make_state = ms
    from false_edm_mode_scan import setup_fields, measure_dSy_dt_model
    from false_edm_4d import find_co_4d, _T_rev
    cfg = json.load(open(os.path.join(BASE, "params.json"))); cfg["direction"] = float(direction)
    fields, y0, beta0, R0_, p_mag, d = setup_fields(cfg)
    z = np.zeros(NQ)
    tilt = z if tilt is None else np.asarray(tilt)
    v_co, resid = find_co_4d(fields, p_mag, d, dx, dy, tilt, _T_rev(fields, beta0, R0_),
                             n_turns=n_turns, n_iter=n_iter, dG=dG)
    launch = ms(v_co, p_mag, d, [0.0, 0.0, d])
    fields.poincare_quad_index = 0.0
    _, poin, pt = integrator_rev.integrate_particle(launch, 0.0, t2, float(cfg["dt"]), fields=fields,
                                                    return_steps=4000, quad_dx=dx, quad_dy=dy,
                                                    quad_tilt=tilt, quad_dG=dG)
    t = np.asarray(pt, float)
    f = float(measure_dSy_dt_model(np.asarray(poin[:, 7], float), t))
    ang = np.unwrap(np.arctan2(np.asarray(poin[:, 6], float), np.asarray(poin[:, 8], float) * d))
    return f, float(np.polyfit(t, ang, 1)[0]), float(resid)


def measure_frozen(dx, dy, direction=-1, tilt=None, dG=None, n_turns=28, n_iter=4, t2=3e-4):
    """Donmuş spin noktasında sahte EDM [rad/s] ve ayrıntılar. Çağıran, kendi geçici dizininde olmalı
    (C++ cod_data.txt/rf.txt yazar)."""
    cwd = os.getcwd(); os.chdir(tempfile.mkdtemp())
    try:
        dx = np.asarray(dx, float); dy = np.asarray(dy, float)
        f0, w0, r0 = _one(dx, dy, direction, 0.0, tilt, dG, n_turns, n_iter, t2)
        d1 = -w0 / S_W_DELTA
        f1, w1, r1 = _one(dx, dy, direction, d1, tilt, dG, n_turns, n_iter, t2)
        fs = f0 + (f1 - f0) * (0.0 - w0) / (w1 - w0) if w1 != w0 else f1
        ds = d1 * (0.0 - w0) / (w1 - w0) if w1 != w0 else d1
        return dict(f=float(fs), f0=f0, f1=f1, w0=w0, w1=w1, delta1=d1, delta_star=float(ds),
                    co_resid=max(r0, r1))
    finally:
        os.chdir(cwd)


# ───────────────────────── σ² kampanyası ─────────────────────────
OUT_SIGMA = os.path.join(REV, "sonuclar", "donmus_sigma.json")


def _sig_task(task):
    sg, seed, direction = task
    rng = np.random.default_rng(3000 + seed)          # paper_runs 'sigma' ile aynı desen
    dx = rng.normal(0, sg, NQ); dy = rng.normal(0, sg, NQ)
    r = measure_frozen(dx, dy, direction=direction)
    r.update(sigma_um=sg * 1e6, seed=seed, direction=direction, f_target=r["f"] / 1e-9)
    return r


def run_sigma(workers=4, seeds=(0, 1, 2), sigmas=(2.5e-6, 5e-6, 10e-6), directions=(-1,)):
    from concurrent.futures import ProcessPoolExecutor
    res = json.load(open(OUT_SIGMA)) if os.path.exists(OUT_SIGMA) else []
    done = {(r["sigma_um"], r["seed"], r["direction"]) for r in res}
    tasks = [(s, k, d) for s in sigmas for k in seeds for d in directions
             if (s * 1e6, k, d) not in done]
    print(f"{len(tasks)} görev (her biri 2 izleme)", flush=True)
    with ProcessPoolExecutor(workers) as pool:
        for r in pool.map(_sig_task, tasks):
            res.append(r); json.dump(res, open(OUT_SIGMA, "w"), indent=1)
            print(f"σ={r['sigma_um']:4.1f} μm seed {r['seed']} yön {r['direction']:+d}: f* = {r['f_target']:9.3f} × hedef "
                  f"(δ* = {r['delta_star']:+.2e}; f0 = {r['f0']/1e-9:.1f}, ω0 = {r['w0']:+.3f}, ω1 = {r['w1']:+.2e})",
                  flush=True)
    # σ² üssü
    for d in directions:
        rows = [r for r in res if r["direction"] == d]
        xs = np.log([r["sigma_um"] for r in rows]); ys = np.log([abs(r["f"]) for r in rows])
        p, c = np.polyfit(xs, ys, 1)
        per_seed = {}
        for k in seeds:
            rr = sorted([r for r in rows if r["seed"] == k], key=lambda r: r["sigma_um"])
            if len(rr) >= 2:
                per_seed[k] = float(np.polyfit(np.log([r["sigma_um"] for r in rr]),
                                               np.log([abs(r["f"]) for r in rr]), 1)[0])
        print(f"yön {d:+d}: σ² üssü (tümü) p = {p:.4f}; seed başına {per_seed}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sigma"])
    ap.add_argument("-w", "--workers", type=int, default=4)
    ap.add_argument("--directions", type=int, nargs="+", default=[-1])
    a = ap.parse_args()
    run_sigma(a.workers, directions=tuple(a.directions))
