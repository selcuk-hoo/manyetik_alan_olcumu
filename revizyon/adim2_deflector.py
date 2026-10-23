#!/usr/bin/env python3
"""adim2_deflector.py — PRAB revizyonu Adım 2: elektrostatik saptırıcı hataları (Hakem A, madde 1).

Motor: revizyon/integrator_rev.cpp (saptırıcı başına ELEKTRİK alan hatası ε ve roll φ).
Mevcut betikler (build_response_matrix, false_edm_4d, fast_est) değiştirilmeden, modül
takası ile yeni motoru kullanır: sys.modules["integrator"] = integrator_rev.

Modlar:
  kick   : (i) sıfır testi: hata yokken yeni motor = eski motor
           (ii) tek saptırıcı ε=1e-4 ve φ=0.1 mrad → 48-BPM kapalı yörünge, CW ve CCW
           (iii) karşılaştırma: manyetik dipole_tilt (eski model) CCW'de işaret çevirir mi
  matrix : 48 saptırıcı için tepki matrisleri D_x (ε→x) ve D_y (φ→y), CW + CCW
  fit    : yığılmış fit matrisine saptırıcı sütunları ekle → rank, koşul, dejenerelik
  scan   : saptırıcı sürüklenme taraması (ε 1e-7..1e-5, φ 0.1..10 μrad) → simetrik izleme
"""
import os, sys, json, argparse, time
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REV = os.path.join(BASE, "revizyon")
for p in (REV, BASE, os.path.join(BASE, "kmod_drivers"), os.path.join(BASE, "berry_data")):
    sys.path.insert(0, p)
import integrator_rev
sys.modules["integrator"] = integrator_rev       # tüm alt betikler yeni motoru kullanır
OUTDIR = os.path.join(REV, "sonuclar")
NQ = 48


def cod_run(task):
    """Tek kapalı-yörünge koşumu (eksenden fırlatma + tur ortalaması, R üretimiyle aynı yöntem).
    task = (direction, quad_dy, quad_dx, dipole_tilt, defl_dE, defl_roll, t2, lib)"""
    import tempfile
    os.chdir(tempfile.mkdtemp())
    direction, qdy, qdx, dtilt, dE, roll, t2, lib = task
    if lib == "old":                               # eski motor (sıfır testi için)
        import importlib.util
        spec = importlib.util.spec_from_file_location("integrator_old",
                                                      os.path.join(BASE, "integrator.py"))
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        sys.modules["integrator"] = mod
    else:
        sys.modules["integrator"] = integrator_rev
    import build_response_matrix as brm
    # işçi süreçleri yeniden kullanıldığından motor HER görevde açıkça seçilir
    brm.integrate_particle = sys.modules["integrator"].integrate_particle
    cfg = json.load(open(os.path.join(BASE, "params.json")))
    cfg["direction"] = float(direction); cfg["t2"] = t2
    alanlar, st = brm.setup_fields(cfg)
    integrator_rev.DEFL_DE = np.array(dE); integrator_rev.DEFL_ROLL = np.array(roll)
    x, y = brm.run_sim(alanlar, st, cfg, np.array(qdy), np.array(qdx),
                       dipole_tilt=np.array(dtilt))
    return x, y


def run_kick(workers):
    from concurrent.futures import ProcessPoolExecutor
    z = np.zeros(NQ)
    qdy = z.copy(); qdy[7] = 10e-6
    k = 10                                          # test saptırıcısı (hücre 5, ARC1)
    e1 = z.copy(); e1[k] = 1e-4                      # ε = 1e-4
    r1 = z.copy(); r1[k] = 1e-4                      # φ = 0.1 mrad
    T2 = 3e-4
    tasks = {
        "null_old": (-1, qdy, z, z, z, z, T2, "old"),
        "null_rev": (-1, qdy, z, z, z, z, T2, "rev"),
        "dE_ccw": (-1, z, z, z, e1, z, T2, "rev"), "dE_cw": (+1, z, z, z, e1, z, T2, "rev"),
        "roll_ccw": (-1, z, z, z, z, r1, T2, "rev"), "roll_cw": (+1, z, z, z, z, r1, T2, "rev"),
        "mtilt_ccw": (-1, z, z, r1, z, z, T2, "rev"), "mtilt_cw": (+1, z, z, r1, z, z, T2, "rev"),
    }
    keys = list(tasks)
    with ProcessPoolExecutor(workers) as pool:
        res = dict(zip(keys, pool.map(cod_run, [tasks[k_] for k_ in keys])))
    out = {}
    xo, yo = res["null_old"]; xr, yr = res["null_rev"]
    out["null_max_diff_m"] = float(max(np.abs(xo - xr).max(), np.abs(yo - yr).max()))
    out["null_orbit_rms_m"] = float(np.std(yo))
    for name, plane in (("dE", 0), ("roll", 1), ("mtilt", 1)):
        a = np.array(res[f"{name}_ccw"][plane]); b = np.array(res[f"{name}_cw"][plane])
        out[name] = {"ccw_rms_um": float(np.std(a) * 1e6), "cw_rms_um": float(np.std(b) * 1e6),
                     "ccw_max_um": float(np.abs(a).max() * 1e6),
                     "cw_max_um": float(np.abs(b).max() * 1e6),
                     "corr_cw_ccw": float(np.corrcoef(a, b)[0, 1]),
                     "ccw_um": (a * 1e6).tolist(), "cw_um": (b * 1e6).tolist()}
    json.dump(out, open(os.path.join(OUTDIR, "adim2_kick.json"), "w"), indent=1)
    print(f"sıfır testi: max|Δorbit| eski/yeni = {out['null_max_diff_m']:.2e} m "
          f"(orbit rms {out['null_orbit_rms_m']*1e6:.2f} μm)")
    for name, lab in (("dE", "ε=1e-4 → x"), ("roll", "elektrik roll 0.1 mrad → y"),
                      ("mtilt", "manyetik dipole_tilt 0.1 mrad → y")):
        d = out[name]
        print(f"{lab}: CCW rms {d['ccw_rms_um']:.1f} μm (max {d['ccw_max_um']:.1f}), "
              f"CW rms {d['cw_rms_um']:.1f} μm (max {d['cw_max_um']:.1f}), "
              f"corr(CW,CCW) = {d['corr_cw_ccw']:+.3f}")


DPATH = lambda kind, d: os.path.join(OUTDIR, f"D_{kind}_{d}.npy")


def run_matrix(workers, delta=1e-5, t2=3e-4):
    """Saptırıcı tepki matrisleri: D_eps (ε_k → x_BPM) ve D_roll (φ_k → y_BPM), CW ve CCW.
    Birim: [m / birim ε] ve [m / rad]. Fark: (pert − ref)/delta, aynı fırlatma yöntemi."""
    from concurrent.futures import ProcessPoolExecutor
    z = np.zeros(NQ)
    for d, tag in ((-1, "ccw"), (+1, "cw")):
        if os.path.exists(DPATH("roll", tag)):
            print(tag, "mevcut"); continue
        tasks = [(d, z, z, z, z, z, t2, "rev")]
        for k in range(NQ):
            e = z.copy(); e[k] = delta
            tasks.append((d, z, z, z, e, z, t2, "rev"))
        for k in range(NQ):
            r = z.copy(); r[k] = delta
            tasks.append((d, z, z, z, z, r, t2, "rev"))
        t0 = time.time()
        with ProcessPoolExecutor(workers) as pool:
            res = list(pool.map(cod_run, tasks))
        x0, y0 = res[0]
        De = np.array([(res[1 + k][0] - x0) / delta for k in range(NQ)]).T
        Dr = np.array([(res[1 + NQ + k][1] - y0) / delta for k in range(NQ)]).T
        np.save(DPATH("eps", tag), De); np.save(DPATH("roll", tag), Dr)
        print(f"{tag}: D_eps, D_roll yazıldı ({time.time()-t0:.0f}s); "
              f"|D_eps| sütun rms {np.std(De,axis=0).mean():.3g} m, "
              f"|D_roll| sütun rms {np.std(Dr,axis=0).mean():.3g} m/rad", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["kick", "matrix", "fit", "scan"])
    ap.add_argument("-w", "--workers", type=int, default=4)
    a = ap.parse_args()
    if a.mode == "kick":
        run_kick(a.workers)
    elif a.mode == "matrix":
        run_matrix(a.workers)
