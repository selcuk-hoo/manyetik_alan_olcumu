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


def projectors():
    Ps = np.zeros((NQ, NQ)); Pa = np.zeros((NQ, NQ))
    for k in range(NQ // 2):
        a, b = 2 * k, 2 * k + 1
        Ps[a, a] = Ps[b, b] = Ps[a, b] = Ps[b, a] = 0.5
        Pa[a, a] = Pa[b, b] = 0.5
        Pa[a, b] = Pa[b, a] = -0.5
    return Ps, Pa


def sym_basis():
    """Simetrik (24 boyut) ve antisimetrik (24 boyut) ortonormal tabanlar, 48×24."""
    Bs = np.zeros((NQ, NQ // 2)); Ba = np.zeros((NQ, NQ // 2))
    for k in range(NQ // 2):
        Bs[2 * k, k] = Bs[2 * k + 1, k] = 1 / np.sqrt(2)
        Ba[2 * k, k] = 1 / np.sqrt(2); Ba[2 * k + 1, k] = -1 / np.sqrt(2)
    return Bs, Ba


def load_mats():
    """Kuadrupol (R) ve saptırıcı (D) tepki matrisleri; düzlem başına:
    x: kuadrupol dx + saptırıcı ε; y: kuadrupol dy + saptırıcı roll φ."""
    M = {}
    for pl, rk, dk in (("x", "dx", "eps"), ("y", "dy", "roll")):
        M[pl] = {"R_ccw": np.load(os.path.join(BASE, f"R_{rk}_1.npy")),
                 "R_cw": np.load(os.path.join(BASE, f"R_{rk}_cw.npy")),
                 "D_ccw": np.load(DPATH(dk, "ccw")), "D_cw": np.load(DPATH(dk, "cw"))}
    return M


def principal_cos(A, B):
    """span(A) ile span(B) arasındaki asal açıların kosinüsleri (azalan)."""
    Qa, _ = np.linalg.qr(A); Qb, _ = np.linalg.qr(B)
    return np.linalg.svd(Qa.T @ Qb, compute_uv=False)


def run_fit():
    """Yığılmış fit matrisine saptırıcı sütunları eklenince rank/koşul/dejenerelik."""
    M = load_mats(); Bs, Ba = sym_basis(); out = {}
    for pl in ("x", "y"):
        m = M[pl]
        Sq = np.vstack([m["R_ccw"], m["R_cw"]])               # 96×48 kuadrupol
        Sd = np.vstack([m["D_ccw"], m["D_cw"]])               # 96×48 saptırıcı
        # birimleri eşitle: saptırıcı sütunlarını "tipik sürüklenme" ile ölçekle değil,
        # sütun normuyla normalize et (rank/açı birimden bağımsız; koşul için ikisi de)
        Sdn = Sd / np.linalg.norm(Sd, axis=0); Sqn = Sq / np.linalg.norm(Sq, axis=0)
        S = np.hstack([Sqn, Sdn])                              # 96×96
        sv = np.linalg.svd(S, compute_uv=False)
        sv_q = np.linalg.svd(Sqn, compute_uv=False)
        # toplam/fark kanalları
        sum_d = m["D_ccw"] + m["D_cw"]; dif_d = m["D_ccw"] - m["D_cw"]
        # dejenerelik: simetrik / antisimetrik kuadrupol alt-uzayı ile saptırıcı alt-uzayı
        cs = principal_cos(Sq @ Bs, Sd); ca = principal_cos(Sq @ Ba, Sd)
        # simetrik kuadrupol modlarının saptırıcılarla ne kadar taklit edilebildiği:
        # her simetrik birim deseni q için min_e ||Sq q − Sd e|| / ||Sq q||
        Pd = Sd @ np.linalg.pinv(Sd, rcond=1e-12)
        def mimic(B):
            r = [np.linalg.norm((np.eye(96) - Pd) @ (Sq @ B[:, k])) / np.linalg.norm(Sq @ B[:, k])
                 for k in range(B.shape[1])]
            return np.array(r)
        ms, ma = mimic(Bs), mimic(Ba)
        out[pl] = {
            "rank_S": int(np.sum(sv > 1e-10 * sv[0])), "cond_S": float(sv[0] / sv[-1]),
            "cond_quad_only": float(sv_q[0] / sv_q[-1]),
            "sv_S_smallest10_rel": (sv[-10:] / sv[0]).tolist(),
            "D_sum_over_diff_norm": float(np.linalg.norm(sum_d) / np.linalg.norm(dif_d)),
            "cos_sym_defl_top5": cs[:5].tolist(), "n_cos_sym_gt_0.99": int(np.sum(cs > 0.99)),
            "cos_anti_defl_top5": ca[:5].tolist(), "n_cos_anti_gt_0.99": int(np.sum(ca > 0.99)),
            "mimic_sym_median": float(np.median(ms)), "mimic_sym_min": float(ms.min()),
            "mimic_anti_median": float(np.median(ma)), "mimic_anti_min": float(ma.min()),
        }
        o = out[pl]
        print(f"[{pl}] rank(S 96×96) = {o['rank_S']}, cond(S) = {o['cond_S']:.3g} "
              f"(yalnız kuadrupol {o['cond_quad_only']:.3g}); |D_sum|/|D_diff| = "
              f"{o['D_sum_over_diff_norm']:.1f}")
        print(f"     simetrik↔saptırıcı asal kosinüs (ilk 5): {np.round(cs[:5],4)}; "
              f">0.99 olan {o['n_cos_sym_gt_0.99']}/24")
        print(f"     antisim↔saptırıcı asal kosinüs (ilk 5): {np.round(ca[:5],4)}; "
              f">0.99 olan {o['n_cos_anti_gt_0.99']}/24")
        print(f"     saptırıcıyla taklit edilemeyen kesir: simetrik medyan {o['mimic_sym_median']:.3f} "
              f"(min {o['mimic_sym_min']:.3f}), antisim medyan {o['mimic_anti_median']:.3f}")
    json.dump(out, open(os.path.join(OUTDIR, "adim2_fit.json"), "w"), indent=1)


RCONDS = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5, 1e-6]


def tsvd_solve(A, y, rcond):
    """Sütun-normalize TSVD çözümü (birimleri farklı sütunlar için)."""
    nrm = np.linalg.norm(A, axis=0)
    return (np.linalg.pinv(A / nrm, rcond=rcond) @ y) / nrm


def run_scan(nseed=200, sig_n=14e-9, drift=5e-6):
    """Saptırıcı sürüklenmesi (ε_k, φ_k ~ N(0,σ), bağımsız) + simetrik kuadrupol sürüklenmesi
    (5 μm) → iki demetli fit ile simetrik kestirim hatası.
      Q  : yalnız kuadrupol sütunları (makale yöntemi)  — saptırıcı sinyali sızar
      QD : kuadrupol + saptırıcı sütunları (96 bilinmeyen / 96 denklem)
    Her yöntemin rcond'u, saptırıcı sürüklenmesi YOKKEN değil, her σ'da ayrı en iyi seçilir."""
    M = load_mats(); Ps, _ = projectors()
    grids = {"x": np.logspace(-7, -5, 9),                  # ε (bağıl alan)
             "y": np.logspace(-7, -5, 9)}                  # φ [rad] = 0.1..10 μrad
    out = {"params": {"nseed": nseed, "sig_n": sig_n, "drift_sym_um": drift * 1e6}, "rows": []}
    for pl in ("x", "y"):
        m = M[pl]
        Sq = np.vstack([m["R_ccw"], m["R_cw"]]); Sd = np.vstack([m["D_ccw"], m["D_cw"]])
        Sqd = np.hstack([Sq, Sd])
        for sd in np.concatenate([[0.0], grids[pl]]):
            errs = {"Q": {}, "QD": {}}
            for rc in RCONDS:
                eq, eqd = [], []
                for s_ in range(nseed):
                    rng = np.random.default_rng(20_000 + s_)
                    dm = Ps @ rng.normal(0, 1, NQ); dm *= drift / np.std(dm)
                    de = rng.normal(0, 1, NQ) * sd
                    y = Sq @ dm + Sd @ de + rng.normal(0, sig_n, 2 * NQ)
                    mq = tsvd_solve(Sq, y, rc)
                    mqd = tsvd_solve(Sqd, y, rc)[:NQ]
                    eq.append(np.std(Ps @ (dm - mq))); eqd.append(np.std(Ps @ (dm - mqd)))
                errs["Q"][rc] = np.array(eq); errs["QD"][rc] = np.array(eqd)
            row = {"plane": pl, "sigma_defl": float(sd)}
            for k in ("Q", "QD"):
                best = min(RCONDS, key=lambda rc: np.median(errs[k][rc]))
                a = errs[k][best] * 1e6
                row[f"{k}_rcond"] = best
                row[f"{k}_med"] = float(np.median(a))
                row[f"{k}_p16"] = float(np.percentile(a, 16))
                row[f"{k}_p84"] = float(np.percentile(a, 84))
            # saptırıcı kaynaklı yörünge (düzeltme öncesi) rms — ölçek fikri için
            rng = np.random.default_rng(1)
            row["orbit_from_defl_um"] = float(np.std(Sd @ (rng.normal(0, 1, NQ) * sd)) * 1e6)
            out["rows"].append(row)
            print(f"[{pl}] σ_defl={sd:.1e}: simetrik hata Q {row['Q_med']:.3f} μm "
                  f"(rc {row['Q_rcond']:.0e}) | QD {row['QD_med']:.3f} μm (rc {row['QD_rcond']:.0e})"
                  f" | saptırıcı yörüngesi {row['orbit_from_defl_um']:.2f} μm", flush=True)
    json.dump(out, open(os.path.join(OUTDIR, "adim2_scan.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["kick", "matrix", "fit", "scan"])
    ap.add_argument("-w", "--workers", type=int, default=4)
    a = ap.parse_args()
    if a.mode == "kick":
        run_kick(a.workers)
    elif a.mode == "matrix":
        run_matrix(a.workers)
    elif a.mode == "fit":
        run_fit()
    elif a.mode == "scan":
        run_scan()
