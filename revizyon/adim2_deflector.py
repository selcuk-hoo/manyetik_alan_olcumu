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
    direction, qdy, qdx, dtilt, dE, roll, t2, lib = task[:8]
    dG = np.array(task[8]) if len(task) > 8 and task[8] is not None else None
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
                       dipole_tilt=np.array(dtilt), quad_dG=dG)
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


def DPATH(kind, d, sig_g=0.0):
    suf = f"_g{int(round(sig_g*1e4))}" if sig_g > 0 else ""
    return os.path.join(OUTDIR, f"D_{kind}_{d}{suf}.npy")


def run_matrix(workers, delta=1e-5, t2=3e-4, sig_g=0.0):
    """Saptırıcı tepki matrisleri: D_eps (ε_k → x_BPM) ve D_roll (φ_k → y_BPM), CW ve CCW.
    Birim: [m / birim ε] ve [m / rad]. Fark: (pert − ref)/delta, aynı fırlatma yöntemi."""
    from concurrent.futures import ProcessPoolExecutor
    z = np.zeros(NQ)
    dG = None
    if sig_g > 0:
        from build_rtrue import dG_pattern
        dG = dG_pattern(sig_g)
    for d, tag in ((-1, "ccw"), (+1, "cw")):
        if os.path.exists(DPATH("roll", tag, sig_g)):
            print(tag, "mevcut"); continue
        tasks = [(d, z, z, z, z, z, t2, "rev", dG)]
        for k in range(NQ):
            e = z.copy(); e[k] = delta
            tasks.append((d, z, z, z, e, z, t2, "rev", dG))
        for k in range(NQ):
            r = z.copy(); r[k] = delta
            tasks.append((d, z, z, z, z, r, t2, "rev", dG))
        t0 = time.time()
        with ProcessPoolExecutor(workers) as pool:
            res = list(pool.map(cod_run, tasks))
        x0, y0 = res[0]
        De = np.array([(res[1 + k][0] - x0) / delta for k in range(NQ)]).T
        Dr = np.array([(res[1 + NQ + k][1] - y0) / delta for k in range(NQ)]).T
        np.save(DPATH("eps", tag, sig_g), De); np.save(DPATH("roll", tag, sig_g), Dr)
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


def run_scan(nseed=200, sig_n=14e-9, drift=5e-6, sig_g=0.0):
    """Saptırıcı sürüklenmesi (ε_k, φ_k ~ N(0,σ), bağımsız) + simetrik kuadrupol sürüklenmesi
    (5 μm) → iki demetli fit ile simetrik kestirim hatası.
      Q  : yalnız kuadrupol sütunları (makale yöntemi)  — saptırıcı sinyali sızar
      QD : kuadrupol + saptırıcı sütunları (96 bilinmeyen / 96 denklem)
    Her yöntemin rcond'u, saptırıcı sürüklenmesi YOKKEN değil, her σ'da ayrı en iyi seçilir."""
    M = load_mats(); Ps, _ = projectors()
    # gerçek makine (okuma) matrisleri: β-beat'li R_true, D_true; kestirici NOMİNAL M kullanır
    T = M
    if sig_g > 0:
        from build_rtrue import path as rtp
        T = {}
        for pl, rk, dk in (("x", "dx", "eps"), ("y", "dy", "roll")):
            T[pl] = {"R_ccw": np.load(rtp(rk, "ccw", sig_g)), "R_cw": np.load(rtp(rk, "cw", sig_g)),
                     "D_ccw": np.load(DPATH(dk, "ccw", sig_g)), "D_cw": np.load(DPATH(dk, "cw", sig_g))}
    grids = {"x": np.logspace(-7, -5, 9),                  # ε (bağıl alan)
             "y": np.logspace(-7, -5, 9)}                  # φ [rad] = 0.1..10 μrad
    out = {"params": {"nseed": nseed, "sig_n": sig_n, "drift_sym_um": drift * 1e6,
                      "sig_g": sig_g}, "rows": []}
    for pl in ("x", "y"):
        m = M[pl]
        Sq = np.vstack([m["R_ccw"], m["R_cw"]]); Sd = np.vstack([m["D_ccw"], m["D_cw"]])
        Sqd = np.hstack([Sq, Sd])
        Tq = np.vstack([T[pl]["R_ccw"], T[pl]["R_cw"]]); Td = np.vstack([T[pl]["D_ccw"], T[pl]["D_cw"]])
        for sd in np.concatenate([[0.0], grids[pl]]):
            errs = {"Q": {}, "QD": {}}
            for rc in RCONDS:
                nq = np.linalg.norm(Sq, axis=0); nqd = np.linalg.norm(Sqd, axis=0)
                Pq = np.linalg.pinv(Sq / nq, rcond=rc) / nq[:, None]
                Pqd = (np.linalg.pinv(Sqd / nqd, rcond=rc) / nqd[:, None])[:NQ]
                eq, eqd = [], []
                for s_ in range(nseed):
                    rng = np.random.default_rng(20_000 + s_)
                    dm = Ps @ rng.normal(0, 1, NQ); dm *= drift / np.std(dm)
                    de = rng.normal(0, 1, NQ) * sd
                    y = Tq @ dm + Td @ de + rng.normal(0, sig_n, 2 * NQ)
                    mq = Pq @ y
                    mqd = Pqd @ y
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
            row["orbit_from_defl_um"] = float(np.std(Td @ (rng.normal(0, 1, NQ) * sd)) * 1e6)
            out["rows"].append(row)
            print(f"[{pl}] σ_defl={sd:.1e}: simetrik hata Q {row['Q_med']:.3f} μm "
                  f"(rc {row['Q_rcond']:.0e}) | QD {row['QD_med']:.3f} μm (rc {row['QD_rcond']:.0e})"
                  f" | saptırıcı yörüngesi {row['orbit_from_defl_um']:.2f} μm", flush=True)
    suf = (f"_g{int(round(sig_g*1e4))}" if sig_g > 0 else "") + f"_n{sig_n*1e9:.0f}nm"
    json.dump(out, open(os.path.join(OUTDIR, f"adim2_scan{suf}.json"), "w"), indent=1)


EDM2_JSONL = os.path.join(OUTDIR, "adim2_edm.jsonl")
SIG_EPS, SIG_ROLL = 3e-7, 3e-7          # sızmanın ~3 μm simetrik hataya ulaştığı seviye


def _edm2_case(seed, case, sig_n=14e-9, drift=5e-6, rc_q=1e-2, rc_qd=1e-4):
    """Bir seed için (dx', dy', ε, φ): sahte EDM'ye girecek kuadrupol artıkları + saptırıcı hataları.
    case: 'defl'  — yalnız saptırıcı sürüklenmesi (kuadrupoller kusursuz)
          'none'  — simetrik kuadrupol sürüklenmesi + saptırıcı sürüklenmesi, düzeltme yok
          'Q'     — makale yöntemi (yalnız kuadrupol sütunları) ile kuadrupol düzeltmesi
          'QD'    — saptırıcı sütunlu fit ile kuadrupol düzeltmesi (saptırıcılar düzeltilmez)"""
    M = load_mats(); Ps, _ = projectors()
    rng = np.random.default_rng(30_000 + seed)
    out = {}
    for pl in ("x", "y"):
        m = M[pl]
        Sq = np.vstack([m["R_ccw"], m["R_cw"]]); Sd = np.vstack([m["D_ccw"], m["D_cw"]])
        dm = Ps @ rng.normal(0, 1, NQ); dm *= drift / np.std(dm)
        de = rng.normal(0, 1, NQ) * (SIG_EPS if pl == "x" else SIG_ROLL)
        y = Sq @ dm + Sd @ de + rng.normal(0, sig_n, 2 * NQ)
        if case == "defl":
            res = np.zeros(NQ)
        elif case == "none":
            res = dm
        elif case == "Q":
            res = dm - tsvd_solve(Sq, y, rc_q)
        else:
            res = dm - tsvd_solve(np.hstack([Sq, Sd]), y, rc_qd)[:NQ]
        out[pl] = (res, de)
    return out["x"][0], out["y"][0], out["x"][1], out["y"][1]


def _edm2_task(task):
    import tempfile
    os.chdir(tempfile.mkdtemp())
    sys.modules["integrator"] = integrator_rev
    from fast_est import fast_measure
    seed, case = task
    dx, dy, eps, roll = _edm2_case(seed, case)
    integrator_rev.DEFL_DE = eps; integrator_rev.DEFL_ROLL = roll
    fcw, _ = fast_measure(dx, dy, direction=+1, signed=True, n_turns=28, n_iter=4)
    fccw, _ = fast_measure(dx, dy, direction=-1, signed=True, n_turns=28, n_iter=4)
    Ps, _ = projectors()
    return dict(seed=seed, case=case, f_cw=fcw / 1e-9, f_ccw=fccw / 1e-9,
                C=abs(fcw - fccw) / 2 / 1e-9,
                sym_x=float(np.std(Ps @ dx) * 1e6), sym_y=float(np.std(Ps @ dy) * 1e6),
                sig_eps=SIG_EPS, sig_roll=SIG_ROLL)


def run_edm2(seeds, workers):
    from concurrent.futures import ProcessPoolExecutor
    done = set()
    if os.path.exists(EDM2_JSONL):
        done = {(json.loads(l)["seed"], json.loads(l)["case"]) for l in open(EDM2_JSONL)}
    tasks = [(s_, c) for s_ in seeds for c in ("defl", "none", "Q", "QD") if (s_, c) not in done]
    print(f"{len(tasks)} görev", flush=True)
    with ProcessPoolExecutor(workers) as pool:
        for r in pool.map(_edm2_task, tasks):
            open(EDM2_JSONL, "a").write(json.dumps(r) + "\n")
            print(f"  seed {r['seed']} {r['case']}: C={r['C']:.4f}× f_CW={r['f_cw']:.3f} "
                  f"f_CCW={r['f_ccw']:.3f} (sim {r['sym_x']:.2f}/{r['sym_y']:.2f} μm)", flush=True)


def run_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9})
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True)
    styles = {("", "14nm"): ("#0072B2", "-"), ("_g20", "14nm"): ("#0072B2", "--"),
              ("", "1000nm"): ("#D55E00", "-"), ("_g20", "1000nm"): ("#D55E00", "--")}
    for (g, n), (c, ls) in styles.items():
        f = os.path.join(OUTDIR, f"adim2_scan{g}_n{n}.json")
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        lab = (f"noise {'14 nm' if n=='14nm' else '1 $\\mu$m'}"
               + (", grad. err. 0.2%" if g else ", exact model"))
        for ax, pl in zip(axs, ("x", "y")):
            rows = [r for r in d["rows"] if r["plane"] == pl and r["sigma_defl"] > 0]
            sd = np.array([r["sigma_defl"] for r in rows])
            xs = sd if pl == "x" else sd * 1e6
            ax.plot(xs, [r["Q_med"] for r in rows], ls=ls, color=c, marker="o", ms=3,
                    label="quad-only fit (paper), " + lab)
            ax.plot(xs, [r["QD_med"] for r in rows], ls=ls, color=c, marker="D", ms=3,
                    alpha=0.55, label="quad + deflector fit, " + lab)
    for ax, xl, t in zip(axs, (r"deflector field drift $\sigma_\varepsilon$ (relative)",
                               r"deflector roll drift $\sigma_\phi$ [$\mu$rad]"),
                         ("(a) horizontal", "(b) vertical")):
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(xl); ax.set_title(t, fontsize=9)
        ax.grid(alpha=0.3, which="both", lw=0.4)
    axs[0].set_ylabel(r"symmetric reconstruction error [$\mu$m]")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=6.5, bbox_to_anchor=(0.5, -0.2))
    fig.suptitle(r"Two-beam fit with deflector drift; symmetric quad drift 5 $\mu$m, 200 seeds (median)",
                 fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"adim2_scan.{ext}"), dpi=200, bbox_inches="tight")
    print("adim2_scan.pdf/png yazıldı")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["kick", "matrix", "fit", "scan", "edm", "plot"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("-w", "--workers", type=int, default=4)
    ap.add_argument("--sig-g", type=float, default=0.0, help="β-beat'li gerçek makine (σ_g)")
    ap.add_argument("--sig-n", type=float, default=14e-9)
    a = ap.parse_args()
    if a.mode == "kick":
        run_kick(a.workers)
    elif a.mode == "matrix":
        run_matrix(a.workers, sig_g=a.sig_g)
    elif a.mode == "fit":
        run_fit()
    elif a.mode == "scan":
        run_scan(sig_n=a.sig_n, sig_g=a.sig_g)
    elif a.mode == "edm":
        run_edm2(a.seeds, a.workers)
    elif a.mode == "plot":
        run_plot()
