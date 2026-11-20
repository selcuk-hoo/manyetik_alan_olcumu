#!/usr/bin/env python3
"""kusurlu_polarite.py — Omarov'un polarite çevirmeli birleşimi kusurlu olunca ne kalır, izleyici ne katar?

Bulgu (saptirici_testleri T4, teshis_statik_saptirici, MAC_OZET_v2 İş B): modelimizde kuadrupol polaritesi
çevrilince f_CW(−g) = f_CCW(+g) birebir; Omarov birleşimi E = ½(odd₊ + odd₋) (odd = ½(f_CW − f_CCW))
saptırıcı ve kuadrupol kaynaklı sahte EDM'yi tabana indirir. Gerçekte iki kusur var:
  A) Polariteler arası gradyan büyüklüğü farkı: çevrilmiş koşuda g → −g(1+η), η = 1e-4, 1e-3.
  B) İki polarite koşusu arasında sürüklenme: kuadrupol Δq (rms 0.3 μm, bağımsız; ~20 dk ATL mertebesi,
     tahmin) ve saptırıcı Δε, Δφ (rms 1e-7). Üç durum: düzeltme yok / tek demetli izleyici (E2) /
     iki demetli izleyici (E4) ikinci koşudan önce Δq'yu kestirip düzeltir (saptırıcı sürüklenmesinde
     E4 saptırıcı sütunlu fit + trim).
İzleyici modeli: BPM'ler kuadrupole rijit bağlı (ρ = 1): Δr_b = (R_b − I)Δq − δ + n; δ: BPM–kuadrupol göreli
kayması (rms 0.05 μm koşular arası), n: 14 nm. E2: yalnız CCW, E4: iki demet; TSVD 1e-4 (nominal model;
gerçek okuma %0.2 gradyan hatalı matrisle). Spin: C++ donmuş spin (14 tur/2 iter), dG = %0.2 desen.
Başlangıç durumu S0: kuadrupol artığı 0.3 μm (BBA sonrası), statik saptırıcı σ_d ∈ {0, 1e-5} (ortalama
çıkarılmış), demet düzelticilerle kuadrupol merkezlerinden geçirilmiş (teshis_statik_saptirici.state).
Çıktı: sonuclar/kusurlu_polarite.jsonl.  Kullanım: python3 kusurlu_polarite.py -w 4 ; ... --ozet
"""
import os, sys, json, argparse, tempfile
import numpy as np

REV = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(REV)
sys.path[:0] = [REV, os.path.join(REV, "tests")]
OUT = os.path.join(REV, "sonuclar", "kusurlu_polarite.jsonl")
NQ = 48
SEEDS = range(3)
SIG_Q0, DQ, DDEF, DELTA_BPM, SIG_N = 0.3e-6, 0.3e-6, 1e-7, 0.05e-6, 14e-9


def _one(dx, dy, direction, delta, dG, gscale, n_turns=14, n_iter=2, t2=3e-4):
    import sahte_edm_donmus as SD
    import integrator_rev, false_edm_4d, false_edm_mode_scan
    ms = SD.launcher(delta)
    false_edm_4d._make_state = ms; false_edm_mode_scan._make_state = ms
    from false_edm_mode_scan import setup_fields, measure_dSy_dt_model
    from false_edm_4d import find_co_4d, _T_rev
    cfg = json.load(open(os.path.join(BASE, "params.json"))); cfg["direction"] = float(direction)
    for k in ("g0", "g1", "g2"):
        cfg[k] = float(cfg[k]) * gscale
    fields, y0, beta0, R0_, p_mag, d = setup_fields(cfg)
    z = np.zeros(NQ)
    v_co, _ = find_co_4d(fields, p_mag, d, dx, dy, z, _T_rev(fields, beta0, R0_), n_turns=n_turns,
                         n_iter=n_iter, dG=dG)
    launch = ms(v_co, p_mag, d, [0.0, 0.0, d])
    fields.poincare_quad_index = 0.0
    _, poin, pt = integrator_rev.integrate_particle(launch, 0.0, t2, float(cfg["dt"]), fields=fields,
                                                    return_steps=4000, quad_dx=dx, quad_dy=dy, quad_tilt=z,
                                                    quad_dG=dG)
    t = np.asarray(pt, float)
    f = float(measure_dSy_dt_model(np.asarray(poin[:, 7], float), t))
    ang = np.unwrap(np.arctan2(np.asarray(poin[:, 6], float), np.asarray(poin[:, 8], float) * d))
    return f, float(np.polyfit(t, ang, 1)[0])


def measure(dx, dy, direction, dG, gscale):
    import sahte_edm_donmus as SD
    cwd = os.getcwd(); os.chdir(tempfile.mkdtemp())
    try:
        f0, w0 = _one(dx, dy, direction, 0.0, dG, gscale)
        d1 = -w0 / SD.S_W_DELTA
        f1, w1 = _one(dx, dy, direction, d1, dG, gscale)
        fs = f0 + (f1 - f0) * (0.0 - w0) / (w1 - w0)
        ds = d1 * (0.0 - w0) / (w1 - w0)
        if abs(w1) > SD.W1_ESIK:
            fs, _ = _one(dx, dy, direction, ds, dG, gscale)
        return float(fs), float(w1)
    finally:
        os.chdir(cwd)


def states(seed, sig_d):
    """S0 (birinci polarite koşusu) ve ikinci koşu durumları {durum: (qx, qy, eps, roll)}."""
    import adim2_deflector as A2
    import teshis_statik_saptirici as TS
    from build_rtrue import path as rtp
    rng = np.random.default_rng(110_000 + seed)
    if sig_d > 0:
        e, c = TS.state(seed, sig_d)
    else:
        e = {pl: np.zeros(NQ) for pl in ("x", "y")}; c = {pl: np.zeros(NQ) for pl in ("x", "y")}
    q0 = {pl: c[pl] + rng.normal(0, SIG_Q0, NQ) for pl in ("x", "y")}   # düzeltici + BBA artığı
    dq = {pl: rng.normal(0, DQ, NQ) for pl in ("x", "y")}
    dd = {pl: rng.normal(0, DDEF, NQ) for pl in ("x", "y")}
    for pl in dd:
        dd[pl] -= dd[pl].mean()
    M = A2.load_mats(); I = np.eye(NQ)
    out = {"S0": (q0["x"], q0["y"], e["x"], e["y"])}
    q_un = {pl: q0[pl] + dq[pl] for pl in q0}
    out["dq_yok"] = (q_un["x"], q_un["y"], e["x"], e["y"])
    for meth in ("E2", "E4"):
        qc = {}
        for pl in ("x", "y"):
            Tq = {b: np.load(rtp("d" + pl, b, 0.002)) for b in ("ccw", "cw")}
            dl = rng.normal(0, DELTA_BPM, NQ)
            y = [(Tq[b] - I) @ dq[pl] - dl + rng.normal(0, SIG_N, NQ) for b in ("ccw", "cw")]
            if meth == "E2":
                est = A2.tsvd_solve(M[pl]["R_ccw"] - I, y[0], 1e-4)
            else:
                est = A2.tsvd_solve(np.vstack([M[pl]["R_ccw"] - I, M[pl]["R_cw"] - I]), np.r_[y[0], y[1]], 1e-4)
            qc[pl] = q0[pl] + dq[pl] - est
        out[f"dq_{meth}"] = (qc["x"], qc["y"], e["x"], e["y"])
    out["dd_yok"] = (q0["x"], q0["y"], e["x"] + dd["x"], e["y"] + dd["y"])
    ee = {}
    qq = {}
    for pl in ("x", "y"):
        m = M[pl]; dk = "eps" if pl == "x" else "roll"
        Td = {b: np.load(A2.DPATH(dk, b, 0.002)) for b in ("ccw", "cw")}
        y = np.r_[Td["ccw"] @ dd[pl], Td["cw"] @ dd[pl]] + rng.normal(0, SIG_N, 2 * NQ)
        S = np.hstack([np.vstack([m["R_ccw"], m["R_cw"]]), np.vstack([m["D_ccw"], m["D_cw"]])])
        est = A2.tsvd_solve(S, y, 1e-4)
        qq[pl] = q0[pl] - est[:NQ]; ee[pl] = e[pl] + dd[pl] - est[NQ:]
    out["dd_E4"] = (qq["x"], qq["y"], ee["x"], ee["y"])
    return out


SIGDS = (0.0, 1e-5)


def tasks():
    T = []
    for sig_d in SIGDS:
        for s in SEEDS:
            for d in (-1, 1):
                T.append(dict(sig_d=sig_d, seed=s, state="S0", gs=1.0, direction=d))     # birinci koşu
                for eta in (1e-4, 1e-3):
                    T.append(dict(sig_d=sig_d, seed=s, state="S0", gs=-(1 + eta), direction=d))
                T.append(dict(sig_d=sig_d, seed=s, state="S0", gs=-1.0, direction=d))    # kusursuz kontrol
                for st in ("dq_yok", "dq_E2", "dq_E4", "dd_yok", "dd_E4"):
                    if sig_d == 0.0 and st.startswith("dd"):
                        pass
                    T.append(dict(sig_d=sig_d, seed=s, state=st, gs=-1.0, direction=d))
    return T


def key(t):
    return (t["sig_d"], t["seed"], t["state"], t["gs"], t["direction"])


def _task(t):
    import integrator_rev
    from build_rtrue import dG_pattern
    qx, qy, ex, ey = states(t["seed"], t["sig_d"])[t["state"]]
    integrator_rev.DEFL_DE = np.asarray(ex, float); integrator_rev.DEFL_ROLL = np.asarray(ey, float)
    f, w1 = measure(np.asarray(qx), np.asarray(qy), t["direction"], dG_pattern(0.002), t["gs"])
    return dict(t, f=f, f_target=f / 1e-9, w1=w1)


def run(workers):
    from concurrent.futures import ProcessPoolExecutor
    done = set()
    if os.path.exists(OUT):
        done = {key(r) for r in map(json.loads, open(OUT))}
    T = [t for t in tasks() if key(t) not in done]
    print(f"{len(T)} görev", flush=True)
    with ProcessPoolExecutor(workers) as pool:
        for r in pool.map(_task, T):
            open(OUT, "a").write(json.dumps(r) + "\n")
            print(r["sig_d"], r["seed"], r["state"], r["gs"], r["direction"], f"f*={r['f_target']:.4g}", flush=True)
    ozet()


def ozet():
    rows = [json.loads(l) for l in open(OUT)]
    f = {}
    for r in rows:
        f.setdefault((r["sig_d"], r["seed"], r["state"], r["gs"]), {})[r["direction"]] = r["f_target"]
    odd = {k: (v[1] - v[-1]) / 2 for k, v in f.items() if len(v) == 2}
    print("E = ½|odd₊(S0) + odd₋(durum)|  (hedef biriminde; odd₊: birinci koşu, normal polarite)")
    allsig = sorted({r["sig_d"] for r in rows}); allseed = sorted({r["seed"] for r in rows})
    for sig_d in allsig:
        SEEDS = [s for s in allseed if (sig_d, s, "S0", 1.0) in odd]
        cases = [("S0", -1.0, "kusursuz çevirme"), ("S0", -1.0001, "η = 1e-4"), ("S0", -1.001, "η = 1e-3"),
                 ("dq_yok", -1.0, "Δq 0.3 μm, düzeltme yok"), ("dq_E2", -1.0, "Δq, tek demet izleyici"),
                 ("dq_E4", -1.0, "Δq, iki demet izleyici"), ("dd_yok", -1.0, "Δε,Δφ 1e-7, trim yok"),
                 ("dd_E4", -1.0, "Δε,Δφ, iki demet fit + trim")]
        ref = [abs(odd.get((sig_d, s, "S0", 1.0), np.nan)) for s in SEEDS]
        print(f"\nσ_d = {sig_d:g}: n={len(SEEDS)}  |odd₊| (polarite çevirmeden C) medyan {np.nanmedian(ref):.3g}")
        for st, gs, lab in cases:
            E = [abs(odd[(sig_d, s, "S0", 1.0)] + odd[(sig_d, s, st, gs)]) / 2 for s in SEEDS
                 if (sig_d, s, "S0", 1.0) in odd and (sig_d, s, st, gs) in odd]
            if E:
                print(f"  {lab:32s} n={len(E):2d} medyan {np.median(E):.3g}  %84 {np.percentile(E, 84):.3g}"
                      f"  maks {np.max(E):.3g}  hedef-üstü {sum(e > 1 for e in E)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("-w", type=int, default=4); ap.add_argument("--ozet", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", help="desen (seed) listesi; varsayılan 0 1 2")
    ap.add_argument("--sigd", type=float, nargs="+", help="statik saptırıcı σ_d listesi; varsayılan 0 1e-5")
    a = ap.parse_args()
    if a.seeds: SEEDS = a.seeds
    if a.sigd: SIGDS = tuple(a.sigd)
    ozet() if a.ozet else run(a.w)
