#!/usr/bin/env python3
"""Hızlı estimator sürücüsü (foreground, azaltılmış CO-tur sayısı).

measure_false_edm'i n_turns=14 CO + t2=3e-4 ile çağırır (orijinal 28/5e-4).
p≈2 ve mutlak ölçek AYNI koşumda doğrulanır; p≈2 çıkarsa azaltılmış ayar
göreli karşılaştırmalar için geçerlidir.
"""
import os, sys, json, time, argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# BASE = repo kökü (bu betik kmod_drivers/ altında). Taşınabilir: __file__'dan türet.
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # kmod_drivers/
sys.path.insert(0, os.path.join(BASE, "berry_data"))
sys.path.insert(0, BASE)
os.makedirs("/tmp/kmod_recover", exist_ok=True)                  # çıktı (ephemeral)
import ac_bba_linchpin as L

NQ = L.NQ
CFG = L.CFG


def fast_measure(dx, dy, tilt=None, n_turns=14, t2=3e-4, direction=None, gflip=False,
                 gscale=1.0, dG=None):
    """Azaltılmış CO-tur ile measure_false_edm eşdeğeri (tilt/yön/polarite opsiyonel).

    direction: +1/-1 (CW/CCW); gflip=True → tüm quad polariteleri çevrilir (g→-g).
    gscale: tüm gradyanlara çarpan (flip-kalibrasyon hatası testi: gflip+gscale=1+ε).
    dG: per-quad fraksiyonel gradyan sapması (β-beat), boy 2*nFODO. Polarite-flip'te
        gradyanla birlikte döner (bağıl hata korunur) — flip dejenerasyonuna uyar.
    """
    from false_edm_mode_scan import setup_fields, _make_state, measure_dSy_dt_model
    from false_edm_4d import find_co_4d, _T_rev
    from integrator import integrate_particle
    import json as _j
    with open(os.path.join(BASE, "params.json")) as f:
        cfg = _j.load(f)
    if direction is not None:
        cfg["direction"] = float(direction)
    if gflip:                                    # polarite çevirme (CW/CCW telafi)
        cfg["g1"] = -float(cfg.get("g1", 0.21))
        cfg["g0"] = -float(cfg.get("g0", cfg["g1"]))
    if gscale != 1.0:                            # flip-kalibrasyon hatası
        cfg["g1"] = float(cfg.get("g1", 0.21)) * gscale
        cfg["g0"] = float(cfg.get("g0", cfg["g1"])) * gscale
    DT = float(cfg["dt"])
    fields, y0, beta0, R0, p_mag, direction = setup_fields(cfg)
    T_rev = _T_rev(fields, beta0, R0)
    if tilt is None:
        tilt = np.zeros(NQ)
    v_co, resid = find_co_4d(fields, p_mag, direction, dx, dy, tilt, T_rev,
                             n_turns=n_turns, n_iter=2, dG=dG)
    launch = _make_state(v_co, p_mag, direction, [0.0, 0.0, direction])
    fields.poincare_quad_index = 0.0
    _, poin, pt = integrate_particle(launch, 0.0, t2, DT, fields=fields,
                                     return_steps=4000, quad_dx=dx, quad_dy=dy,
                                     quad_tilt=tilt, quad_dG=dG)
    sv = np.asarray(poin[:, 7], float)
    slope = float(measure_dSy_dt_model(sv, np.asarray(pt, float)))
    return abs(slope), resid


def _w(task):
    import tempfile
    os.chdir(tempfile.mkdtemp())
    kind, payload = task
    if kind == "white":
        sg, seed = payload
        rng = np.random.default_rng(3000 + seed)
        dx = rng.normal(0, sg, NQ); dy = rng.normal(0, sg, NQ)
    elif kind == "resid":
        eps, seed = payload
        rng = np.random.default_rng(5000 + seed)
        dx0 = rng.normal(0, 10e-6, NQ); dy0 = rng.normal(0, 10e-6, NQ)
        dx, dy = L.bba_residual_offset(CFG, dx0, dy0, 0.02, eps, 1e-6, 1.0, rng)
    elif kind == "tilt":  # (mis_sigma, theta, seed) — sabit misalignment + tilt
        mis_sigma, theta, seed = payload
        rng = np.random.default_rng(7000 + seed)
        dx = rng.normal(0, mis_sigma, NQ) if mis_sigma > 0 else np.zeros(NQ)
        dy = rng.normal(0, mis_sigma, NQ) if mis_sigma > 0 else np.zeros(NQ)
        tilt = rng.normal(0, theta, NQ) if theta > 0 else np.zeros(NQ)  # İŞARETLİ
        f, resid = fast_measure(dx, dy, tilt=tilt)
        return kind, payload, f, resid
    else:  # cwccw: (theta, direction, gflip, seed) — AYNI desen, yön/polarite tara
        theta, direction, gflip, seed = payload
        rng = np.random.default_rng(7000 + seed)          # tilt mode ile AYNI desen
        dx = rng.normal(0, 10e-6, NQ); dy = rng.normal(0, 10e-6, NQ)
        tilt = rng.normal(0, theta, NQ) if theta > 0 else np.zeros(NQ)
        f, resid = fast_measure(dx, dy, tilt=tilt, direction=direction, gflip=gflip)
        return kind, payload, f, resid
    f, resid = fast_measure(dx, dy)
    return kind, payload, f, resid


def run(tasks, workers):
    out = []
    import build_response_matrix as brm
    with ProcessPoolExecutor(workers, initializer=brm._worker_init) as pool:
        for r in pool.map(_w, tasks):
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["calib", "sweep", "tilt", "cwccw"])
    ap.add_argument("-w", "--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()

    if args.mode == "calib":
        sigmas = [10e-6, 5e-6]
        tasks = [("white", (sg, sd)) for sg in sigmas for sd in range(args.seeds)]
        res = run(tasks, args.workers)
        d = {}
        for kind, (sg, sd), f, resid in res:
            d.setdefault(sg, []).append(f)
        print("=== KALİBRASYON (azaltılmış ayar; p≈2 doğrulaması) ===")
        for sg in sigmas:
            a = np.array(d[sg])
            print(f"  σ={sg*1e6:4.1f}μm: |f|={a.mean():.3e} ± {a.std():.1e} rad/s (n={len(a)})")
        m10, m5 = np.mean(d[10e-6]), np.mean(d[5e-6])
        p = np.log(m10/m5)/np.log(2.0)
        A = m10/(10e-6)**2
        print(f"  p (10→5μm) = {p:.3f}   (geometrik faz beklenen 2.00)")
        print(f"  ölçek A = f/σ² = {A:.3e} rad/s/m²   f@10μm={A*(10e-6)**2:.3e}")
        np.save("/tmp/kmod_recover/calib_fast.npy",
                np.array([[10e-6, m10], [5e-6, m5]]))

    if args.mode == "sweep":
        epss = [0.0, 0.01, 0.05]
        tasks = [("resid", (eps, sd)) for eps in epss for sd in range(args.seeds)]
        res = run(tasks, args.workers)
        d = {}
        for kind, (eps, sd), f, resid in res:
            d.setdefault(eps, []).append(f)
        print("=== LINCHPIN SWEEP (β-beating → kalan sahte-EDM) ===")
        print(f"  Hedef 1 nrad/s = 1e-9 ; gerçek EDM = 9.81e-10 rad/s")
        print(f"  {'β-beat':>8} {'|sahte-EDM| rad/s':>22} {'hedef<1e-9?':>12}")
        rows = []
        for eps in epss:
            a = np.array(d[eps])
            ok = "EVET" if a.mean() < 1e-9 else "HAYIR"
            print(f"  {eps*100:6.1f}% {a.mean():.3e} ± {a.std():.1e}    {ok}")
            rows.append([eps, a.mean(), a.std()])
        np.save("/tmp/kmod_recover/sweep_fast.npy", np.array(rows))

    if args.mode == "tilt":
        # (mis_sigma, theta) çiftleri: pure tilt, residual±tilt, full+tilt
        cfgs = [(0.0, 1e-3),          # saf tilt 1 mrad (on-axis → ~0 bekle)
                (0.3e-6, 0.0),        # residual baseline (tilt yok)
                (0.3e-6, 1e-4),       # residual + 0.1 mrad tilt
                (0.3e-6, 1e-3),       # residual + 1 mrad tilt
                (10e-6, 1e-3)]        # full misalignment + 1 mrad tilt
        nseed = max(2, args.seeds if args.seeds <= 2 else 2)
        tasks = [("tilt", (m, th, sd)) for (m, th) in cfgs for sd in range(nseed)]
        res = run(tasks, args.workers)
        d = {}
        for kind, (m, th, sd), f, resid in res:
            d.setdefault((m, th), []).append(f)
        print("=== QUAD-TİLT SİSTEMATİĞİ (sahte-EDM vs tilt) ===")
        print(f"  Hedef 1 nrad/s = 1e-9 ; gerçek EDM = 9.81e-10 rad/s")
        print(f"  {'mis σ':>8} {'tilt θ':>10} {'|sahte-EDM| rad/s':>22} {'hedef?':>8}")
        rows = []
        for (m, th) in cfgs:
            a = np.array(d[(m, th)])
            ok = "EVET" if a.mean() < 1e-9 else "HAYIR"
            print(f"  {m*1e6:6.2f}μm {th*1e3:7.2f}mrad {a.mean():.3e} ± {a.std():.1e}   {ok}")
            rows.append([m, th, a.mean(), a.std()])
        np.save("/tmp/kmod_recover/tilt_fast.npy", np.array(rows))

    if args.mode == "cwccw":
        # AYNI 10μm misalignment deseni (seed=0); tilt yok vs 1 mrad;
        # yön (CW/CCW) × polarite (g flip) 4'lü kombinasyon.
        seed = 0
        combos = [(+1, False), (-1, False), (+1, True), (-1, True)]
        thetas = [0.0, 1e-3]               # tilt=0 (kontrol) ve 1 mrad (soru)
        tasks = [("cwccw", (th, d, gf, seed)) for th in thetas for (d, gf) in combos]
        res = run(tasks, args.workers)
        F = {}
        for kind, (th, d, gf, sd), f, resid in res:
            F[(th, d, gf)] = f
        print("=== CW/CCW + QUAD-FLIP, TİLT ETKİSİNİ GİDERİR Mİ? ===")
        print("  Aynı 10μm misalignment deseni; sahte-EDM (EDMSwitch=0).")
        print(f"  {'tilt':>8} {'f(CW,g+)':>11} {'f(CCW,g+)':>11} {'f(CW,g-)':>11} "
              f"{'f(CCW,g-)':>11}")
        for th in thetas:
            fpp = F[(th, +1, False)]; fmp = F[(th, -1, False)]
            fpm = F[(th, +1, True)]; fmm = F[(th, -1, True)]
            print(f"  {th*1e3:5.1f}mr {fpp:+.3e} {fmp:+.3e} {fpm:+.3e} {fmm:+.3e}")
        print("\n  Telafi analizi (sahte-EDM'in EDM-kanalına sızan ARTIĞI):")
        print(f"  {'tilt':>8} {'ham |f|':>11} {'(CW-CCW)/2':>12} {'4-lü telafi':>12} "
              f"{'kazanç':>8}")
        rows = []
        for th in thetas:
            fpp = F[(th, +1, False)]; fmp = F[(th, -1, False)]
            fpm = F[(th, +1, True)]; fmm = F[(th, -1, True)]
            raw = abs(fpp)
            cwccw = abs((fpp - fmp) / 2)                    # CW/CCW differential
            fourfold = abs((fpp - fmp - fpm + fmm) / 4)     # 4-lü kombinasyon
            gain = raw / cwccw if cwccw > 0 else float("inf")
            print(f"  {th*1e3:5.1f}mr {raw:.3e} {cwccw:12.3e} {fourfold:12.3e} {gain:7.1f}×")
            rows.append([th, raw, cwccw, fourfold])
        # dejenerasyon kontrolü CCW ≡ CW+flip
        for th in thetas:
            fmp = F[(th, -1, False)]; fpm = F[(th, +1, True)]
            print(f"  [dejenerasyon th={th*1e3:.0f}mr] f(CCW,g+)={fmp:+.2e} vs "
                  f"f(CW,g-)={fpm:+.2e}  oran={fmp/fpm if fpm else 0:+.2f}")
        np.save("/tmp/kmod_recover/cwccw_fast.npy", np.array(rows))
    print(f"[toplam {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
