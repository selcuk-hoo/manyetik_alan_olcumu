#!/usr/bin/env python3
"""paper_runs.py — makale C++ figür verileri (makale_orbit_bastirma.md §5)

İki uzun koşum; sonuçlar artımlı JSONL (/tmp) + kalıcı JSON (bu klasörde,
git'e girer — .npy/.npz gitignore'da olduğundan JSON seçildi):

  sigma  : sahte-EDM vs σ (2.5/5/10 μm × seeds) → σ²-üssü p doğrulaması
           (Fig: log-log eğim ~2; Omarov Fig. 9a karşılığı)
  gscale : sahte-EDM vs quad gradyanı (0.21/0.40/0.69 T/m, AYNI kaçıklık deseni)
           → yüksek-Q'nun g³ cezası (Fig 7; akilli_duzeltme.md §6.15.1)

Kullanım:
  python3 kmod_drivers/paper_runs.py sigma  -w 4 --seeds 3
  python3 kmod_drivers/paper_runs.py gscale -w 4 --seeds 2

Not: fast_est.fast_measure azaltılmış ayar (n_turns=14, t2=3e-4) kullanır;
sigma modu p≈2'yi aynı koşumda doğruladığından göreli karşılaştırmalar geçerli.
g-ölçeklemede cell-0 QF tuzağına (CLAUDE.md #8) karşı g0 VE g1 birlikte
ölçeklenir (işaret korunur).
"""
import os, sys, json, time, argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(BASE, "berry_data"))
sys.path.insert(0, BASE)
os.makedirs("/tmp/kmod_recover", exist_ok=True)

import ac_bba_linchpin as L
NQ = L.NQ

JSONL = "/tmp/kmod_recover/paper_runs.jsonl"
RESULT = os.path.join(HERE, "paper_runs_results.json")


def measure_fourparticle(dx, dy, delta=100e-6, t2=3e-4):
    """CLAUDE.md reçetesinin PARÇACIK-tabanlı 4-katlısı: ideal eksenden
    (±δ, ±δ) dört simetrik başlangıç; S_y(t) izleri ORTALANIR (betatron
    çiftler hâlinde söner), sonra model-fit seküler eğim. CO araması YOK."""
    from false_edm_mode_scan import setup_fields, _make_state, measure_dSy_dt_model
    from integrator import integrate_particle
    with open(os.path.join(BASE, "params.json")) as f:
        cfg = json.load(f)
    DT = float(cfg["dt"])
    tilt = np.zeros(NQ)
    traces, times = [], None
    for sx in (+1, -1):
        for sy in (+1, -1):
            fields, y0, beta0, R0, p_mag, direction = setup_fields(cfg)
            launch = _make_state([sx * delta, sy * delta, 0.0, 0.0],
                                 p_mag, direction, [0.0, 0.0, direction])
            fields.poincare_quad_index = 0.0
            _, poin, pt = integrate_particle(launch, 0.0, t2, DT, fields=fields,
                                             return_steps=4000, quad_dx=dx,
                                             quad_dy=dy, quad_tilt=tilt)
            traces.append(np.asarray(poin[:, 7], float))
            times = np.asarray(pt, float)
    n = min(len(tr) for tr in traces)
    sy_avg = np.mean([tr[:n] for tr in traces], axis=0)
    return float(measure_dSy_dt_model(sy_avg, times[:n]))


def measure_fourparticle_co(dx, dy, delta=100e-6, n_turns=14, t2=3e-4):
    """4-katlının DOĞRU sürümü: dört simetrik parçacık KAPALI YÖRÜNGE etrafında
    (v_co ± (±δ,±δ)); S_y izleri ortalanır → betatron çiftler hâlinde söner,
    model-fit seküler eğim. 4D-CO+model-fit ile eşdeğerlik testi."""
    from false_edm_mode_scan import setup_fields, _make_state, measure_dSy_dt_model
    from false_edm_4d import find_co_4d, _T_rev
    from integrator import integrate_particle
    with open(os.path.join(BASE, "params.json")) as f:
        cfg = json.load(f)
    DT = float(cfg["dt"])
    tilt = np.zeros(NQ)
    fields, y0, beta0, R0, p_mag, direction = setup_fields(cfg)
    T_rev = _T_rev(fields, beta0, R0)
    v_co, resid = find_co_4d(fields, p_mag, direction, dx, dy, tilt, T_rev,
                             n_turns=n_turns, n_iter=2)
    traces, times = [], None
    for sx in (+1, -1):
        for sy in (+1, -1):
            v = [v_co[0] + sx * delta, v_co[1] + sy * delta, v_co[2], v_co[3]]
            launch = _make_state(v, p_mag, direction, [0.0, 0.0, direction])
            fields.poincare_quad_index = 0.0
            _, poin, pt = integrate_particle(launch, 0.0, t2, DT, fields=fields,
                                             return_steps=4000, quad_dx=dx,
                                             quad_dy=dy, quad_tilt=tilt)
            traces.append(np.asarray(poin[:, 7], float))
            times = np.asarray(pt, float)
    n = min(len(tr) for tr in traces)
    sy_avg = np.mean([tr[:n] for tr in traces], axis=0)
    return float(measure_dSy_dt_model(sy_avg, times[:n])), resid


def fast_measure_g(dx, dy, g_scale=1.0, n_turns=14, t2=3e-4):
    """fast_est.fast_measure eşdeğeri + gradyan ölçekleme (g0 VE g1 birlikte)."""
    from false_edm_mode_scan import setup_fields, _make_state, measure_dSy_dt_model
    from false_edm_4d import find_co_4d, _T_rev
    from integrator import integrate_particle
    with open(os.path.join(BASE, "params.json")) as f:
        cfg = json.load(f)
    if g_scale != 1.0:
        cfg["g1"] = float(cfg.get("g1", 0.21)) * g_scale
        cfg["g0"] = float(cfg.get("g0", 0.2)) * g_scale     # cell-0 QF tuzağı
        if "g2" in cfg:
            cfg["g2"] = float(cfg["g2"]) * g_scale
    DT = float(cfg["dt"])
    fields, y0, beta0, R0, p_mag, direction = setup_fields(cfg)
    T_rev = _T_rev(fields, beta0, R0)
    tilt = np.zeros(NQ)
    v_co, resid = find_co_4d(fields, p_mag, direction, dx, dy, tilt, T_rev,
                             n_turns=n_turns, n_iter=2)
    launch = _make_state(v_co, p_mag, direction, [0.0, 0.0, direction])
    fields.poincare_quad_index = 0.0
    _, poin, pt = integrate_particle(launch, 0.0, t2, DT, fields=fields,
                                     return_steps=4000, quad_dx=dx, quad_dy=dy,
                                     quad_tilt=tilt)
    sv = np.asarray(poin[:, 7], float)
    slope = float(measure_dSy_dt_model(sv, np.asarray(pt, float)))
    return abs(slope), resid


def measure_noCO(dx, dy, t2=3e-4):
    """CO ARAMADAN ölçüm: tek parçacık ideal eksenden (0,0,0,0) fırlatılır,
    model-fit seküler eğim (İŞARETLİ). 4-katlı desen-çevrim kombinasyonu için."""
    from false_edm_mode_scan import setup_fields, _make_state, measure_dSy_dt_model
    from integrator import integrate_particle
    with open(os.path.join(BASE, "params.json")) as f:
        cfg = json.load(f)
    DT = float(cfg["dt"])
    fields, y0, beta0, R0, p_mag, direction = setup_fields(cfg)
    launch = _make_state([0.0, 0.0, 0.0, 0.0], p_mag, direction,
                         [0.0, 0.0, direction])
    fields.poincare_quad_index = 0.0
    tilt = np.zeros(NQ)
    _, poin, pt = integrate_particle(launch, 0.0, t2, DT, fields=fields,
                                     return_steps=4000, quad_dx=dx, quad_dy=dy,
                                     quad_tilt=tilt)
    sv = np.asarray(poin[:, 7], float)
    return float(measure_dSy_dt_model(sv, np.asarray(pt, float)))


def fast_measure_signed(dx, dy, n_turns=14, t2=3e-4):
    """fast_measure_g'nin İŞARETLİ sürümü (4-katlı kombinasyon işaret ister)."""
    from false_edm_mode_scan import setup_fields, _make_state, measure_dSy_dt_model
    from false_edm_4d import find_co_4d, _T_rev
    from integrator import integrate_particle
    with open(os.path.join(BASE, "params.json")) as f:
        cfg = json.load(f)
    DT = float(cfg["dt"])
    fields, y0, beta0, R0, p_mag, direction = setup_fields(cfg)
    T_rev = _T_rev(fields, beta0, R0)
    tilt = np.zeros(NQ)
    v_co, resid = find_co_4d(fields, p_mag, direction, dx, dy, tilt, T_rev,
                             n_turns=n_turns, n_iter=2)
    launch = _make_state(v_co, p_mag, direction, [0.0, 0.0, direction])
    fields.poincare_quad_index = 0.0
    _, poin, pt = integrate_particle(launch, 0.0, t2, DT, fields=fields,
                                     return_steps=4000, quad_dx=dx, quad_dy=dy,
                                     quad_tilt=tilt)
    sv = np.asarray(poin[:, 7], float)
    return float(measure_dSy_dt_model(sv, np.asarray(pt, float))), resid


def sym_anti_patterns(seed, rms=10e-6):
    """Rastgele 48-vektörün hücre-içi simetrik/antisimetrik projeksiyonları,
    RMS'e normalize (dikey desenler; CR-ayrım körlük testi, omarov.md §9.3)."""
    rng = np.random.default_rng(9000 + seed)
    v = rng.normal(0, 1.0, NQ)
    sym = np.empty(NQ); anti = np.empty(NQ)
    for k in range(NQ // 2):
        s = 0.5 * (v[2*k] + v[2*k+1]); a = 0.5 * (v[2*k] - v[2*k+1])
        sym[2*k] = sym[2*k+1] = s
        anti[2*k] = a; anti[2*k+1] = -a
    sym *= rms / np.sqrt(np.mean(sym**2))
    anti *= rms / np.sqrt(np.mean(anti**2))
    return sym, anti


def measure_cod(dy, direction):
    """48-quad dikey COD (C++, zaman-ortalamalı; ideal eksenden fırlatma)."""
    import build_response_matrix as brm
    with open(os.path.join(BASE, "params.json")) as f:
        cfg = json.load(f)
    cfg["direction"] = float(direction)
    alanlar, state0 = brm.setup_fields(cfg)
    x, y = brm.run_sim(alanlar, state0, cfg, dy, np.zeros(NQ))
    return y  # [m]


def _worker(task):
    import tempfile
    os.chdir(tempfile.mkdtemp())
    mode, payload = task
    if mode == "crsep":
        # (desen, seed, direction): dikey COD ölç
        kind, seed, direction = payload
        sym, anti = sym_anti_patterns(seed)
        dy = sym if kind == "sym" else anti
        y = measure_cod(dy, direction)
        rec = {"mode": mode, "payload": list(payload),
               "y_cod": [float(v) for v in y], "t": time.time()}
        with open(JSONL, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        return rec
    if mode == "sigma":
        sg, seed = payload
        rng = np.random.default_rng(3000 + seed)          # fast_est 'white' ile aynı
        dx = rng.normal(0, sg, NQ); dy = rng.normal(0, sg, NQ)
        f, resid = fast_measure_g(dx, dy, g_scale=1.0)
    elif mode == "fourfold":
        # (sx, sy, method): dx→sx·dx, dy→sy·dy desen-çevrimi; seed 0, σ=10 μm
        sx, sy, method = payload
        rng = np.random.default_rng(3000 + 0)
        dx = sx * rng.normal(0, 10e-6, NQ); dy = sy * rng.normal(0, 10e-6, NQ)
        if method == "co":
            f, resid = fast_measure_signed(dx, dy)
        else:
            f, resid = measure_noCO(dx, dy), 0.0
    elif mode == "fourpart":
        # parçacık-tabanlı 4-katlı: (delta_um,) — seed 0 deseni, CO araması yok
        (delta_um,) = payload
        rng = np.random.default_rng(3000 + 0)
        dx = rng.normal(0, 10e-6, NQ); dy = rng.normal(0, 10e-6, NQ)
        f, resid = measure_fourparticle(dx, dy, delta=delta_um * 1e-6), 0.0
    elif mode == "fourpart_co":
        # 4-katlı, CO ETRAFINDA: (delta_um,) — eşdeğerlik testi
        (delta_um,) = payload
        rng = np.random.default_rng(3000 + 0)
        dx = rng.normal(0, 10e-6, NQ); dy = rng.normal(0, 10e-6, NQ)
        f, resid = measure_fourparticle_co(dx, dy, delta=delta_um * 1e-6)
    else:  # gscale: AYNI 10 μm desen (seed sabitler), g ölçeklenir
        gsc, seed, co_turns = payload
        rng = np.random.default_rng(3000 + seed)
        dx = rng.normal(0, 10e-6, NQ); dy = rng.normal(0, 10e-6, NQ)
        f, resid = fast_measure_g(dx, dy, g_scale=gsc, n_turns=co_turns)
    rec = {"mode": mode, "payload": list(payload), "f": f, "resid": float(resid),
           "t": time.time()}
    with open(JSONL, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def save_result(key, obj):
    data = {}
    if os.path.exists(RESULT):
        with open(RESULT) as fh:
            data = json.load(fh)
    data[key] = obj
    with open(RESULT, "w") as fh:
        json.dump(data, fh, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sigma", "gscale", "fourfold", "fourpart",
                                     "fourpart_co", "crsep"])
    ap.add_argument("-w", "--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--co-turns", type=int, default=14)
    ap.add_argument("--gvals", type=str, default="0.21,0.40,0.69")
    args = ap.parse_args()
    t0 = time.time()

    import build_response_matrix as brm

    if args.mode == "sigma":
        sigmas = [2.5e-6, 5e-6, 10e-6]
        tasks = [("sigma", (sg, sd)) for sg in sigmas for sd in range(args.seeds)]
    elif args.mode == "fourfold":
        combos = [(+1, +1), (+1, -1), (-1, +1), (-1, -1)]
        tasks = [("fourfold", (sx, sy, m)) for m in ("co", "noco")
                 for (sx, sy) in combos]
    elif args.mode == "fourpart":
        tasks = [("fourpart", (d,)) for d in (50.0, 100.0, 200.0)]
    elif args.mode == "fourpart_co":
        tasks = [("fourpart_co", (d,)) for d in (50.0, 100.0, 200.0)]
    elif args.mode == "crsep":
        tasks = [("crsep", (kind, sd, dirn)) for kind in ("sym", "anti")
                 for sd in range(args.seeds) for dirn in (+1, -1)]
    else:
        gscales = [float(g) / 0.21 for g in args.gvals.split(",")]
        tasks = [("gscale", (gs, sd + args.seed_offset, args.co_turns))
                 for gs in gscales for sd in range(args.seeds)]

    with ProcessPoolExecutor(args.workers, initializer=brm._worker_init) as pool:
        res = list(pool.map(_worker, tasks))

    d = {}
    for r in res:
        if "f" in r:
            d.setdefault(r["payload"][0], []).append(r["f"])

    if args.mode == "crsep":
        Y = {}
        for r in res:
            kind, sd, dirn = r["payload"]
            Y[(kind, int(sd), int(dirn))] = np.array(r["y_cod"])
        print("=== CR-AYRIM KÖRLÜĞÜ (COD_CW − COD_CCW; sym vs anti, 10 μm) ===")
        rms = lambda a: float(np.sqrt(np.mean(a**2)))
        out = {"seeds": args.seeds}
        stats = {}
        for kind in ("sym", "anti"):
            cod1 = [rms(Y[(kind, sd, +1)]) for sd in range(args.seeds)]
            cr = [rms(Y[(kind, sd, +1)] - Y[(kind, sd, -1)])
                  for sd in range(args.seeds)]
            stats[kind] = (np.mean(cod1), np.mean(cr))
            print(f"  {kind:4s}: tek-yön COD RMS = {np.mean(cod1)*1e6:7.2f} μm   "
                  f"CR-ayrım RMS = {np.mean(cr)*1e6:7.2f} μm")
            out[kind] = {"cod_rms": cod1, "cr_rms": cr}
        s_cod = stats["anti"][0] / stats["sym"][0]
        s_cr = stats["anti"][1] / stats["sym"][1]
        print(f"  BASTIRMA (anti/sym): tek-yön COD {s_cod:.1f}×   CR-ayrım {s_cr:.1f}×")
        print("  → oran ≈ 1 ise CR-ayrım simetriğe sıradan yörünge kadar kör")
        out["suppression_cod"] = s_cod
        out["suppression_cr"] = s_cr
        save_result("crsep", out)

    if args.mode in ("fourpart", "fourpart_co"):
        yer = "CO ETRAFINDA" if args.mode == "fourpart_co" else "ideal eksenden (CO YOK)"
        print(f"=== PARÇACIK-TABANLI 4-KATLI ({yer}; seed 0, 10 μm) ===")
        print("  Referans (4D-CO+model-fit, aynı desen): f = +1.089e-06 rad/s")
        out = {}
        for dlt in sorted(d):
            f = d[dlt][0]
            print(f"  δ = {dlt:5.0f} μm: 4-parçacık ortalama eğim = {f:+.3e} "
                  f"(CO'ya oran {f/1.089e-06:+.2f})")
            out[str(dlt)] = f
        save_result(args.mode, out)

    if args.mode == "fourfold":
        F = {}
        for r in res:
            sx, sy, m = r["payload"]
            F[(int(sx), int(sy), m)] = r["f"]
        print("=== 4-KATLI DESEN-ÇEVRİM ↔ 4D-CO ÇAPRAZ-DOĞRULAMA (seed 0, 10 μm) ===")
        out = {}
        for m in ("co", "noco"):
            fpp, fpm = F[(1, 1, m)], F[(1, -1, m)]
            fmp, fmm = F[(-1, 1, m)], F[(-1, -1, m)]
            bil = (fpp - fpm - fmp + fmm) / 4.0            # bilineer (dx·dy) izolasyon
            print(f"  [{m:4s}] f(++)={fpp:+.3e} f(+-)={fpm:+.3e} "
                  f"f(-+)={fmp:+.3e} f(--)={fmm:+.3e}")
            print(f"         4-katlı bilineer = {bil:+.3e}   (f(++)'a oran "
                  f"{bil/fpp if fpp else 0:+.2f})")
            out[m] = {"fpp": fpp, "fpm": fpm, "fmp": fmp, "fmm": fmm, "bilinear": bil}
        r_co, r_no = out["co"]["bilinear"], out["noco"]["bilinear"]
        print(f"  ÇAPRAZ: CO-yöntemi bilineer {r_co:+.3e}  vs  CO'suz (ideal-eksen) "
              f"bilineer {r_no:+.3e}   oran {r_no/r_co if r_co else 0:+.3f}")
        save_result("fourfold", out)

    if args.mode == "sigma":
        print("=== SİGMA TARAMASI (sahte-EDM ∝ σ^p) ===")
        rows = []
        for sg in sorted(d):
            a = np.array(d[sg])
            print(f"  σ={sg*1e6:4.1f}μm: |f| = {a.mean():.3e} ± {a.std():.1e} (n={len(a)})")
            rows.append({"sigma_um": sg * 1e6, "f_mean": a.mean(), "f_std": a.std(),
                         "f_all": list(a)})
        lgs = np.log([r["sigma_um"] for r in rows])
        lgf = np.log([r["f_mean"] for r in rows])
        p = np.polyfit(lgs, lgf, 1)[0]
        print(f"  üs p (log-log fit, 3 nokta) = {p:.3f}   (beklenen 2.00)")
        save_result("sigma", {"rows": rows, "p": p})
    elif args.mode == "gscale":
        print("=== GRADYAN TARAMASI (yüksek-Q g³ cezası) ===")
        rows = []
        f0 = None
        for gs in sorted(d):
            a = np.array(d[gs])
            g_tm = 0.21 * gs
            if f0 is None:
                f0 = a.mean()
            print(f"  g={g_tm:.2f} T/m (×{gs:.2f}): |f| = {a.mean():.3e} ± "
                  f"{a.std():.1e}  → g=0.21'e göre {a.mean()/f0:.1f}×")
            rows.append({"g_scale": gs, "g_Tm": g_tm, "f_mean": a.mean(),
                         "f_std": a.std(), "f_all": list(a)})
        save_result(f"gscale_co{args.co_turns}", {"rows": rows})

    print(f"[toplam {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
