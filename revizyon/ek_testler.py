#!/usr/bin/env python3
"""ek_testler.py — ek rapor (soru listesi) için kısa testler; donmuş spin prosedürü (kısaltılmış ayar).

Modlar:
  rho_lin : (madde 15) gerçek ρ = 1 iken kestirici ρ̂ ≠ 1 varsayarsa ne olur? 200 seed, doğrusal:
            kalan simetrik rms ve m = 2–3 güç payı; E2 ve E4, en iyi rcond (ρ = 1 için seçilen).
  edm     : C++ sahte EDM, iki yön, C = ½|f_CW − f_CCW|. Görev türleri:
            rhohat : ρ = 1, ρ̂ ∈ {0.9, 0.95}, E4 ve E2, seed 0–5          (madde 15)
            rho05  : ρ = 0.5 (ρ biliniyor), E1–E4, seed 0–5               (madde 16)
            Ed     : Adım 3 elektrostatik düzeltici, Ed_zero / Ed_center, seed 0–4  (madde 22)
  ozet    : özet tablo.
Senaryo Adım 1 ile aynı: 5 μm rms saf simetrik sürüklenme, 14 nm gürültü, 'k0' ailesi.
"""
import os, sys, json, argparse
import numpy as np

REV = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REV)
OUT = os.path.join(REV, "sonuclar", "ek_testler_edm.jsonl")
OUT_LIN = os.path.join(REV, "sonuclar", "ek_testler_rho_lin.json")
NQ = 48
RHOHAT = [0.8, 0.9, 0.95, 0.98, 1.0]


def residual_rhohat(seed, rhohat, method, sig_n=14e-9, rms=5e-6):
    """Gerçek ρ = 1 ('k0': u = Δm); kestirici (R − ρ̂ I) kullanır. adim1 make_case ile aynı çekimler."""
    import adim1_bpm_model as A1
    rc = A1.best_rcond("k0", "sym", sig_n, 1.0, method)
    rng = np.random.default_rng(10_000 + seed)
    out = {}
    for pl in ("x", "y"):
        dm = A1.drift_pattern(rng, "sym", rms)
        A1.drift_pattern(rng, "gen", rms)                      # ξ çekimiyle hizalı kal
        n1 = rng.normal(0.0, sig_n, NQ); n2 = rng.normal(0.0, sig_n, NQ)
        Rc, Rw = A1.R[pl]["ccw"], A1.R[pl]["cw"]
        r1 = Rc @ dm - dm + n1; r2 = Rw @ dm - dm + n2
        I = np.eye(NQ)
        if method == "E2":
            P = np.linalg.pinv(Rc - rhohat * I, rcond=rc); y = r1
        else:
            P = np.linalg.pinv(np.vstack([Rc - rhohat * I, Rw - rhohat * I]), rcond=rc)
            y = np.concatenate([r1, r2])
        out[pl] = dm - P @ y
    return out["x"], out["y"], rc


def harm_frac(v):
    import adim1_bpm_model as A1
    F = np.abs(np.fft.rfft((A1.PS @ v)[::2])) ** 2
    return F / F.sum()


def run_lin(nseed=200):
    import adim1_bpm_model as A1
    rows = []
    for m in ("E4", "E2"):
        for rh in RHOHAT:
            sym, f23 = [], []
            for s in range(nseed):
                dx, dy, rc = residual_rhohat(s, rh, m)
                sym.append(np.mean([np.std(A1.PS @ dx), np.std(A1.PS @ dy)]) * 1e6)
                f23.append(np.mean([harm_frac(dx)[2:4].sum(), harm_frac(dy)[2:4].sum()]))
            rows.append(dict(method=m, rhohat=rh, rcond=rc, sym_med_um=float(np.median(sym)),
                             sym_p16=float(np.percentile(sym, 16)), sym_p84=float(np.percentile(sym, 84)),
                             m23_frac_med=float(np.median(f23))))
            print(f"{m} ρ̂={rh:.2f}: kalan sim medyan {rows[-1]['sym_med_um']:.3f} μm "
                  f"[{rows[-1]['sym_p16']:.3f}, {rows[-1]['sym_p84']:.3f}], m=2–3 payı {rows[-1]['m23_frac_med']:.3f}",
                  flush=True)
    json.dump(dict(aciklama="gerçek ρ=1, kestirici ρ̂; 5 μm sim, 14 nm, 200 seed", rows=rows),
              open(OUT_LIN, "w"), indent=1)


def tasks_all():
    T = []
    for s in range(6):
        for rh in (0.9, 0.95):
            for m in ("E4", "E2"):
                T.append(("rhohat", s, f"{m}_{rh}"))
        for m in ("E1", "E2", "E3", "E4"):
            T.append(("rho05", s, m))
    for s in range(5):
        for c in ("Ed_zero", "Ed_center"):
            T.append(("Ed", s, c))
    return [(k, s, c, d) for (k, s, c) in T for d in (-1, +1)]


def task(t):
    kind, seed, case, d = t
    from sahte_edm_donmus import measure_frozen
    import integrator_rev
    import adim1_bpm_model as A1
    z = np.zeros(NQ)
    integrator_rev.DEFL_DE = z; integrator_rev.DEFL_ROLL = z
    integrator_rev.DEFL_BV = z; integrator_rev.DEFL_BR = z
    extra = {}
    if kind == "rhohat":
        m, rh = case.split("_")
        dx, dy, rc = residual_rhohat(seed, float(rh), m)
        extra["rcond"] = rc
    elif kind == "rho05":
        rc = A1.best_rcond("k0", "sym", 14e-9, 0.5, case)
        res = A1.make_case(seed, "sym", 5e-6, 0.5, 14e-9, "k0", rc)
        dx, dy = res["x"][case], res["y"][case]
        extra["rcond"] = rc
    else:
        import adim3_correctors as A3
        dx, dy, corr, pred = A3.make_case(seed, case)
        integrator_rev.DEFL_DE = np.asarray(corr["eps"], float)
        integrator_rev.DEFL_ROLL = np.asarray(corr["roll"], float)
        integrator_rev.DEFL_BV = np.asarray(corr["Bv"], float)
        integrator_rev.DEFL_BR = np.asarray(corr["Br"], float)
        extra["corr_eps_rms"] = float(np.std(corr["eps"])); extra["corr_roll_rms"] = float(np.std(corr["roll"]))
        extra["beam_minus_center_um"] = pred["y"]["beam_minus_center_rms_um"]
    r = measure_frozen(dx, dy, direction=d, n_turns=14, n_iter=2)
    return dict(kind=kind, seed=seed, case=case, direction=d, f=r["f"], f_target=r["f"] / 1e-9,
                w1=r["w1"], delta_star=r["delta_star"],
                sym_um=float(np.mean([np.std(A1.PS @ dx), np.std(A1.PS @ dy)]) * 1e6), **extra)


def run_edm(workers):
    from concurrent.futures import ProcessPoolExecutor
    done = set()
    if os.path.exists(OUT):
        done = {(r["kind"], r["seed"], r["case"], r["direction"]) for r in map(json.loads, open(OUT))}
    T = [t for t in tasks_all() if t not in done]
    print(f"{len(T)} görev (her biri 2 izleme)", flush=True)
    with ProcessPoolExecutor(workers) as pool:
        for r in pool.map(task, T):
            open(OUT, "a").write(json.dumps(r) + "\n")
            print(f"{r['kind']:6s} seed {r['seed']} {r['case']:9s} yön {r['direction']:+d}: f* = {r['f_target']:12.4f} × hedef "
                  f"(sim {r['sym_um']:.3f} μm)", flush=True)
    summarize()


def summarize():
    rows = [json.loads(l) for l in open(OUT)]
    print("C = ½|f_CW − f_CCW|, çift = ½(f_CW + f_CCW) [× hedef]")
    for kind, case in sorted({(r["kind"], r["case"]) for r in rows}):
        Cs, ev, sy = [], [], []
        for s in sorted({r["seed"] for r in rows}):
            rr = [r for r in rows if r["kind"] == kind and r["case"] == case and r["seed"] == s]
            f = {r["direction"]: r["f"] for r in rr}
            if len(f) == 2:
                Cs.append(abs(f[1] - f[-1]) / 2 / 1e-9); ev.append((f[1] + f[-1]) / 2 / 1e-9); sy.append(rr[0]["sym_um"])
        if Cs:
            Cs = np.array(Cs)
            print(f"  {kind:6s} {case:9s}: n={len(Cs)}  C medyan {np.median(Cs):.4g}  [{Cs.min():.3g}, {Cs.max():.3g}]  "
                  f"hedef-üstü {int((Cs > 1).sum())}/{len(Cs)}  |çift| medyan {np.median(np.abs(ev)):.3g}  "
                  f"(sim medyan {np.median(sy):.3f} μm)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["rho_lin", "edm", "ozet"])
    ap.add_argument("-w", "--workers", type=int, default=4)
    a = ap.parse_args()
    {"rho_lin": run_lin, "edm": lambda: run_edm(a.workers), "ozet": summarize}[a.mode]()
