#!/usr/bin/env python3
"""donmus_adim6.py — Adım 6 (zaman senaryosu) C++ noktalarının donmuş spin aracıyla, daha fazla seed'le
yeniden ölçümü.

Zaman serisi adim6_time.simulate ile aynıdır (ATL zemin hareketi A = 1e-5 μm²/(s·m), BBA artığı 0.3 μm,
BPM–kuadrupol göreli kayması 0.2 μm/√gün, BBA eşiği σ_δ* = 0.3 μm, 10 nm gürültü tabanı). Seçilen
saatlerdeki etkin kaçıklık m_eff = q − c iki yönde donmuş spin noktasında ölçülür; C = ½|f_CW − f_CCW|.
t = 0 bütün senaryolarda aynı (BBA artığı) olduğundan yalnız bir kez ölçülür.
"""
import os, sys, json, argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor

REV = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REV)
from sahte_edm_donmus import measure_frozen

HOURS = [0, 24, 96, 168]


def task(t):
    seed, w, m, h, d, n_turns, n_iter = t
    import adim6_time as A6
    o = A6.simulate(seed, w, m, record_hours=[h])
    dx, dy = (np.array(a) for a in o["m_eff"][h])
    r = measure_frozen(dx, dy, direction=d, n_turns=n_turns, n_iter=n_iter)
    return dict(seed=seed, world=w, method=m, hour=h, direction=d, f=r["f"], f_target=r["f"] / 1e-9,
                w1=r["w1"], sym=float(np.mean([np.std(A6.A4.PS @ dx), np.std(A6.A4.PS @ dy)]) * 1e6))


def main(seeds, workers, ayar):
    import adim6_time as A6
    n_turns, n_iter = (14, 2) if ayar == "kisa" else (28, 4)
    out = os.path.join(REV, "sonuclar", f"donmus_adim6_{ayar}.jsonl")
    done = set()
    if os.path.exists(out):
        done = {(r["seed"], r["world"], r["method"], r["hour"], r["direction"]) for r in map(json.loads, open(out))}
    tasks = [(s, w, m, h, d, n_turns, n_iter) for s in seeds for w, m in A6.SCEN for h in HOURS
             for d in (-1, +1) if not (h == 0 and (w, m) != ("W1", "none")) and (s, w, m, h, d) not in done]
    print(f"{len(tasks)} görev (her biri 2 izleme)", flush=True)
    with ProcessPoolExecutor(workers) as pool:
        for r in pool.map(task, tasks):
            open(out, "a").write(json.dumps(r) + "\n")
            print(f"seed {r['seed']} {r['world']}-{r['method']:8s} {r['hour']:3d} sa yön {r['direction']:+d}: "
                  f"f* = {r['f_target']:10.4f} × hedef (sim {r['sym']:.3f} μm)", flush=True)
    summarize(out)


def summarize(out):
    import adim6_time as A6
    rows = [json.loads(l) for l in open(out)]
    print(f"\n{os.path.basename(out)}: C = ½|f_CW − f_CCW| [× hedef]")
    for w, m in [("W1", "none")] + [s for s in A6.SCEN if s != ("W1", "none")]:
        for h in HOURS:
            ww, mm = ("W1", "none") if h == 0 else (w, m)
            if h == 0 and (w, m) != ("W1", "none"):
                continue
            Cs, sy = [], []
            for s in sorted(set(r["seed"] for r in rows)):
                rr = [r for r in rows if r["seed"] == s and r["world"] == ww and r["method"] == mm and r["hour"] == h]
                f = {r["direction"]: r["f"] for r in rr}
                if len(f) == 2:
                    Cs.append(abs(f[1] - f[-1]) / 2 / 1e-9); sy.append(rr[0]["sym"])
            if Cs:
                Cs = np.array(Cs)
                lab = "t=0 (BBA artığı)" if h == 0 else f"{w}-{m} {h} sa"
                print(f"  {lab:22s}: n={len(Cs):2d}  medyan {np.median(Cs):.4g}  [16–84%: {np.percentile(Cs,16):.3g}, "
                      f"{np.percentile(Cs,84):.3g}]  max {Cs.max():.3g}  hedef-üstü {int((Cs > 1).sum())}/{len(Cs)}  "
                      f"(sim medyan {np.median(sy):.3f} μm)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    ap.add_argument("-w", "--workers", type=int, default=4)
    ap.add_argument("--ayar", choices=["kisa", "tam"], default="kisa")
    ap.add_argument("--ozet", action="store_true")
    a = ap.parse_args()
    if a.ozet:
        summarize(os.path.join(REV, "sonuclar", f"donmus_adim6_{a.ayar}.jsonl"))
    else:
        main(a.seeds, a.workers, a.ayar)
