#!/usr/bin/env python3
"""donmus_t2.py — T2 (rijit BPM, BPM–kuadrupol göreli kayması σ_δ) sahte EDM sonuçlarının donmuş spin
aracıyla, daha fazla seed'le yeniden ölçümü.

Senaryo t2_bpm_rigid.py 'edm' ile aynı: 5 μm saf simetrik sürüklenme, 14 nm gürültü, σ_δ ∈ {0.1, 0.3,
1.0} μm, kestiriciler E4 (iki demet, ρ-bilen) ve E2 (tek demet, ρ-bilen); rcond t2_recon.json'daki en
iyi değer. Her durum iki yönde donmuş spin noktasında ölçülür; C = ½|f_CW − f_CCW|.
"""
import os, sys, json, argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor

REV = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REV)
from sahte_edm_donmus import measure_frozen

METHODS = ("E4", "E2")


def task(t):
    seed, sd, k, d, n_turns, n_iter = t
    import t2_bpm_rigid as T2
    import adim1_bpm_model as A1
    drift, sig_n = 5e-6, 14e-9
    rc = T2._best_rc(drift, sig_n, sd, k)
    dx = T2.residual(seed, "x", drift, sd, sig_n, k, rc)
    dy = T2.residual(seed, "y", drift, sd, sig_n, k, rc)
    r = measure_frozen(dx, dy, direction=d, n_turns=n_turns, n_iter=n_iter)
    return dict(seed=seed, sig_d=sd, method=k, direction=d, rcond=rc, f=r["f"], f_target=r["f"] / 1e-9,
                w1=r["w1"], sym_um=float(np.mean([np.std(A1.PS @ dx), np.std(A1.PS @ dy)]) * 1e6))


def main(seeds, workers, ayar):
    import t2_bpm_rigid as T2
    n_turns, n_iter = (14, 2) if ayar == "kisa" else (28, 4)
    out = os.path.join(REV, "sonuclar", f"donmus_t2_{ayar}.jsonl")
    done = set()
    if os.path.exists(out):
        done = {(r["seed"], r["sig_d"], r["method"], r["direction"]) for r in map(json.loads, open(out))}
    tasks = [(s, sd, k, d, n_turns, n_iter) for s in seeds for sd in T2.EDM_SD for k in METHODS
             for d in (-1, +1) if (s, sd, k, d) not in done]
    print(f"{len(tasks)} görev (her biri 2 izleme)", flush=True)
    with ProcessPoolExecutor(workers) as pool:
        for r in pool.map(task, tasks):
            open(out, "a").write(json.dumps(r) + "\n")
            print(f"seed {r['seed']} σ_δ={r['sig_d']*1e6:.1f} μm {r['method']} yön {r['direction']:+d}: "
                  f"f* = {r['f_target']:10.4f} × hedef (sim {r['sym_um']:.3f} μm)", flush=True)
    summarize(out)


def summarize(out):
    import t2_bpm_rigid as T2
    rows = [json.loads(l) for l in open(out)]
    print(f"\n{os.path.basename(out)}: C = ½|f_CW − f_CCW| [× hedef]")
    for sd in T2.EDM_SD:
        for k in METHODS:
            Cs, sy = [], []
            for s in sorted(set(r["seed"] for r in rows)):
                rr = [r for r in rows if r["seed"] == s and r["sig_d"] == sd and r["method"] == k]
                f = {r["direction"]: r["f"] for r in rr}
                if len(f) == 2:
                    Cs.append(abs(f[1] - f[-1]) / 2 / 1e-9); sy.append(rr[0]["sym_um"])
            if Cs:
                Cs = np.array(Cs)
                print(f"  σ_δ={sd*1e6:.1f} μm {k}: n={len(Cs):2d}  medyan {np.median(Cs):.4g}  "
                      f"[16–84%: {np.percentile(Cs,16):.3g}, {np.percentile(Cs,84):.3g}]  max {Cs.max():.3g}  "
                      f"hedef-üstü {int((Cs > 1).sum())}/{len(Cs)}  (sim artık medyan {np.median(sy):.3f} μm)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("-w", "--workers", type=int, default=4)
    ap.add_argument("--ayar", choices=["kisa", "tam"], default="kisa")
    ap.add_argument("--ozet", action="store_true")
    a = ap.parse_args()
    if a.ozet:
        summarize(os.path.join(REV, "sonuclar", f"donmus_t2_{a.ayar}.jsonl"))
    else:
        main(a.seeds, a.workers, a.ayar)
