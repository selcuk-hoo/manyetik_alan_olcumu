#!/usr/bin/env python3
"""donmus_adim1.py — Adım 1 (BPM modeli) sahte EDM sonuçlarının donmuş spin aracıyla, daha fazla seed'le
yeniden ölçümü.

Senaryo adim1_bpm_model.py ile aynı: 5 μm rms saf simetrik sürüklenme, 14 nm gürültü, 'k0' ailesi
(u = ρΔm), her kestiricinin en iyi rcond'u (adim1_recon.json). Durumlar:
  E0 (düzeltme yok) ; ρ=0: E1 (tek demet), E3 (iki demet) ; ρ=1: E1, E2, E3, E4
Her durum iki yönde ölçülür; C = ½|f_CW − f_CCW|. Ölçüm ayarı --ayar ile (kisa: 14/2, tam: 28/4).
"""
import os, sys, json, argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor

REV = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REV)
from sahte_edm_donmus import measure_frozen

CASES = [(0.0, "E0"), (0.0, "E1"), (0.0, "E3"), (1.0, "E1"), (1.0, "E2"), (1.0, "E3"), (1.0, "E4")]


def task(t):
    seed, rho, method, d, n_turns, n_iter = t
    import adim1_bpm_model as A1
    rc = A1.best_rcond("k0", "sym", 14e-9, rho, method) if method != "E0" else A1.RCOND
    res = A1.make_case(seed, "sym", 5e-6, rho, 14e-9, "k0", rc)
    dx, dy = res["x"][method], res["y"][method]
    r = measure_frozen(dx, dy, direction=d, n_turns=n_turns, n_iter=n_iter)
    return dict(seed=seed, rho=rho, method=method, direction=d, rcond=rc, f=r["f"],
                f_target=r["f"] / 1e-9, w1=r["w1"],
                sym_um=float(np.mean([np.std(A1.PS @ dx), np.std(A1.PS @ dy)]) * 1e6))


def main(seeds, workers, ayar):
    n_turns, n_iter = (14, 2) if ayar == "kisa" else (28, 4)
    out = os.path.join(REV, "sonuclar", f"donmus_adim1_{ayar}.jsonl")
    done = set()
    if os.path.exists(out):
        done = {(r["seed"], r["rho"], r["method"], r["direction"]) for r in map(json.loads, open(out))}
    tasks = [(s, rho, m, d, n_turns, n_iter) for s in seeds for rho, m in CASES for d in (-1, +1)
             if (s, rho, m, d) not in done]
    print(f"{len(tasks)} görev (her biri 2 izleme)", flush=True)
    with ProcessPoolExecutor(workers) as pool:
        for r in pool.map(task, tasks):
            open(out, "a").write(json.dumps(r) + "\n")
            print(f"seed {r['seed']} ρ={r['rho']} {r['method']} yön {r['direction']:+d}: f* = {r['f_target']:10.4f} × hedef "
                  f"(sim {r['sym_um']:.3f} μm)", flush=True)
    summarize(out)


def summarize(out):
    rows = [json.loads(l) for l in open(out)]
    print(f"\n{os.path.basename(out)}: C = ½|f_CW − f_CCW| [× hedef]")
    for rho, m in CASES:
        Cs = []
        for s in sorted(set(r["seed"] for r in rows)):
            f = {r["direction"]: r["f"] for r in rows if r["seed"] == s and r["rho"] == rho and r["method"] == m}
            if len(f) == 2:
                Cs.append(abs(f[1] - f[-1]) / 2 / 1e-9)
        if Cs:
            Cs = np.array(Cs)
            print(f"  ρ={rho} {m}: n={len(Cs):2d}  medyan {np.median(Cs):.4g}  [16–84%: {np.percentile(Cs,16):.3g}, "
                  f"{np.percentile(Cs,84):.3g}]  max {Cs.max():.3g}  hedef-üstü {int((Cs > 1).sum())}/{len(Cs)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("-w", "--workers", type=int, default=4)
    ap.add_argument("--ayar", choices=["kisa", "tam"], default="kisa")
    ap.add_argument("--ozet", action="store_true")
    a = ap.parse_args()
    if a.ozet:
        summarize(os.path.join(REV, "sonuclar", f"donmus_adim1_{a.ayar}.jsonl"))
    else:
        main(a.seeds, a.workers, a.ayar)
