#!/usr/bin/env python3
"""donmus_t1.py — T1 (saptırıcı sürüklenmesini kaynağında trimlemek) sahte EDM sonuçlarının donmuş spin
aracıyla, daha fazla seed'le yeniden ölçümü.

Senaryo t1_defl_trim.py ile aynı (seed 30000+, 5 μm simetrik kuadrupol sürüklenmesi, ε, φ ~ N(0, 3e-7),
14 nm gürültü; durumlar trim / oracle / trim_g20 / QD_g20). Saptırıcıda kalan hata (ε, φ)
integrator_rev.DEFL_DE / DEFL_ROLL ile uygulanır; _g20 durumlarında gerçek makinede %0.2 gradyan hatası
(dG) hem kapalı yörünge aramasında hem izlemede vardır. Donmuş spin noktası her durum ve yön için
ayrı aranır (saptırıcı hatası da enerji/presesyonu kaydırır). C = ½|f_CW − f_CCW|.
"""
import os, sys, json, argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor

REV = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REV)
from sahte_edm_donmus import measure_frozen
import integrator_rev

CASES = ["trim", "oracle", "trim_g20", "QD_g20"]


def task(t):
    seed, case, d, n_turns, n_iter = t
    import t1_defl_trim as T1
    import adim1_bpm_model as A1
    o = T1.make(seed, case)
    dx, dy = o["x"][0], o["y"][0]
    integrator_rev.DEFL_DE = np.asarray(o["x"][1], float)
    integrator_rev.DEFL_ROLL = np.asarray(o["y"][1], float)
    dG = None
    if case.endswith("_g20"):
        from build_rtrue import dG_pattern
        dG = dG_pattern(0.002)
    r = measure_frozen(dx, dy, direction=d, dG=dG, n_turns=n_turns, n_iter=n_iter)
    rel = lambda k: float(np.std(o[k][1]) / np.std(o[k][2]))
    return dict(seed=seed, case=case, direction=d, f=r["f"], f_target=r["f"] / 1e-9, w1=r["w1"],
                delta_star=r["delta_star"],
                sym_um=float(np.mean([np.std(A1.PS @ dx), np.std(A1.PS @ dy)]) * 1e6),
                defl_left_rel_eps=rel("x"), defl_left_rel_roll=rel("y"))


def main(seeds, workers, ayar):
    n_turns, n_iter = (14, 2) if ayar == "kisa" else (28, 4)
    out = os.path.join(REV, "sonuclar", f"donmus_t1_{ayar}.jsonl")
    done = set()
    if os.path.exists(out):
        done = {(r["seed"], r["case"], r["direction"]) for r in map(json.loads, open(out))}
    tasks = [(s, c, d, n_turns, n_iter) for s in seeds for c in CASES for d in (-1, +1)
             if (s, c, d) not in done]
    print(f"{len(tasks)} görev (her biri 2 izleme)", flush=True)
    with ProcessPoolExecutor(workers) as pool:
        for r in pool.map(task, tasks):
            open(out, "a").write(json.dumps(r) + "\n")
            print(f"seed {r['seed']} {r['case']:8s} yön {r['direction']:+d}: f* = {r['f_target']:10.4f} × hedef "
                  f"(sim {r['sym_um']:.3f} μm; kalan ε {r['defl_left_rel_eps']:.3f}, φ {r['defl_left_rel_roll']:.3f})",
                  flush=True)
    summarize(out)


def summarize(out):
    rows = [json.loads(l) for l in open(out)]
    print(f"\n{os.path.basename(out)}: C = ½|f_CW − f_CCW|, çift = ½(f_CW + f_CCW) [× hedef]")
    for c in CASES:
        Cs, ev = [], []
        for s in sorted(set(r["seed"] for r in rows)):
            f = {r["direction"]: r["f"] for r in rows if r["seed"] == s and r["case"] == c}
            if len(f) == 2:
                Cs.append(abs(f[1] - f[-1]) / 2 / 1e-9); ev.append((f[1] + f[-1]) / 2 / 1e-9)
        if Cs:
            Cs = np.array(Cs); ev = np.array(ev)
            print(f"  {c:8s}: n={len(Cs):2d}  C medyan {np.median(Cs):.4g}  [16–84%: {np.percentile(Cs,16):.3g}, "
                  f"{np.percentile(Cs,84):.3g}]  max {Cs.max():.3g}  hedef-üstü {int((Cs > 1).sum())}/{len(Cs)}  "
                  f"| çift medyan|.| {np.median(np.abs(ev)):.3g}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("-w", "--workers", type=int, default=4)
    ap.add_argument("--ayar", choices=["kisa", "tam"], default="kisa")
    ap.add_argument("--ozet", action="store_true")
    a = ap.parse_args()
    if a.ozet:
        summarize(os.path.join(REV, "sonuclar", f"donmus_t1_{a.ayar}.jsonl"))
    else:
        main(a.seeds, a.workers, a.ayar)
