#!/usr/bin/env python3
"""donmus_t1_genlik.py — T1 (saptırıcı trimi) genlik taraması: saptırıcı başına bağımsız yavaş sürüklenme
ne kadar olabilir? (ek rapor, saptırıcı kararlılığı sorusu)

Senaryo t1_defl_trim.make ile aynı (5 μm simetrik kuadrupol sürüklenmesi, 14 nm, saptırıcı sütunlu fit +
trim); yalnız saptırıcı sürüklenmesinin genliği A = σ_ε = σ_φ değişir: 1e-6, 3e-6, 1e-5 (T1'de 3e-7).
Durumlar: trim (nominal model), trim_g20 (%0.2 gradyan hatası). Seed 0–3, iki yön, donmuş spin (kısaltılmış).
Beklenti (tahmin): C ∝ A² → 3e-7'deki 0.010 / 0.016 medyandan 1e-6'da ~0.1, 3e-6'da ~1, 1e-5'te ~10.
"""
import os, sys, json, argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor

REV = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REV)
from sahte_edm_donmus import measure_frozen
import integrator_rev

AMPS = [1e-6, 3e-6, 1e-5]
CASES = ["trim", "trim_g20"]
OUT = os.path.join(REV, "sonuclar", "donmus_t1_genlik.jsonl")


def task(t):
    amp, seed, case, d = t
    import adim2_deflector as A2
    A2.SIG_EPS = amp; A2.SIG_ROLL = amp
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
    r = measure_frozen(dx, dy, direction=d, dG=dG, n_turns=14, n_iter=2)
    return dict(amp=amp, seed=seed, case=case, direction=d, f=r["f"], f_target=r["f"] / 1e-9,
                w1=r["w1"], sym_um=float(np.mean([np.std(A1.PS @ dx), np.std(A1.PS @ dy)]) * 1e6),
                defl_left_eps=float(np.std(o["x"][1])), defl_left_roll=float(np.std(o["y"][1])))


def main(workers):
    done = set()
    if os.path.exists(OUT):
        done = {(r["amp"], r["seed"], r["case"], r["direction"]) for r in map(json.loads, open(OUT))}
    T = [(a, s, c, d) for a in AMPS for c in CASES for s in range(4) for d in (-1, +1) if (a, s, c, d) not in done]
    print(f"{len(T)} görev", flush=True)
    with ProcessPoolExecutor(workers) as pool:
        for r in pool.map(task, T):
            open(OUT, "a").write(json.dumps(r) + "\n")
            print(f"A={r['amp']:.0e} seed {r['seed']} {r['case']:8s} yön {r['direction']:+d}: f* = {r['f_target']:12.4f}", flush=True)
    summarize()


def summarize():
    rows = [json.loads(l) for l in open(OUT)]
    ref = {"trim": 0.0103, "trim_g20": 0.0160}     # 3e-7, donmus_t1_kisa medyanları
    for c in CASES:
        for a in AMPS:
            Cs = []
            for s in range(4):
                f = {r["direction"]: r["f"] for r in rows if r["amp"] == a and r["case"] == c and r["seed"] == s}
                if len(f) == 2:
                    Cs.append(abs(f[1] - f[-1]) / 2e-9)
            if Cs:
                pred = ref[c] * (a / 3e-7) ** 2
                print(f"{c:8s} A={a:.0e}: n={len(Cs)} C medyan {np.median(Cs):.3g} [{min(Cs):.3g}, {max(Cs):.3g}]  "
                      f"(A² öngörüsü {pred:.3g})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("-w", "--workers", type=int, default=4)
    ap.add_argument("--ozet", action="store_true"); a = ap.parse_args()
    summarize() if a.ozet else main(a.workers)
