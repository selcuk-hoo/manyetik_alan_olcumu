#!/usr/bin/env python3
"""teshis_ortak_besleme.py — Tek (ortak) HV beslemesi: ortak ε zararsız mı? (S+ sıfırlanmış hâliyle)

Önceki deneme (saptirici_testleri 'ortak') sonuçsuzdu: ortak ε eklenince iki demet ayrı ayrı donmuş spin
enerjisine kilitleniyor, aradaki küçük enerji farkı sıfırlanmamış çift kanalı (S+ ~ 100 × hedef) tek kanala
sızdırıyor (C ≈ |S+|·2.7Δδ). Gerçek makinede Omarov'un ortak düşey trimi S+'yı sıfırlar. Burada önce S+
sıfırlanır, sonra C ölçülür.

Prosedür (her seed ve ortak ε = 0, 1e-5, 1e-4 için):
  1. Durum: dinamik_saptirici r = 1e-7/√h, 168 h trim-öncesi (kuadrupol artığı + bağımsız ε, φ) + ortak ε.
  2. İki yönde ölç → S+₀ = ½(f_CW + f_CCW).
  3. Ortak roll φ₁ = 1e-11 ekle, iki yönde ölç → eğim k = (S+₁ − S+₀)/φ₁.
  4. φ* = −S+₀/k ekle (ortak trim), iki yönde ölç → C = ½|f_CW − f_CCW|, kalan S+.
Çıktı: sonuclar/teshis_ortak_besleme.jsonl
"""
import os, sys, json
import numpy as np
from concurrent.futures import ProcessPoolExecutor

T = os.path.dirname(os.path.abspath(__file__)); REV = os.path.dirname(T)
sys.path[:0] = [REV, T]
OUT = os.path.join(REV, "sonuclar", "teshis_ortak_besleme.jsonl")
SEEDS = range(4)
AMPS = (0.0, 1e-5, 1e-4)
PHI1 = 1e-11
WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 4   # işçi sayısı: python3 teshis_ortak_besleme.py 10


def meas(t):
    import integrator_rev
    import dinamik_saptirici as DS
    import saptirici_testleri as ST
    from build_rtrue import dG_pattern
    amp, seed, phi, d = t
    sn = json.load(open(DS.OUT_LIN))["runs"][f"1e-07_1_{seed}"]["snaps"]["168"]
    integrator_rev.DEFL_DE = np.asarray(sn["x"][1], float) + amp
    integrator_rev.DEFL_ROLL = np.asarray(sn["y"][1], float) + phi
    r = ST.measure(np.asarray(sn["x"][0]), np.asarray(sn["y"][0]), d, dG_pattern(0.002), False)
    return dict(amp=amp, seed=seed, phi=phi, direction=d, f=r["f"], f_target=r["f"] / 1e-9, w1=r["w1"])


def load():
    return [json.loads(l) for l in open(OUT)] if os.path.exists(OUT) else []


def run_batch(tasks):
    done = {(r["amp"], r["seed"], r["phi"], r["direction"]) for r in load()}
    tasks = [t for t in tasks if t not in done]
    if not tasks:
        return
    with ProcessPoolExecutor(WORKERS) as pool:
        for r in pool.map(meas, tasks):
            open(OUT, "a").write(json.dumps(r) + "\n")
            print(r, flush=True)


def f_of(amp, seed, phi):
    v = {r["direction"]: r["f_target"] for r in load() if r["amp"] == amp and r["seed"] == seed and r["phi"] == phi}
    return v if len(v) == 2 else None


if __name__ == "__main__":
    run_batch([(a, s, 0.0, d) for a in AMPS for s in SEEDS for d in (-1, 1)] +
              [(a, s, PHI1, d) for a in AMPS for s in SEEDS for d in (-1, 1)])
    star = {}
    for a in AMPS:
        for s in SEEDS:
            v0, v1 = f_of(a, s, 0.0), f_of(a, s, PHI1)
            sp0 = (v0[1] + v0[-1]) / 2; sp1 = (v1[1] + v1[-1]) / 2
            star[(a, s)] = float(-sp0 / ((sp1 - sp0) / PHI1))
    run_batch([(a, s, star[(a, s)], d) for a in AMPS for s in SEEDS for d in (-1, 1)])
    print("\nortak ε | seed | S+ önce | φ* | S+ sonra | C sonra")
    for a in AMPS:
        Cs = []
        for s in SEEDS:
            v0, vs = f_of(a, s, 0.0), f_of(a, s, star[(a, s)])
            C = abs(vs[1] - vs[-1]) / 2; Cs.append(C)
            print(f"{a:7.0e} | {s} | {(v0[1]+v0[-1])/2:9.3g} | {star[(a, s)]:.3e} | {(vs[1]+vs[-1])/2:9.3g} | {C:.3g}")
        print(f"  ortak ε {a:.0e}: C medyan {np.median(Cs):.3g} maks {max(Cs):.3g}")
