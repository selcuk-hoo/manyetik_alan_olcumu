#!/usr/bin/env python3
"""teshis_roll_cift_kanal.py — saptırıcı roll'unun birinci mertebe (çift kanal) etkisi:
(1) yalnız Σφ'ye mi bağlı? Aynı rastgele roll deseni (adim2_roll_C: rng 30000, rms 3e-7), ortalaması
    çıkarılmış hâli ile karşılaştırılır. Kuadrupol kaçıklığı yok. Donmuş spin, iki yön.
(2) roll kaynaklı f'nin demet enerjisine duyarlılığı: tam desen, CCW, δ = ±1e-6 (donmuş noktaya göre
    değil, enerji korunumlu başlangıca göre) → d ln f / dδ → iki demetin enerji farkının yarattığı asimetri.
"""
import os, sys, json, tempfile
import numpy as np
B = "/home/user/manyetik_alan_olcumu"
sys.path[:0] = [B + "/revizyon", B + "/revizyon/tests", B, B + "/berry_data", B + "/kmod_drivers"]
OUT = B + "/revizyon/sonuclar/teshis_roll_cift_kanal.json"
NQ = 48


def roll(kind):
    e = np.random.default_rng(30000).normal(0, 1, NQ) * 3e-7
    return e - e.mean() if kind == "ortsiz" else e


def run(t):
    os.chdir(tempfile.mkdtemp())
    import integrator_rev, sahte_edm_donmus as SD
    kind, d, delta = t
    integrator_rev.DEFL_ROLL = roll(kind); integrator_rev.DEFL_DE = np.zeros(NQ)
    z = np.zeros(NQ)
    if delta is None:
        r = SD.measure_frozen(z, z, direction=d, n_turns=14, n_iter=2)
        return dict(kind=kind, direction=d, delta=None, f=r["f"], f_target=r["f"] / 1e-9, w1=r["w1"],
                    sum_phi=float(roll(kind).sum()))
    f, w, _ = SD._one(z, z, d, delta, None, None, 14, 2, 3e-4)
    return dict(kind=kind, direction=d, delta=delta, f=f, f_target=f / 1e-9, w=w, sum_phi=float(roll(kind).sum()))


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    T = [("tam", -1, None), ("tam", 1, None), ("ortsiz", -1, None), ("ortsiz", 1, None),
         ("tam", -1, -1e-6), ("tam", -1, 1e-6)]
    with ProcessPoolExecutor(3) as p:
        R = list(p.map(run, T))
    json.dump(R, open(OUT, "w"), indent=1)
    for r in R:
        print(r)
