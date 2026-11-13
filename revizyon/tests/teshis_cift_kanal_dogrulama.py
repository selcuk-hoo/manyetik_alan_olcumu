#!/usr/bin/env python3
"""teshis_cift_kanal_dogrulama.py — çift kanal geri beslemesi önerisinin doğrulaması (ChatGPT değerlendirmesi).

A  EDM korunuyor mu? EDM açık; (i) roll yok, (ii) rastgele roll (rms 3e-7), (iii) aynı roll + ortak trim
   (ortalama çıkarılmış = trim uygulanmış). İki yön. Tek kanal ½(f_CW − f_CCW) EDM'ye eşit kalmalı.
B  Σφ = 0 iken birinci mertebe iptalin genelliği: 4 farklı rastgele desen + seed-30000 deseninin 2
   permütasyonu (ortalama çıkarılmış), CCW; ayrıca 0.3 μm rms kuadrupol kaçıklığı (BBA artığı düzeyi,
   kapalı yörünge ≠ 0) ile roll'lu/roll'suz, iki yön.
C  Kanal tablosu: düzgün radyal manyetik alan (saptırıcılarda B_r = 1e-13 T), iki yön;
   CW yönünde d ln f/dδ (roll kaynaklı terim).
"""
import os, sys, json, tempfile
import numpy as np
B = "/home/user/manyetik_alan_olcumu"
sys.path[:0] = [B + "/revizyon", B + "/revizyon/tests", B, B + "/berry_data", B + "/kmod_drivers"]
OUT = B + "/revizyon/sonuclar/teshis_cift_kanal_dogrulama.jsonl"
NQ = 48


def roll_pattern(seed, mean_removed, perm=None):
    e = np.random.default_rng(seed).normal(0, 1, NQ) * 3e-7
    if perm is not None:
        e = e[np.random.default_rng(perm).permutation(NQ)]
    return e - e.mean() if mean_removed else e


def tasks():
    T = []
    for d in (-1, 1):
        T += [("A_edm_yalin", d), ("A_edm_roll", d), ("A_edm_roll_trim", d)]
    for s in (30001, 30002, 30003, 30004):
        T.append((f"B_desen_{s}", -1))
    for p in (1, 2):
        T.append((f"B_perm_{p}", -1))
    for d in (-1, 1):
        T += [("B_q03_yalin", d), ("B_q03_roll_trim", d)]
    for d in (-1, 1):
        T.append(("C_Br", d))
    T += [("C_cw_dm", 1), ("C_cw_dp", 1)]
    return T


def run(t):
    name, d = t
    os.chdir(tempfile.mkdtemp())
    import integrator_rev, sahte_edm_donmus as SD, false_edm_mode_scan as FM
    z = np.zeros(NQ)
    integrator_rev.DEFL_ROLL = z; integrator_rev.DEFL_DE = z; integrator_rev.DEFL_BR = z; integrator_rev.DEFL_BV = z
    dx = dy = z
    edm = name.startswith("A_")
    if edm:
        orig = FM.setup_fields
        def setup_edm(cfg):
            r = orig(cfg); r[0].EDMSwitch = 1.0; return r
        FM.setup_fields = setup_edm
    if name in ("A_edm_roll",):
        integrator_rev.DEFL_ROLL = roll_pattern(30000, False)
    if name in ("A_edm_roll_trim", "B_q03_roll_trim"):
        integrator_rev.DEFL_ROLL = roll_pattern(30000, True)
    if name.startswith("B_desen_"):
        integrator_rev.DEFL_ROLL = roll_pattern(int(name.split("_")[-1]), True)
    if name.startswith("B_perm_"):
        integrator_rev.DEFL_ROLL = roll_pattern(30000, True, perm=int(name.split("_")[-1]))
    if name.startswith("B_q03"):
        rng = np.random.default_rng(777)
        dx = rng.normal(0, 0.3e-6, NQ); dy = rng.normal(0, 0.3e-6, NQ)
    if name == "C_Br":
        integrator_rev.DEFL_BR = np.full(NQ, 1e-13)
    if name.startswith("C_cw_d"):
        integrator_rev.DEFL_ROLL = roll_pattern(30000, False)
        delta = -1e-6 if name.endswith("dm") else 1e-6
        f, w, _ = SD._one(z, z, d, delta, None, None, 14, 2, 3e-4)
        return dict(name=name, direction=d, f=f, f_target=f / 1e-9, w=w)
    r = SD.measure_frozen(dx, dy, direction=d, n_turns=14, n_iter=2)
    return dict(name=name, direction=d, f=r["f"], f_target=r["f"] / 1e-9, w1=r["w1"],
                sum_phi=float(np.sum(integrator_rev.DEFL_ROLL)))


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    done = set()
    if os.path.exists(OUT):
        done = {(r["name"], r["direction"]) for r in map(json.loads, open(OUT))}
    T = [t for t in tasks() if t not in done]
    with ProcessPoolExecutor(4) as p:
        for r in p.map(run, T):
            open(OUT, "a").write(json.dumps(r) + "\n")
            print(r, flush=True)
