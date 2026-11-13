#!/usr/bin/env python3
"""teshis_roll_eslestirilmis.py — roll × kapalı yörünge çapraz teriminin eşleştirilmiş testi (ChatGPT madde 4).

Her desen için AYNI başlangıç: kuadrupol kaçıklığı 0.3 μm rms (iki düzlem, BBA artığı), saptırıcı roll
φ ~ N(0, 3e-7) (ε = 0). Üç durum:
  notrim : kuadrupol ve roll olduğu gibi
  ortak  : yalnız ortak trim (roll ortalaması çıkarılır)
  birey  : ortak trim + iki demetli fitten (kuadrupol + saptırıcı sütunları, 14 nm, TSVD 1e-4)
           bireysel roll trimi (φ − φ̂) ve kuadrupol düzeltmesi (dm − m̂); ortalama yine çıkarılır
  ref    : roll hiç yok, kuadrupol kaçıklığı aynı (karşılaştırma)
Seed 0–4, iki yön, donmuş spin (kısaltılmış). C = ½|f_CW − f_CCW|.
"""
import os, sys, json, tempfile
import numpy as np
B = "/home/user/manyetik_alan_olcumu"
sys.path[:0] = [B + "/revizyon", B + "/revizyon/tests", B, B + "/berry_data", B + "/kmod_drivers"]
OUT = B + "/revizyon/sonuclar/teshis_roll_eslestirilmis.jsonl"
NQ = 48


def config(seed, case):
    import adim2_deflector as A2
    M = A2.load_mats()
    rng = np.random.default_rng(70_000 + seed)
    dm = {pl: rng.normal(0, 0.3e-6, NQ) for pl in "xy"}
    phi = rng.normal(0, 3e-7, NQ)
    nz = {pl: rng.normal(0, 14e-9, 2 * NQ) for pl in "xy"}
    if case == "ref":
        return dm["x"], dm["y"], np.zeros(NQ)
    if case == "notrim":
        return dm["x"], dm["y"], phi
    if case == "ortak":
        return dm["x"], dm["y"], phi - phi.mean()
    # birey: iki demetli fit, kuadrupol + saptırıcı sütunları (y: roll; x: ε sütunları, ε=0)
    out = {}
    for pl, de in (("x", np.zeros(NQ)), ("y", phi)):
        m = M[pl]
        Sq = np.vstack([m["R_ccw"], m["R_cw"]]); Sd = np.vstack([m["D_ccw"], m["D_cw"]])
        y = Sq @ dm[pl] + Sd @ de + nz[pl]
        est = A2.tsvd_solve(np.hstack([Sq, Sd]), y, 1e-4)
        out[pl] = (dm[pl] - est[:NQ], de - est[NQ:])
    left = out["y"][1]
    return out["x"][0], out["y"][0], left - left.mean()


def run(t):
    seed, case, d = t
    os.chdir(tempfile.mkdtemp())
    import integrator_rev, sahte_edm_donmus as SD
    dx, dy, roll = config(seed, case)
    z = np.zeros(NQ)
    integrator_rev.DEFL_ROLL = roll; integrator_rev.DEFL_DE = z
    integrator_rev.DEFL_BR = z; integrator_rev.DEFL_BV = z
    r = SD.measure_frozen(dx, dy, direction=d, n_turns=14, n_iter=2)
    return dict(seed=seed, case=case, direction=d, f=r["f"], f_target=r["f"] / 1e-9, w1=r["w1"],
                roll_rms=float(np.std(roll)), sum_phi=float(roll.sum()))


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    done = set()
    if os.path.exists(OUT):
        done = {(r["seed"], r["case"], r["direction"]) for r in map(json.loads, open(OUT))}
    T = [(s, c, d) for s in range(5) for c in ("ref", "notrim", "ortak", "birey") for d in (-1, 1)
         if (s, c, d) not in done]
    with ProcessPoolExecutor(4) as p:
        for r in p.map(run, T):
            open(OUT, "a").write(json.dumps(r) + "\n"); print(r, flush=True)
