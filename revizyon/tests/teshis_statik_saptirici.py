#!/usr/bin/env python3
"""teshis_statik_saptirici.py — STATİK (yapım kaynaklı) saptırıcı hataları: polarite çevirmeden ne kalır?

Bulgu (saptirici_testleri.py T4): kuadrupol polaritesi çevrilince CW ve CCW'nin sahte EDM'si birebir yer
değiştiriyor (f_CW(−g) = f_CCW(+g)); bu yüzden Omarov birleşimi ½(odd₊ + odd₋) saptırıcı ε×φ terimini
sayısal tabana (0.02) indiriyor. Bizim şema polarite çevirmez. Dinamik testlerde saptırıcı hatası BBA
referansında sıfırdı; gerçek makinede ise yapımdan gelen STATİK ε, φ vardır ve iki demetli fit yalnız
referanstan bu yana DEĞİŞİMİ görür. Soru: BBA sonrası (demet kuadrupol merkezlerinden geçecek şekilde
düzelticilerle yönlendirilmiş) statik ε, φ ne kadar sahte EDM bırakır?

Model: ε, φ ~ N(0, σ) (ortalaması çıkarılmış: ortak trim + enerji geri beslemesi), kuadrupoller kusursuz.
Düzeltici ≡ kuadrupol merkezinin etkin kayması c (CLAUDE.md tuzak 10); c, iki demetin kuadrupollerdeki
kapalı yörüngesini en küçük kareler anlamında sıfırlar: [R_ccw; R_cw] c = −[D_ccw; D_cw] e (nominal model,
TSVD 1e-4). Sonra C++ donmuş spin ile ölçülür. σ = 1e-6, 1e-5; seed 0–2; ayrıca σ = 1e-5'te polarite çevrili.
Çıktı: sonuclar/teshis_statik_saptirici.jsonl
"""
import os, sys, json
import numpy as np
from concurrent.futures import ProcessPoolExecutor

T = os.path.dirname(os.path.abspath(__file__)); REV = os.path.dirname(T)
sys.path.insert(0, REV)
OUT = os.path.join(REV, "sonuclar", "teshis_statik_saptirici.jsonl")
NQ = 48


def state(seed, sig):
    import adim2_deflector as A2
    M = A2.load_mats(); rng = np.random.default_rng(90_000 + seed)
    e = {pl: rng.normal(0, sig, NQ) for pl in ("x", "y")}
    for pl in e:
        e[pl] -= e[pl].mean()
    c = {}
    for pl in ("x", "y"):
        m = M[pl]
        Rq = np.vstack([m["R_ccw"], m["R_cw"]]); Dd = np.vstack([m["D_ccw"], m["D_cw"]])
        c[pl] = -A2.tsvd_solve(Rq, Dd @ e[pl], 1e-4)
        res = np.linalg.norm(Rq @ c[pl] + Dd @ e[pl]) / np.linalg.norm(Dd @ e[pl])
        c[pl + "_res"] = float(res)
    return e, c


def task(t):
    import integrator_rev
    import saptirici_testleri as ST
    sig, seed, d, flip = t
    e, c = state(seed, sig)
    integrator_rev.DEFL_DE = e["x"]; integrator_rev.DEFL_ROLL = e["y"]
    r = ST.measure(c["x"], c["y"], d, None, flip)
    return dict(sigma=sig, seed=seed, direction=d, flip=flip, f=r["f"], f_target=r["f"] / 1e-9, w1=r["w1"],
                orbit_res_x=c["x_res"], orbit_res_y=c["y_res"],
                corr_rms_um=float(np.mean([np.std(c["x"]), np.std(c["y"])]) * 1e6))


if __name__ == "__main__":
    done = set()
    if os.path.exists(OUT):
        done = {(r["sigma"], r["seed"], r["direction"], r["flip"]) for r in map(json.loads, open(OUT))}
    TS = [(s, k, d, False) for s in (1e-6, 1e-5) for k in range(3) for d in (-1, 1)]
    TS += [(1e-5, k, d, True) for k in range(3) for d in (-1, 1)]
    TS = [t for t in TS if t not in done]
    print(len(TS), "görev", flush=True)
    with ProcessPoolExecutor(4) as pool:
        for r in pool.map(task, TS):
            open(OUT, "a").write(json.dumps(r) + "\n")
            print(r, flush=True)
    rows = [json.loads(l) for l in open(OUT)]
    g = {}
    for r in rows:
        g.setdefault((r["sigma"], r["flip"], r["seed"]), {})[r["direction"]] = r["f_target"]
    for (s, fl, k), v in sorted(g.items()):
        if len(v) == 2:
            print(f"σ={s:.0e} flip={fl} seed {k}: odd {(v[1]-v[-1])/2:+.4g}  even {(v[1]+v[-1])/2:+.4g}")
