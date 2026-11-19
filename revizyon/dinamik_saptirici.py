#!/usr/bin/env python3
"""dinamik_saptirici.py — Saptırıcı sürüklenmesi + periyodik trim döngüsü (yarı-analitik dinamik test).

Soru (makale Sec. V.B / VI): haftalık senaryoda saptırıcı alanı (ε) ve roll'u (φ) bağımsız rastgele
yürüyüşle sürüklenirse, iki demetli fit + kuadrupol düzeltmesi + bireysel saptırıcı trimi + ortak düşey
trim (Omarov) döngüsü sahte EDM'yi hedefin altında tutar mı? Statik testler (donmus_t1_*) yalnız tek
seferlik trimi ölçüyordu; burada trimler tekrarlanır ve kestirilemeyen yönlerin birikip birikmediği görülür.

Model:
  - Yörünge düzeyi döngü DOĞRUSAL (tepki matrisleri; t1_defl_trim.make ile aynı fit): okuma
    y = T_q dm + T_d de + gürültü (T: %0.2 gradyan hatalı 'gerçek' makine), fit nominal modelle
    (TSVD, rcond 1e-4, kuadrupol + saptırıcı sütunları). Her trim adımında dm -= m̂ (kuadrupol içi
    düzeltici), de -= ê (bireysel trim), ortak trim: ortalama roll sıfırlanır; ortalama ε donmuş-spin
    enerji geri beslemesince emilir (çıkarılır).
  - Saptırıcı sürüklenmesi: her saptırıcıda ε ve φ bağımsız, hız r (1/√saat).
  - Kuadrupol sürüklenmesi: bağımsız rastgele yürüyüş, q_rate (m/√saat) [varsayım].
  - Spin (C++ donmuş-spin izleme) yalnız seçili anlarda, o anki gerçek durumla (dm, de) ölçülür:
    trimden HEMEN ÖNCE (son trim aralığının sürüklenmesi dahil; en kötü an).
Çıktı: sonuclar/dinamik_saptirici_lin.json (zaman serileri), sonuclar/dinamik_saptirici_edm.jsonl.
Kullanım:  python3 dinamik_saptirici.py lin
           python3 dinamik_saptirici.py edm -w 4
"""
import os, sys, json, argparse
import numpy as np

REV = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REV)
import adim2_deflector as A2
NQ = 48
RES = os.path.join(REV, "sonuclar")
OUT_LIN = os.path.join(RES, "dinamik_saptirici_lin.json")
OUT_EDM = os.path.join(RES, "dinamik_saptirici_edm.jsonl")

RATES = [1e-7, 1e-6, 1e-5]       # saptırıcı sürüklenme hızı (1/√saat)
TRIM_H = [1.0]                   # trim aralığı (saat)
HOURS = 168
SEEDS = range(4)
Q_RATE = 0.1e-6                  # kuadrupol sürüklenmesi, m/√saat (varsayım; ~1.3 μm/hafta)
SIG_N = 14e-9
RC = 1e-4
DM0 = 0.3e-6                     # BBA sonrası kalan rms kaçıklık


def mats():
    from build_rtrue import path as rtp
    M = A2.load_mats(); T = {}
    for pl, rk, dk in (("x", "dx", "eps"), ("y", "dy", "roll")):
        T[pl] = {"R_ccw": np.load(rtp(rk, "ccw", 0.002)), "R_cw": np.load(rtp(rk, "cw", 0.002)),
                 "D_ccw": np.load(A2.DPATH(dk, "ccw", 0.002)), "D_cw": np.load(A2.DPATH(dk, "cw", 0.002))}
    S = {}
    for pl in ("x", "y"):
        m, t = M[pl], T[pl]
        S[pl] = dict(S=np.hstack([np.vstack([m["R_ccw"], m["R_cw"]]), np.vstack([m["D_ccw"], m["D_cw"]])]),
                     Tq=np.vstack([t["R_ccw"], t["R_cw"]]), Td=np.vstack([t["D_ccw"], t["D_cw"]]))
    return S


def run_one(S, seed, rate, trim_h, hours=HOURS, keep=(24, 168)):
    """Bir gerçekleşme. Dönüş: zaman serisi + seçili saatlerde trim-öncesi durum."""
    rng = np.random.default_rng(70_000 + seed)
    Ps, _ = A2.projectors()
    dm = {pl: rng.normal(0, DM0, NQ) for pl in ("x", "y")}
    de = {pl: np.zeros(NQ) for pl in ("x", "y")}
    n = int(round(hours / trim_h)); ser = dict(h=[], de_pre=[], de_post=[], sym=[]); snaps = {}
    for k in range(1, n + 1):
        h = k * trim_h
        for pl in ("x", "y"):
            dm[pl] += rng.normal(0, Q_RATE * np.sqrt(trim_h), NQ)
            de[pl] += rng.normal(0, rate * np.sqrt(trim_h), NQ)
        # ortak trim (Omarov) + enerji geri beslemesi: ortalamalar çıkarılır (sürekli çalışır)
        for pl in ("x", "y"):
            de[pl] -= de[pl].mean()
        pre = {pl: (dm[pl].copy(), de[pl].copy()) for pl in ("x", "y")}
        if int(round(h)) in keep and abs(h - round(h)) < 1e-9:
            snaps[int(round(h))] = {pl: [pre[pl][0].tolist(), pre[pl][1].tolist()] for pl in ("x", "y")}
        de_pre = float(np.sqrt(np.mean(np.r_[de["x"], de["y"]] ** 2)))
        for pl in ("x", "y"):
            s = S[pl]
            y = s["Tq"] @ dm[pl] + s["Td"] @ de[pl] + rng.normal(0, SIG_N, 2 * NQ)
            est = A2.tsvd_solve(s["S"], y, RC)
            dm[pl] -= est[:NQ]; de[pl] -= est[NQ:]
        if int(round(h)) in keep and abs(h - round(h)) < 1e-9:
            snaps[f"{int(round(h))}p"] = {pl: [dm[pl].tolist(), de[pl].tolist()] for pl in ("x", "y")}
        ser["h"].append(h); ser["de_pre"].append(de_pre)
        ser["de_post"].append(float(np.sqrt(np.mean(np.r_[de["x"], de["y"]] ** 2))))
        ser["sym"].append(float(np.mean([np.std(Ps @ dm["x"]), np.std(Ps @ dm["y"])])))
    return ser, snaps


def lin():
    S = mats(); out = {"meta": dict(RATES=RATES, TRIM_H=TRIM_H, HOURS=HOURS, Q_RATE=Q_RATE, SIG_N=SIG_N,
                                     RC=RC, DM0=DM0, seed_base=70000), "runs": {}}
    for r in RATES:
        for T in TRIM_H:
            for s in SEEDS:
                ser, snaps = run_one(S, s, r, T)
                out["runs"][f"{r:.0e}_{T:g}_{s}"] = dict(rate=r, trim_h=T, seed=s, ser=ser, snaps=snaps)
                d = np.array(ser["de_post"]); p = np.array(ser["de_pre"])
                print(f"r={r:.0e}/√h T={T:g}h seed {s}: de trim-öncesi son {p[-1]:.2e}, trim-sonrası "
                      f"1h {d[0]:.2e} 24h {d[23]:.2e} 168h {d[-1]:.2e}, sym {ser['sym'][-1]*1e6:.3f} μm", flush=True)
    json.dump(out, open(OUT_LIN, "w"))


def _edm_task(t):
    key, h, d = t
    import integrator_rev
    from sahte_edm_donmus import measure_frozen
    from build_rtrue import dG_pattern
    run = json.load(open(OUT_LIN))["runs"][key]
    sn = run["snaps"][str(h)]
    integrator_rev.DEFL_DE = np.asarray(sn["x"][1], float)
    integrator_rev.DEFL_ROLL = np.asarray(sn["y"][1], float)
    r = measure_frozen(np.asarray(sn["x"][0]), np.asarray(sn["y"][0]), direction=d, dG=dG_pattern(0.002),
                       n_turns=14, n_iter=2)
    return dict(key=key, rate=run["rate"], trim_h=run["trim_h"], seed=run["seed"], hour=h, direction=d,
                f=r["f"], f_target=r["f"] / 1e-9, w1=r["w1"])


def edm(workers, hours=(168,), rates=None):
    from concurrent.futures import ProcessPoolExecutor
    import contextlib, io
    runs = json.load(open(OUT_LIN))["runs"]
    done = set()
    if os.path.exists(OUT_EDM):
        done = {(r["key"], r["hour"], r["direction"]) for r in map(json.loads, open(OUT_EDM))}
    T = [(k, h, d) for k in runs for h in hours for d in (-1, +1) if (k, h, d) not in done
         and (rates is None or runs[k]["rate"] in rates)]
    print(f"{len(T)} görev", flush=True)
    with ProcessPoolExecutor(workers) as pool:
        for r in pool.map(_edm_task, T):
            open(OUT_EDM, "a").write(json.dumps(r) + "\n")
            print(f"{r['key']} h={r['hour']} yön {r['direction']:+d}: f*={r['f_target']:.4g}", flush=True)
    summ()


def summ():
    rows = [json.loads(l) for l in open(OUT_EDM)]
    d = {}
    for r in rows:
        d.setdefault((r["rate"], r["trim_h"], r["hour"], r["seed"]), {})[r["direction"]] = r["f"]
    agg = {}
    for (rate, T, h, s), v in d.items():
        if len(v) == 2:
            agg.setdefault((rate, T, h), []).append(abs(v[1] - v[-1]) / 2 / 1e-9)
    for k in sorted(agg):
        c = np.array(agg[k])
        print(f"r={k[0]:.0e}/√h T={k[1]:g}h saat {k[2]}: n={len(c)} C medyan {np.median(c):.3g} maks {c.max():.3g}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["lin", "edm", "edmpost", "summ"])
    ap.add_argument("-w", type=int, default=4); a = ap.parse_args()
    {"lin": lin, "edm": lambda: edm(a.w), "summ": summ,
     # trim-SONRASI durum (h='168p'): sinyali birikmiş sürüklenme mi, trim artığı mı taşıyor?
     "edmpost": lambda: edm(a.w, hours=("168p",), rates=(1e-6, 1e-5))}[a.mode]()
