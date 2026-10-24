#!/usr/bin/env python3
"""t1_defl_trim.py — Tartışma testi T1: saptırıcı sürüklenmesini kaynağında trimlemek.

Adım 2'de saptırıcı sütunlu fit (QD) saptırıcı sürüklenmesini doğru KESTİRİYOR ama yalnız
kuadrupoller düzeltildiğinde ε×φ çarpım etkisi (saptırıcıların kendi ürettiği x_CO·y_CO
geometrik fazı) C≈1 bırakıyordu. Burada kestirilen Δε, Δφ her saptırıcıya geri uygulanır:
  Δε → saptırıcı yüksek-gerilim trimi
  Δφ → saptırıcıda düşey elektrik trimi (üst/alt plaka gerilim farkı); öncü terimi E_z≈E0·φ
       olduğundan simülasyonda roll'dan çıkarılarak modellenir (alan deseninin dönmesinden
       gelen ikinci mertebe fark ihmal).
Senaryo Adım 2 'edm' ile aynı (seed 30000+, 5 μm simetrik kuadrupol sürüklenmesi,
ε, φ ~ N(0, 3e-7), 14 nm gürültü).

Durumlar:
  trim       : QD kestirimi → kuadrupol (Mq) + saptırıcı trimi, nominal model
  oracle     : saptırıcı sürüklenmesi TAM biliniyor ve trimleniyor; kuadrupol QD ile (kontrol)
  trim_g20   : gerçek makine %0.2 gradyan hatalı (okuma + spin), model nominal
  QD_g20     : aynı, yalnız kuadrupol düzeltmesi (karşılaştırma)
"""
import os, sys, json, argparse
import numpy as np

REV = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REV)
import adim2_deflector as A2            # integrator_rev'i 'integrator' olarak takar
import integrator_rev

OUTDIR = A2.OUTDIR
NQ = 48
TARGET = 1e-9
JSONL = os.path.join(OUTDIR, "t1_edm.jsonl")
CASES = ["trim", "oracle", "trim_g20", "QD_g20"]


def make(seed, case, sig_n=14e-9, drift=5e-6, rc_qd=1e-4):
    g20 = case.endswith("_g20")
    M = A2.load_mats(); Ps, _ = A2.projectors()
    T = M
    if g20:
        from build_rtrue import path as rtp
        T = {}
        for pl, rk, dk in (("x", "dx", "eps"), ("y", "dy", "roll")):
            T[pl] = {"R_ccw": np.load(rtp(rk, "ccw", 0.002)), "R_cw": np.load(rtp(rk, "cw", 0.002)),
                     "D_ccw": np.load(A2.DPATH(dk, "ccw", 0.002)),
                     "D_cw": np.load(A2.DPATH(dk, "cw", 0.002))}
    rng = np.random.default_rng(30_000 + seed)      # Adım 2 'edm' ile aynı çekim sırası
    out = {}
    for pl in ("x", "y"):
        m, t = M[pl], T[pl]
        Sq = np.vstack([m["R_ccw"], m["R_cw"]]); Sd = np.vstack([m["D_ccw"], m["D_cw"]])
        Tq = np.vstack([t["R_ccw"], t["R_cw"]]); Td = np.vstack([t["D_ccw"], t["D_cw"]])
        dm = Ps @ rng.normal(0, 1, NQ); dm *= drift / np.std(dm)
        de = rng.normal(0, 1, NQ) * (A2.SIG_EPS if pl == "x" else A2.SIG_ROLL)
        y = Tq @ dm + Td @ de + rng.normal(0, sig_n, 2 * NQ)
        est = A2.tsvd_solve(np.hstack([Sq, Sd]), y, rc_qd)
        m_hat, e_hat = est[:NQ], est[NQ:]
        res = dm - m_hat
        if case == "oracle":
            e_left = np.zeros(NQ)
        elif case == "QD_g20":
            e_left = de
        else:
            e_left = de - e_hat
        out[pl] = (res, e_left, de, e_hat)
    return out


def _task(task):
    import tempfile
    os.chdir(tempfile.mkdtemp())
    sys.modules["integrator"] = integrator_rev
    from fast_est import fast_measure
    seed, case = task
    o = make(seed, case)
    dx, dy = o["x"][0], o["y"][0]
    integrator_rev.DEFL_DE = o["x"][1]; integrator_rev.DEFL_ROLL = o["y"][1]
    dG = None
    if case.endswith("_g20"):
        from build_rtrue import dG_pattern
        dG = dG_pattern(0.002)
    fcw, _ = fast_measure(dx, dy, direction=+1, signed=True, n_turns=28, n_iter=4, dG=dG)
    fccw, _ = fast_measure(dx, dy, direction=-1, signed=True, n_turns=28, n_iter=4, dG=dG)
    Ps, _ = A2.projectors()
    rel = lambda k: float(np.std(o[k][1]) / np.std(o[k][2]))
    return dict(seed=seed, case=case, f_cw=fcw / TARGET, f_ccw=fccw / TARGET,
                C=abs(fcw - fccw) / 2 / TARGET, even=(fcw + fccw) / 2 / TARGET,
                sym_x=float(np.std(Ps @ dx) * 1e6), sym_y=float(np.std(Ps @ dy) * 1e6),
                defl_left_rel_eps=rel("x"), defl_left_rel_roll=rel("y"))


def run(seeds, workers):
    from concurrent.futures import ProcessPoolExecutor
    done = set()
    if os.path.exists(JSONL):
        done = {(d["seed"], d["case"]) for d in map(json.loads, open(JSONL))}
    tasks = [(s, c) for s in seeds for c in CASES if (s, c) not in done]
    print(f"{len(tasks)} görev", flush=True)
    with ProcessPoolExecutor(workers) as pool:
        for r in pool.map(_task, tasks):
            open(JSONL, "a").write(json.dumps(r) + "\n")
            print(f"  seed {r['seed']} {r['case']}: C={r['C']:.4f}× çift={r['even']:.4g}× "
                  f"(sim {r['sym_x']:.3f}/{r['sym_y']:.3f} μm; kalan saptırıcı ε {r['defl_left_rel_eps']:.3f}, "
                  f"φ {r['defl_left_rel_roll']:.3f} ×başlangıç)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--check", action="store_true", help="yalnız doğrusal kestirim kalitesini yazdır")
    a = ap.parse_args()
    if a.check:
        for c in CASES:
            o = make(0, c)
            print(c, {pl: dict(kalan_sapt_rel=round(float(np.std(o[pl][1]) / np.std(o[pl][2])), 4),
                               sim_um=round(float(np.std(A2.projectors()[0] @ o[pl][0]) * 1e6), 4))
                      for pl in "xy"})
    else:
        run(a.seeds, a.workers)
