#!/usr/bin/env python3
"""tiltscan.py — 0.3μm kalan misalignment üstüne tilt ψ taraması: sahte-EDM
ψ=0.2 mrad'da 1 nrad/s'ye iniyor mu?

Her seed: AYNI 0.3μm white misalignment (BBA-sonrası kalan temsili) + sabit birim
tilt deseni u (per-quad random N(0,1,48)) × ψ ölçeği. f(ψ)-f(0) eşlenik (paired)
olarak tilt katkısını izole eder. 6 seed, tek yön.

Artımlı: /tmp/kmod_recover/tiltscan.jsonl
"""
import os, sys, json, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # kmod_drivers/ (fast_est)
sys.path.insert(0, os.path.join(BASE, "berry_data")); sys.path.insert(0, BASE)
import fast_est as FE
NQ = FE.NQ
os.makedirs("/tmp/kmod_recover", exist_ok=True)
OUT = "/tmp/kmod_recover/tiltscan.jsonl"
MIS = 0.3e-6
PSIS = [0.0, 1e-4, 2e-4, 5e-4]            # 0, 0.1, 0.2, 0.5 mrad
NSEED = 6


def _task(args):
    seed, psi = args
    import tempfile; os.chdir(tempfile.mkdtemp())
    rng_m = np.random.default_rng(4000 + seed)        # misalignment: ψ'den BAĞIMSIZ
    dx = rng_m.normal(0, MIS, NQ); dy = rng_m.normal(0, MIS, NQ)
    rng_u = np.random.default_rng(6000 + seed)        # birim tilt deseni (sabit)
    u = rng_u.normal(0, 1, NQ)
    tilt = psi * u
    f, resid = FE.fast_measure(dx, dy, tilt=tilt)
    return dict(seed=seed, psi=psi, f=float(f), resid=float(resid))


def main():
    os.chdir(BASE)
    tasks = [(s, p) for s in range(NSEED) for p in PSIS]
    open(OUT, "w").close()
    t0 = time.time(); done = 0
    import build_response_matrix as brm
    with ProcessPoolExecutor(4, initializer=brm._worker_init) as pool:
        futs = [pool.submit(_task, t) for t in tasks]
        for fut in as_completed(futs):
            r = fut.result()
            open(OUT, "a").write(json.dumps(r) + "\n")
            done += 1
            if done % 4 == 0:
                print(f"  [{done}/{len(tasks)}] {time.time()-t0:.0f}s", flush=True)
    analyze()
    print(f"[toplam {time.time()-t0:.0f}s]")


def analyze():
    rows = [json.loads(l) for l in open(OUT)]
    F = {}
    for r in rows:
        F.setdefault(r["seed"], {})[r["psi"]] = r["f"]
    seeds = sorted(s for s in F if len(F[s]) == len(PSIS))
    print(f"\n=== TİLT TARAMASI (0.3μm kalan misalignment, n={len(seeds)} seed) ===")
    print(f"  Hedef 1 nrad/s = 1e-9 ; gerçek EDM = 9.81e-10")
    print(f"  {'ψ [mrad]':>9} {'|f| ort':>11} {'±sem':>9} {'tilt katkısı f(ψ)-f(0)':>24} {'hedef?':>7}")
    base = np.array([F[s][0.0] for s in seeds])
    for p in PSIS:
        fp = np.array([F[s][p] for s in seeds])
        absf = np.abs(fp)
        contrib = fp - base                                  # eşlenik tilt katkısı
        sem = np.std(absf) / np.sqrt(len(absf))
        ok = "EVET" if np.mean(absf) < 1e-9 else "hayır"
        print(f"  {p*1e3:8.2f} {np.mean(absf):.3e} {sem:8.1e} "
              f"  ort {np.mean(contrib):+.2e} ± {np.std(contrib)/np.sqrt(len(contrib)):.1e}   {ok}")
    # ψ-ölçekleme: log-log eğim (katkının RMS'i)
    ps = np.array(PSIS[1:]); cr = []
    for p in ps:
        c = np.array([F[s][p] - F[s][0.0] for s in seeds])
        cr.append(np.sqrt(np.mean(c**2)))
    cr = np.array(cr)
    if np.all(cr > 0):
        n = np.polyfit(np.log(ps), np.log(cr), 1)[0]
        print(f"  tilt katkısı RMS ∝ ψ^{n:.2f}  (1=lineer, 2=kuadratik)")


if __name__ == "__main__":
    if "--analyze-only" in sys.argv:
        analyze()
    else:
        main()
