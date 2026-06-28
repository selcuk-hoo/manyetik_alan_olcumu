#!/usr/bin/env python3
"""cwccw_ens.py — CW/CCW + quad-flip'in TİLT sahte-EDM'ini giderip gidermediği,
20-seed ENSEMBLE (tek-seed CO-gürültüsünü söndürmek için).

Her seed: AYNI 10μm per-quad misalignment + per-quad random tilt (48 bağımsız).
4 ölçüm: f(CW,tilt=0), f(CCW,tilt=0), f(CW,tilt=1mrad), f(CCW,tilt=1mrad).

Analiz:
  EDM-kanalı sahte-EDM (gerçek EDM bu kanalda çıkar):  C(d-çifti) = (f_CW - f_CCW)/2
  - C0 = tilt'siz ;  Ct = tilt'li ;  marjinal = Ct - C0  (tilt'in EDM-kanalına katkısı)
  Tilt marjinalinin PARİTESİ: δ_CW=f(CW,t)-f(CW,0), δ_CCW=f(CCW,t)-f(CCW,0)
  - even = (δ_CW+δ_CCW)/2  (CW/CCW farkında SÖNER → CW/CCW giderir)
  - odd  = (δ_CW-δ_CCW)/2  (CW/CCW farkında KALIR → giderilemez)

Artımlı JSONL çıktı (kesilse bile kısmi sonuç): /tmp/kmod_recover/cwccw_ens.jsonl
Kullanım: python3 cwccw_ens.py --nseed 20 --workers 4
"""
import os, sys, json, time, argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # kmod_drivers/ (fast_est)
sys.path.insert(0, os.path.join(BASE, "berry_data")); sys.path.insert(0, BASE)
import fast_est as FE                                   # fast_measure (azaltılmış ayar)
NQ = FE.NQ
os.makedirs("/tmp/kmod_recover", exist_ok=True)
OUT = "/tmp/kmod_recover/cwccw_ens.jsonl"
SIG_MIS = 10e-6
THETA = 1e-3                                            # 1 mrad


def _task(args):
    seed, direction, has_tilt = args
    import tempfile; os.chdir(tempfile.mkdtemp())
    rng = np.random.default_rng(9000 + seed)            # seed → desen (CW/CCW AYNI)
    dx = rng.normal(0, SIG_MIS, NQ); dy = rng.normal(0, SIG_MIS, NQ)
    tilt = rng.normal(0, THETA, NQ) if has_tilt else np.zeros(NQ)  # per-quad random
    f, resid = FE.fast_measure(dx, dy, tilt=tilt, direction=direction)
    return dict(seed=seed, direction=direction, has_tilt=int(has_tilt),
                f=float(f), resid=float(resid))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nseed", type=int, default=20)
    ap.add_argument("--workers", "-w", type=int, default=4)
    args = ap.parse_args()
    os.chdir(BASE)

    tasks = [(s, d, ht) for s in range(args.nseed)
             for d in (+1, -1) for ht in (False, True)]
    open(OUT, "w").close()                              # temizle
    t0 = time.time()
    done = 0
    import build_response_matrix as brm
    with ProcessPoolExecutor(args.workers, initializer=brm._worker_init) as pool:
        futs = [pool.submit(_task, t) for t in tasks]
        for fut in as_completed(futs):
            r = fut.result()
            with open(OUT, "a") as fh:
                fh.write(json.dumps(r) + "\n")
            done += 1
            if done % 8 == 0:
                print(f"  [{done}/{len(tasks)}] {time.time()-t0:.0f}s", flush=True)
    analyze()
    print(f"[toplam {time.time()-t0:.0f}s]")


def analyze():
    rows = [json.loads(l) for l in open(OUT)]
    # F[seed][(direction, has_tilt)] = f
    F = {}
    for r in rows:
        F.setdefault(r["seed"], {})[(r["direction"], r["has_tilt"])] = r["f"]
    seeds = sorted(s for s in F if len(F[s]) == 4)
    C0, Ct, dCW, dCCW = [], [], [], []
    for s in seeds:
        fcw0 = F[s][(+1, 0)]; fccw0 = F[s][(-1, 0)]
        fcwt = F[s][(+1, 1)]; fccwt = F[s][(-1, 1)]
        C0.append((fcw0 - fccw0) / 2)
        Ct.append((fcwt - fccwt) / 2)
        dCW.append(fcwt - fcw0); dCCW.append(fccwt - fccw0)
    C0 = np.array(C0); Ct = np.array(Ct)
    dCW = np.array(dCW); dCCW = np.array(dCCW)
    marg = Ct - C0
    even = (dCW + dCCW) / 2
    odd = (dCW - dCCW) / 2
    n = len(seeds)
    sem = lambda a: np.std(a) / np.sqrt(len(a))
    print(f"\n=== CW/CCW + FLIP, TİLT ENSEMBLE (n={n} seed, 10μm mis + 1mrad tilt) ===")
    print(f"  Tilt'siz EDM-kanalı |C0|   : RMS {np.sqrt(np.mean(C0**2)):.3e} rad/s")
    print(f"  Tilt'li  EDM-kanalı |Ct|   : RMS {np.sqrt(np.mean(Ct**2)):.3e} rad/s")
    print(f"  Tilt MARJİNALİ (Ct-C0)     : ort {np.mean(marg):+.3e} ± {sem(marg):.1e} "
          f"(RMS {np.sqrt(np.mean(marg**2)):.3e})")
    print(f"  Tilt marjinali EVEN (söner): RMS {np.sqrt(np.mean(even**2)):.3e} ± {sem(even):.1e}")
    print(f"  Tilt marjinali ODD (kalır) : RMS {np.sqrt(np.mean(odd**2)):.3e} ± {sem(odd):.1e}")
    ratio = np.sqrt(np.mean(odd**2)) / (np.sqrt(np.mean(even**2)) + 1e-30)
    print(f"  ODD/EVEN oranı             : {ratio:.2f}")
    print(f"  → Tilt'in CW/CCW-farkında KALAN kısmı (ODD) hedefe (1e-9) göre: "
          f"{np.sqrt(np.mean(odd**2))/1e-9:.1f}×")
    np.savez("/tmp/kmod_recover/cwccw_ens_summary.npz",
             C0=C0, Ct=Ct, even=even, odd=odd, marg=marg, seeds=seeds)


if __name__ == "__main__":
    if "--analyze-only" in sys.argv:
        analyze()
    else:
        main()
