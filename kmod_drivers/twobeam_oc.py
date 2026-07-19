#!/usr/bin/env python3
"""twobeam_oc.py — İKİ-DEMET (CW/CCW eşzamanlı) YÖRÜNGE DÜZELTME kampanyası.

YENİ TEZ: Tek-demet yörünge düzeltme simetrik kaçıklık modlarını çözemez
(cond(R)=228; en-zayıf modlar %90-98 simetrik) → simetrik taban kalır. AMA iki
karşı-dönen demet simetriği FARKLI görür (R_CW ≠ R_CCW); ikisini birden
düzeltmek koşullanmayı iyileştirir (cond([R_CCW;R_CW])=118; en-zayıf modlar
%63-79 simetrik) → simetrik kısmın büyük bölümü kaldırılır → BBA'sız hedef-altı.

Bu betik çok-seed + gürültü ile bunu test eder:
  her seed (temsili desen, 10μm) için ham kaçıklığa
    (a) TEK-DEMET OC (R_CCW)  ve  (b) İKİ-DEMET OC ([R_CCW;R_CW])
  uygular (BPM gürültülü), kalan sahte-EDM'i CW+CCW spin izleyiciyle ölçer,
  beam-reversal-tek artık C=(f_CW-f_CCW)/2'yi raporlar.

İŞARETLİ ölçüm (fast_measure signed=True); sağlam CO (n_turns=28, n_iter=4).
Restart-güvenli: seed-başı JSON kaydı.

Kullanım:
  python3 kmod_drivers/twobeam_oc.py -w 5 --nseed 5 --bpm-noise 1e-6
  python3 kmod_drivers/twobeam_oc.py -w 5 --nseed 5 --bpm-noise 14e-9   # lock-in
"""
import os, sys, json, time, argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (BASE, os.path.join(BASE, "kmod_drivers"), os.path.join(BASE, "berry_data")):
    sys.path.insert(0, p)
os.chdir(BASE)
import build_response_matrix as brm
from make_orbit_figures import sym_anti_projectors, NQ
from fast_est import fast_measure

TARGET = 1e-9
RCOND = 0.01
Rx_ccw = np.load("R_dx_1.npy"); Ry_ccw = np.load("R_dy_1.npy")     # direction=-1
Rx_cw = np.load("R_dx_cw.npy"); Ry_cw = np.load("R_dy_cw.npy")     # direction=+1
Sx = np.vstack([Rx_ccw, Rx_cw]); Sy = np.vstack([Ry_ccw, Ry_cw])
Ps, Pa = sym_anti_projectors()


def oc1(res, R, y_noise):
    """Tek-demet: R@res + gürültü ölç, düzelt."""
    y = R @ res + y_noise
    return res - np.linalg.pinv(R, rcond=RCOND) @ y


def oc2(res, R1, R2, y1_noise, y2_noise):
    """İki-demet: her iki yörüngeyi birden düzelt (yığılmış)."""
    S = np.vstack([R1, R2])
    y = np.concatenate([R1 @ res + y1_noise, R2 @ res + y2_noise])
    return res - np.linalg.pinv(S, rcond=RCOND) @ y


def _C(dx, dy):
    fcw, _ = fast_measure(dx, dy, direction=+1, signed=True, n_turns=28, n_iter=4)
    fccw, _ = fast_measure(dx, dy, direction=-1, signed=True, n_turns=28, n_iter=4)
    return fcw / TARGET, fccw / TARGET


def _task(task):
    import tempfile; os.chdir(tempfile.mkdtemp())
    seed, sig = task
    rng = np.random.default_rng(seed)
    m_dx = rng.normal(0, 10e-6, NQ); m_dx[0] = 0.0
    m_dy = rng.normal(0, 10e-6, NQ); m_dy[0] = 0.0
    nr = np.random.default_rng(50000 + seed)   # gürültü gerçeklemesi
    nxa = nr.normal(0, sig, NQ); nxb = nr.normal(0, sig, NQ)
    nya = nr.normal(0, sig, NQ); nyb = nr.normal(0, sig, NQ)
    # tek-demet (R_CCW)
    c1x = oc1(m_dx, Rx_ccw, nxa); c1y = oc1(m_dy, Ry_ccw, nya)
    fcw1, fccw1 = _C(c1x, c1y)
    # iki-demet
    c2x = oc2(m_dx, Rx_ccw, Rx_cw, nxa, nxb); c2y = oc2(m_dy, Ry_ccw, Ry_cw, nya, nyb)
    fcw2, fccw2 = _C(c2x, c2y)
    return dict(seed=seed,
                one={"f_CW": fcw1, "f_CCW": fccw1, "C": abs(fcw1 - fccw1) / 2,
                     "sim_dx": float(np.std(Ps @ c1x) * 1e6)},
                two={"f_CW": fcw2, "f_CCW": fccw2, "C": abs(fcw2 - fccw2) / 2,
                     "sim_dx": float(np.std(Ps @ c2x) * 1e6)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--workers", type=int, default=5)
    ap.add_argument("--nseed", type=int, default=5)
    ap.add_argument("--bpm-noise", type=float, default=1e-6)
    args = ap.parse_args()
    seeds = list(range(args.nseed))
    tag = f"{args.bpm_noise*1e9:.0f}nm"
    OUT = os.path.join(BASE, "kmod_drivers", f"twobeam_oc_{tag}.json")
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    todo = [s for s in seeds if str(s) not in out]
    print(f"=== İKİ-DEMET OC: {len(todo)}/{len(seeds)} seed  (σ_bpm={tag}) ===", flush=True)
    t0 = time.time()
    if todo:
        with ProcessPoolExecutor(args.workers, initializer=brm._worker_init) as pool:
            for r in pool.map(_task, [(s, args.bpm_noise) for s in todo]):
                out[str(r["seed"])] = {"one": r["one"], "two": r["two"]}
                json.dump(out, open(OUT, "w"), indent=1)
                print(f"  seed {r['seed']}: TEK-demet C={r['one']['C']:.2f}× "
                      f"(sim {r['one']['sim_dx']:.2f}μm) | İKİ-demet C={r['two']['C']:.3f}× "
                      f"(sim {r['two']['sim_dx']:.2f}μm)  [{time.time()-t0:.0f}s]", flush=True)
    c1 = np.array([out[str(s)]["one"]["C"] for s in seeds])
    c2 = np.array([out[str(s)]["two"]["C"] for s in seeds])
    print(f"\n=== ÖZET (odd artık C, × hedef; σ_bpm={tag}) ===")
    print(f"  TEK-demet OC: medyan {np.median(c1):.2f}×  [{c1.min():.2f}, {c1.max():.2f}]  "
          f"hedef-altı {int((c1<1).sum())}/{len(c1)}")
    print(f"  İKİ-demet OC: medyan {np.median(c2):.3f}×  [{c2.min():.3f}, {c2.max():.3f}]  "
          f"hedef-altı {int((c2<1).sum())}/{len(c2)}")
    print(f"Kaydedildi: {OUT}")


if __name__ == "__main__":
    main()
