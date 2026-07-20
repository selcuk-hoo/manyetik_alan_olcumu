#!/usr/bin/env python3
"""twobeam_oc.py — İKİ-DEMET (CW/CCW eşzamanlı) YÖRÜNGE DÜZELTME kampanyası.

YENİ TEZ: Tek-demet yörünge düzeltme simetrik kaçıklık modlarını çözemez
(cond(R)=228; en-zayıf modlar %90-98 simetrik) → simetrik taban kalır. AMA iki
karşı-dönen demet simetriği FARKLI görür (R_CW ≠ R_CCW); ikisini birden
düzeltmek koşullanmayı iyileştirir (cond([R_CCW;R_CW])=118; en-zayıf modlar
%63-79 simetrik) → simetrik kısmın büyük bölümü kaldırılır → BBA'sız hedef-altı.

Her seed (temsili desen, 10μm) için ham kaçıklığa (a) TEK-DEMET OC (R_CCW) ve
(b) İKİ-DEMET OC ([R_CCW;R_CW]) uygulanır (BPM gürültülü), kalan sahte-EDM
CW+CCW spin izleyiciyle ölçülür, beam-reversal-tek artık C=(f_CW-f_CCW)/2.

--bbeat: β-beat SAĞLAMLIK modu. Düzeltme NOMİNAL R ile hesaplanır (bildiğin
tepki) ama gerçek makine β-beat'li (R_true); orbit R_true@res, sahte-EDM
β-beat'li makinede (dG=bbeat) ölçülür. §IV model-uyuşmazlığı senaryosunun
kötümser hali (nominal model vs β-beat makine). R_true ilk koşuda üretilir
(build_matrices, quad_dG_pert) ve /tmp'ye cache'lenir.

İŞARETLİ ölçüm; sağlam CO (n_turns=28, n_iter=4); restart-güvenli.

Kullanım:
  python3 kmod_drivers/twobeam_oc.py -w 5 --nseed 5 --bpm-noise 14e-9
  python3 kmod_drivers/twobeam_oc.py -w 5 --nseed 5 --bpm-noise 14e-9 --bbeat 0.01
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
Rx_ccw = np.load("R_dx_1.npy"); Ry_ccw = np.load("R_dy_1.npy")     # nominal, dir=-1
Rx_cw = np.load("R_dx_cw.npy"); Ry_cw = np.load("R_dy_cw.npy")     # nominal, dir=+1
Ps, Pa = sym_anti_projectors()

# β-beat modu: gerçek-makine matrisleri (ilk kullanımda üretilir)
BBEAT = None; RTx_ccw = RTx_cw = RTy_ccw = RTy_cw = None


def _ensure_rtrue(bbeat_amp):
    """β-beat'li R_true (CW+CCW) yükle/üret; sabit makine deseni (seed 0)."""
    global BBEAT, RTx_ccw, RTx_cw, RTy_ccw, RTy_cw
    BBEAT = np.random.default_rng(0).normal(0.0, bbeat_amp, NQ); BBEAT[0] = 0.0
    cache = {t: f"/tmp/Rtrue_{p}_{t}.npy" for t in ("ccw", "cw") for p in ("dy", "dx")}
    need = not all(os.path.exists(f"/tmp/Rtrue_dy_{t}.npy") for t in ("ccw", "cw"))
    if need:
        for d, t in ((-1.0, "ccw"), (1.0, "cw")):
            cfg = json.load(open("params.json")); cfg["direction"] = d
            print(f"  R_true_{t} üretiliyor (β-beat {bbeat_amp*100:.0f}%)...", flush=True)
            Ry, Rx = brm.build_matrices(cfg, delta_q=1e-4, sigma_noise=0.0,
                                        n_workers=4, quad_dG_pert=BBEAT)
            np.save(f"/tmp/Rtrue_dy_{t}.npy", Ry); np.save(f"/tmp/Rtrue_dx_{t}.npy", Rx)
    RTy_ccw = np.load("/tmp/Rtrue_dy_ccw.npy"); RTx_ccw = np.load("/tmp/Rtrue_dx_ccw.npy")
    RTy_cw = np.load("/tmp/Rtrue_dy_cw.npy"); RTx_cw = np.load("/tmp/Rtrue_dx_cw.npy")


def oc1(res, Rcorr, Rtrue, ynoise):
    """Tek-demet: gerçek orbit (Rtrue) ölç, NOMİNAL model (Rcorr) ile düzelt."""
    y = Rtrue @ res + ynoise
    return res - np.linalg.pinv(Rcorr, rcond=RCOND) @ y


def oc2(res, Rc1, Rc2, Rt1, Rt2, n1, n2):
    S = np.vstack([Rc1, Rc2])
    y = np.concatenate([Rt1 @ res + n1, Rt2 @ res + n2])
    return res - np.linalg.pinv(S, rcond=RCOND) @ y


def _C(dx, dy, dG=None, tilt=None):
    fcw, _ = fast_measure(dx, dy, tilt=tilt, direction=+1, signed=True, n_turns=28, n_iter=4, dG=dG)
    fccw, _ = fast_measure(dx, dy, tilt=tilt, direction=-1, signed=True, n_turns=28, n_iter=4, dG=dG)
    return fcw / TARGET, fccw / TARGET


def _task(task):
    import tempfile; os.chdir(tempfile.mkdtemp())
    seed, sig, bbeat_amp, tilt_rms = task
    # β-beat modunda gerçek matrisler /tmp'den (main üretmiş); değilse nominal=gerçek
    if bbeat_amp > 0:
        dG = np.random.default_rng(0).normal(0.0, bbeat_amp, NQ); dG[0] = 0.0
        RTyc = np.load("/tmp/Rtrue_dy_ccw.npy"); RTxc = np.load("/tmp/Rtrue_dx_ccw.npy")
        RTyw = np.load("/tmp/Rtrue_dy_cw.npy"); RTxw = np.load("/tmp/Rtrue_dx_cw.npy")
    else:
        dG = None
        RTxc, RTxw, RTyc, RTyw = Rx_ccw, Rx_cw, Ry_ccw, Ry_cw
    # sabit makine roll deseni (düzeltme R'si bilmez → modellenmemiş sistematik)
    tilt = None
    if tilt_rms > 0:
        tilt = np.random.default_rng(9000).normal(0.0, tilt_rms, NQ); tilt[0] = 0.0
    rng = np.random.default_rng(seed)
    m_dx = rng.normal(0, 10e-6, NQ); m_dx[0] = 0.0
    m_dy = rng.normal(0, 10e-6, NQ); m_dy[0] = 0.0
    nr = np.random.default_rng(50000 + seed)
    nxa, nxb = nr.normal(0, sig, NQ), nr.normal(0, sig, NQ)
    nya, nyb = nr.normal(0, sig, NQ), nr.normal(0, sig, NQ)
    # tek-demet (düzeltme nominal R_CCW, orbit gerçek R_true_CCW)
    c1x = oc1(m_dx, Rx_ccw, RTxc, nxa); c1y = oc1(m_dy, Ry_ccw, RTyc, nya)
    fcw1, fccw1 = _C(c1x, c1y, dG, tilt)
    # iki-demet (düzeltme nominal [R_CCW;R_CW], orbit gerçek [R_true...])
    c2x = oc2(m_dx, Rx_ccw, Rx_cw, RTxc, RTxw, nxa, nxb)
    c2y = oc2(m_dy, Ry_ccw, Ry_cw, RTyc, RTyw, nya, nyb)
    fcw2, fccw2 = _C(c2x, c2y, dG, tilt)
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
    ap.add_argument("--bbeat", type=float, default=0.0,
                    help=">0: β-beat sağlamlık (nominal model vs β-beat makine)")
    ap.add_argument("--tilt", type=float, default=0.0,
                    help=">0: makine quad roll rms [rad] (modellenmemiş; ör. 2e-4=0.2mrad)")
    args = ap.parse_args()
    seeds = list(range(args.nseed))
    if args.bbeat > 0:
        _ensure_rtrue(args.bbeat)     # R_true'yu MAIN'de üret (worker'lar yükler)
    tag = f"{args.bpm_noise*1e9:.0f}nm" + (f"_bb{args.bbeat*100:.0f}" if args.bbeat > 0 else "") \
          + (f"_tilt{args.tilt*1e3:.1f}mrad" if args.tilt > 0 else "")
    OUT = os.path.join(BASE, "kmod_drivers", f"twobeam_oc_{tag}.json")
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    todo = [s for s in seeds if str(s) not in out]
    print(f"=== İKİ-DEMET OC: {len(todo)}/{len(seeds)} seed  (σ_bpm={tag}) ===", flush=True)
    t0 = time.time()
    if todo:
        with ProcessPoolExecutor(args.workers, initializer=brm._worker_init) as pool:
            for r in pool.map(_task, [(s, args.bpm_noise, args.bbeat, args.tilt) for s in todo]):
                out[str(r["seed"])] = {"one": r["one"], "two": r["two"]}
                json.dump(out, open(OUT, "w"), indent=1)
                print(f"  seed {r['seed']}: TEK C={r['one']['C']:.2f}× "
                      f"| İKİ C={r['two']['C']:.3f}× (sim {r['two']['sim_dx']:.2f}μm)  "
                      f"[{time.time()-t0:.0f}s]", flush=True)
    c1 = np.array([out[str(s)]["one"]["C"] for s in seeds])
    c2 = np.array([out[str(s)]["two"]["C"] for s in seeds])
    print(f"\n=== ÖZET (odd artık C, × hedef; {tag}) ===")
    print(f"  TEK-demet OC: medyan {np.median(c1):.2f}×  [{c1.min():.2f}, {c1.max():.2f}]  "
          f"hedef-altı {int((c1<1).sum())}/{len(c1)}")
    print(f"  İKİ-demet OC: medyan {np.median(c2):.3f}×  [{c2.min():.3f}, {c2.max():.3f}]  "
          f"hedef-altı {int((c2<1).sum())}/{len(c2)}")
    print(f"Kaydedildi: {OUT}")


if __name__ == "__main__":
    main()
