#!/usr/bin/env python3
"""drift_sequence.py — DRİFT DİZİSİ: iki-demet OC sürekli drift monitörü,
STATİK OLMAYAN BPM ofseti altında (hakem yorumu #1/#3).

Senaryo: t0'da hizalı makine (m0≈0), statik-olmayan BPM ofseti b(t) YAVAŞ kayar,
kuadrupol kaçıklığı m(t) daha HIZLI drift eder. Her adımda:
  - ölçüm y_k = [R_ccw·m_k + b_k ; R_cw·m_k + b_k]  (ofset okumaya girer)
  - YUVARLANAN diferansiyel düzeltme: Δy = y_k − y_{k−1} = R·Δm + Δb
      → yalnız DÖNGÜ-İÇİ ofset değişimi Δb sızar (küçük); c = pinv(S)·Δy
  - SABİT-REFERANS düzeltme (kıyas): Δy = y_k − y_0 → ofset kayması b_k−b_0 BİRİKİR
  - düzeltmesiz (kıyas): m_k düzeltilmez
Her adımda sahte-EDM C=½|f_CW−f_CCW| (C++ spin) ölçülür.

Amaç: sürekli yuvarlanan diferansiyel iki-demet OC, statik-OLMAYAN ofset altında
bile simetrik drifti hedef-altı tutar; sabit-referans ofset kaymasıyla bozulur.

Kullanım: python3 kmod_drivers/drift_sequence.py --nstep 6 --seed 0
"""
import os, sys, json, argparse
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (BASE, os.path.join(BASE, "kmod_drivers")):
    sys.path.insert(0, p)
os.chdir(BASE)
from make_orbit_figures import sym_anti_projectors, NQ
from fast_est import fast_measure

TARGET = 1e-9
RCOND = 0.01
Rx_ccw = np.load("R_dx_1.npy"); Ry_ccw = np.load("R_dy_1.npy")
Rx_cw = np.load("R_dx_cw.npy"); Ry_cw = np.load("R_dy_cw.npy")
Ps, Pa = sym_anti_projectors()
Sx = np.vstack([Rx_ccw, Rx_cw]); Sy = np.vstack([Ry_ccw, Ry_cw])
Sx_inv = np.linalg.pinv(Sx, rcond=RCOND); Sy_inv = np.linalg.pinv(Sy, rcond=RCOND)


def _C(dx, dy):
    fcw, _ = fast_measure(dx, dy, direction=+1, signed=True, n_turns=28, n_iter=4)
    fccw, _ = fast_measure(dx, dy, direction=-1, signed=True, n_turns=28, n_iter=4)
    return abs(fcw - fccw) / 2 / TARGET


def two_beam_corr(Sinv, dy_ccw_reading, dy_cw_reading):
    """Yığın [R_ccw;R_cw] ile diferansiyel okumadan düzeltme vektörü."""
    return Sinv @ np.concatenate([dy_ccw_reading, dy_cw_reading])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nstep", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mis-drift", type=float, default=2e-6, help="adım başına kaçıklık drift rms [m]")
    ap.add_argument("--off-drift", type=float, default=0.6e-6, help="adım başına BPM ofset drift rms [m] (yavaş)")
    ap.add_argument("--boff0", type=float, default=100e-6, help="başlangıç statik ofset rms [m]")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # başlangıç: hizalı makine (m0 küçük), büyük statik ofset
    mx = np.zeros(NQ); my = np.zeros(NQ)
    bx = rng.normal(0, args.boff0, NQ); by = rng.normal(0, args.boff0, NQ)
    bx[0] = by[0] = 0.0
    # üç senaryo için ayrı makine durumu (aynı drift dizisini görürler)
    st = {s: {"mx": mx.copy(), "my": my.copy()} for s in ("none", "rolling", "fixed")}
    y0 = {}   # fixed-ref: t0 okuması
    yprev = {}  # rolling: bir önceki okuma
    out = {"args": vars(args), "steps": []}

    def reading(mx_, my_, bx_, by_):
        return (Rx_ccw @ mx_ + bx_, Rx_cw @ mx_ + bx_,
                Ry_ccw @ my_ + by_, Ry_cw @ my_ + by_)

    for k in range(args.nstep + 1):
        if k > 0:
            # drift ekle: kaçıklık hızlı, ofset yavaş (statik DEĞİL)
            dmx = rng.normal(0, args.mis_drift, NQ); dmx[0] = 0
            dmy = rng.normal(0, args.mis_drift, NQ); dmy[0] = 0
            for s in st:
                st[s]["mx"] += dmx; st[s]["my"] += dmy
            bx += rng.normal(0, args.off_drift, NQ); bx[0] = 0
            by += rng.normal(0, args.off_drift, NQ); by[0] = 0

        rec = {"step": k, "sym_x_drift": float(np.std(Ps @ st["none"]["mx"]) * 1e6),
               "off_rms": float(np.std(bx) * 1e6)}
        for s in ("none", "rolling", "fixed"):
            yxc, yxw, yyc, yyw = reading(st[s]["mx"], st[s]["my"], bx, by)
            if s == "rolling" and k > 0:
                pc = yprev[s]
                st[s]["mx"] -= two_beam_corr(Sx_inv, yxc - pc[0], yxw - pc[1])
                st[s]["my"] -= two_beam_corr(Sy_inv, yyc - pc[2], yyw - pc[3])
            elif s == "fixed" and k > 0:
                r0 = y0[s]
                st[s]["mx"] -= two_beam_corr(Sx_inv, yxc - r0[0], yxw - r0[1])
                st[s]["my"] -= two_beam_corr(Sy_inv, yyc - r0[2], yyw - r0[3])
            # okumayı güncelle (düzeltme SONRASI değil, düzeltme-öncesi referans için)
            if k == 0:
                y0[s] = (yxc, yxw, yyc, yyw)
            yprev[s] = reading(st[s]["mx"], st[s]["my"], bx, by)  # düzeltme sonrası
            rec[f"C_{s}"] = _C(st[s]["mx"], st[s]["my"])
            rec[f"symres_{s}"] = float(np.std(Ps @ st[s]["mx"]) * 1e6)
        out["steps"].append(rec)
        print(f"adım {k}: drift_sym {rec['sym_x_drift']:.2f}um ofset {rec['off_rms']:.1f}um | "
              f"C none {rec['C_none']:.2f} | rolling {rec['C_rolling']:.3f} | "
              f"fixed {rec['C_fixed']:.3f}", flush=True)

    OUT = os.path.join(BASE, "kmod_drivers", f"drift_seq_s{args.seed}.json")
    json.dump(out, open(OUT, "w"), indent=1)
    Cn = [r["C_none"] for r in out["steps"][1:]]
    Cr = [r["C_rolling"] for r in out["steps"][1:]]
    Cf = [r["C_fixed"] for r in out["steps"][1:]]
    print(f"\n=== ÖZET ({args.nstep} adım) ===")
    print(f"  düzeltmesiz:      C son {Cn[-1]:.1f}×  (drift birikir)")
    print(f"  YUVARLANAN diff:  C medyan {np.median(Cr):.3f}×  max {max(Cr):.3f}×  (ofset iptal)")
    print(f"  sabit-referans:   C medyan {np.median(Cf):.3f}×  max {max(Cf):.3f}×  (ofset kayması birikir)")
    print(f"Kaydedildi: {OUT}")


if __name__ == "__main__":
    main()
