#!/usr/bin/env python3
"""channel_split.py — İTERATİF BBA son-geçiş artığının kanal ayrışımı.

classic_bba_iter.py bir koşumun kalan-kaçıklığını (res_x, res_y)
/tmp/kmod_recover/{rkey}_state.json içine yazar. Bu betik o artığı yükler,
simetrik / antisimetrik alt-uzaylara projekte eder ve HER birinin sahte-EDM'ini
C++ spin izleyicisiyle (fast_measure, 4D-CO + model-fit) DOĞRUDAN ölçer.
Makale §13 için f_anti / f_sym sağlam sayıları.

Kullanım (classic_bba_iter --rmatrix cpp koşumundan sonra):
  python3 kmod_drivers/channel_split.py --rkey bba_iter_cpp
  # analitik R koşumu için:  --rkey bba_iter
"""
import os, sys, json, argparse
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
os.chdir(BASE)
from make_orbit_figures import sym_anti_projectors, NQ
sys.path.insert(0, os.path.join(BASE, "kmod_drivers"))
from fast_est import fast_measure

TARGET = 1e-9  # 1 nrad/s ↔ 1e-29 e·cm hedef


def load_resid(rkey):
    st = f"/tmp/kmod_recover/{rkey}_state.json"
    if not os.path.exists(st):
        raise SystemExit(f"STATE dosyası yok: {st}\n"
                         f"Önce classic_bba_iter.py'yi --rmatrix cpp ile koştur.")
    d = json.load(open(st))
    return np.array(d["res_x"]), np.array(d["res_y"]), d.get("hist", [])


def meas(dx, dy, label, bbeat=None):
    f, _ = fast_measure(dx, dy, dG=bbeat)     # fast_measure → (|slope|, resid)
    print(f"  {label:28s}: |f| = {abs(f)/TARGET:8.2f}× hedef  "
          f"(rms dx={np.std(dx)*1e6:.2f}μm dy={np.std(dy)*1e6:.2f}μm)")
    return abs(f) / TARGET


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rkey", default="bba_iter_cpp",
                    help="classic_bba_iter RKEY (cpp: bba_iter_cpp, analitik: bba_iter)")
    ap.add_argument("--bbeat", type=float, default=0.01,
                    help="koşumdaki β-beat (gradyan-hata RMS); seed 0 deseni")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    Ps, Pa = sym_anti_projectors()
    rx, ry, hist = load_resid(args.rkey)
    # koşumun β-beat makinesi (classic_bba_iter ile aynı üretim)
    rng = np.random.default_rng(args.seed)
    bbeat = rng.normal(0.0, args.bbeat, NQ) if args.bbeat > 0 else np.zeros(NQ)
    bbeat[0] = 0.0

    print(f"=== SON-GEÇİŞ ARTIĞI KANAL AYRIŞIMI ({len(hist)} geçiş, {args.rkey}) ===")
    print(f"  artık rms: dx={np.std(rx)*1e6:.2f}μm  dy={np.std(ry)*1e6:.2f}μm")
    f_full = meas(rx, ry, "tam artık", bbeat)
    f_sym = meas(Ps @ rx, Ps @ ry, "yalnız simetrik (P_sym)", bbeat)
    f_anti = meas(Pa @ rx, Pa @ ry, "yalnız antisim (P_anti)", bbeat)
    print(f"\n  ÖZET: f_anti={f_anti:.1f}× ≫ f_sym={f_sym:.1f}×  (tam={f_full:.1f}×)")

    out = {"rkey": args.rkey, "n_pass": len(hist), "f_full": f_full,
           "f_sym": f_sym, "f_anti": f_anti,
           "rms_dx_um": float(np.std(rx) * 1e6), "rms_dy_um": float(np.std(ry) * 1e6),
           "hist": hist}
    outp = os.path.join(BASE, "kmod_drivers", "channel_split_out.json")
    json.dump(out, open(outp, "w"), indent=1)
    print(f"\nKaydedildi: {outp}")


if __name__ == "__main__":
    main()
