#!/usr/bin/env python3
"""
classic_bba_iter.py — İTERATİF BBA: β-beat/nefes çöküşünün çözümü testi
(separation_bba_testleri.md §5.2)

Teşhis: β-beat %1 altında BBA çöküyordu çünkü modülasyon, DİĞER kaçık quad'ların
kurduğu büyük (0.37 mm) yörüngeyi "nefes"le yeniden taşır ve β-beat bu nefesi
feed-down'dan saptırıp null'u ~2 μm kaydırır. İzole test kesin gösterdi: yörünge
sıfırlanınca bias 0.02 μm'e düşer. → Çözüm: yörüngeyi küçült.

İterasyon (standart BBA pratiği "önce orbit düzelt"in doğal hâli):
  geçiş k: kalan-kaçıklık makinesinde (offsets = residual) BBA → merkez kestir
           → residual −= est  (merkezlere "taşı" → yörünge küçülür)
  → her geçişte yörünge ~geometrik oranla küçülür → nefes-bias küçülür.
Her geçiş sonunda kalan sahte-EDM SPİN İZLEYİCİYLE ölçülür (merkez-RMS değil).

Gürültüsüz + ofsetsiz (ikisi ayrı doğrulandı: ofset iptal, gürültü 1/√N) —
burada tek soru β-beat/nefes'in iterasyonla çözülüp çözülmediği.

Kullanım: python3 classic_bba_iter.py -w 4 --passes 3 --bbeat 0.01
"""
import argparse, json, os, sys, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from make_orbit_figures import R_perquad, sym_anti_projectors, NQ, G_NOM
from classic_bba_full import _orbit_xy, estimate, EPS
from analytic_kmod import compute_twiss_at_quads, signed_KL, build_R_analytic, compute_Brho


def build_meta(R0_model, scan):
    """Her (quad,düzlem) için düzlem-özel düzeltici j ve bump shift."""
    meta = []
    for i in range(1, NQ):
        cands = [(i-1) % NQ, (i+1) % NQ, (i-2) % NQ, (i+2) % NQ]
        for plane in ("y", "x"):
            Rp = R0_model[plane]
            j = max(cands, key=lambda jj: abs(Rp[i, jj]))
            shift = scan / Rp[i, j]
            meta.append((i, plane, j, shift))
    return meta


def run_pass(base_x, base_y, bbeat, meta, pool):
    """Kalan-kaçıklık makinesinde bir BBA geçişi → est_x, est_y (C++)."""
    tasks = []
    for (i, plane, j, shift) in meta:
        for sgn in (-1.0, +1.0):
            dxx, dyy = base_x.copy(), base_y.copy()
            if plane == "y":
                dyy[j] += sgn * shift
            else:
                dxx[j] += sgn * shift
            dG_on = bbeat.copy(); dG_on[i] += EPS
            dG_off = bbeat.copy()
            for on in (1, 0):
                tasks.append((list(dxx), list(dyy),
                              list(dG_on if on else dG_off),
                              f"q{i}_{plane}_{'p' if sgn>0 else 'm'}_{on}"))
    recs = list(pool.map(_orbit_xy, tasks))
    by = {r["tag"]: r for r in recs}
    meta3 = [(i, plane, j) for (i, plane, j, _) in meta]
    return estimate(by, meta3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--workers", type=int, default=4)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--scan", type=float, default=150e-6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bbeat", type=float, default=0.01)
    args = ap.parse_args()
    t0 = time.time()

    rng = np.random.default_rng(args.seed)
    m_dx = rng.normal(0.0, 10e-6, NQ); m_dx[0] = 0.0
    m_dy = rng.normal(0.0, 10e-6, NQ); m_dy[0] = 0.0
    bbeat = rng.normal(0.0, args.bbeat, NQ) if args.bbeat > 0 else np.zeros(NQ)
    bbeat[0] = 0.0

    cfg = json.load(open(os.path.join(BASE, "params.json")))
    nF = int(cfg["nFODO"]); Lq = float(cfg["quadLen"]); Brho = compute_Brho(cfg)
    def R0(plane):
        beta, phi, Q = compute_twiss_at_quads(cfg, G_NOM, plane)
        return build_R_analytic(beta, phi, Q, signed_KL(nF, abs(G_NOM)/Brho, Lq, plane))
    R0_model = {"y": R0("y"), "x": R0("x")}
    meta = build_meta(R0_model, args.scan)
    P_sym, _ = sym_anti_projectors()
    rms = lambda a: float(np.sqrt(np.mean(a**2)))

    # yörünge büyüklüğü göstergesi (analitik, kılavuz): ||R·residual||
    Ry = R0_model["y"]
    def orbit_rms(rx, ry):
        return rms(Ry @ ry) * 1e6  # μm, dikey (kaba gösterge)

    sys.path.insert(0, os.path.join(BASE, "kmod_drivers"))
    from fast_est import fast_measure

    print(f"=== İTERATİF BBA (β-beat {args.bbeat*100:.1f}%, {args.passes} geçiş) ===")
    f_raw, _ = fast_measure(m_dx, m_dy)
    print(f"  ham sahte-EDM = {f_raw:.3e} ({f_raw/1e-9:.0f}× hedef); "
          f"başlangıç yörünge ~{orbit_rms(m_dx,m_dy):.0f} μm (dikey)")

    hist = []
    with ProcessPoolExecutor(args.workers, initializer=__import__('build_response_matrix')._worker_init) as pool:
        res_x, res_y = m_dx.copy(), m_dy.copy()
        for p in range(1, args.passes + 1):
            est_x, est_y = run_pass(res_x, res_y, bbeat, meta, pool)
            res_x = res_x - est_x; res_y = res_y - est_y
            res_x[0] = res_y[0] = 0.0
            f_p, resid_co = fast_measure(res_x, res_y)
            print(f"  geçiş {p}: kalan sym dx {rms(P_sym@res_x)*1e6:.3f} / "
                  f"dy {rms(P_sym@res_y)*1e6:.3f} μm  yörünge ~{orbit_rms(res_x,res_y):.1f} μm"
                  f"  →  sahte-EDM {f_p:.3e} ({f_p/1e-9:.2f}× hedef)")
            hist.append({"pass": p, "f": f_p, "sym_dx": rms(P_sym@res_x),
                         "sym_dy": rms(P_sym@res_y), "orbit_um": orbit_rms(res_x, res_y)})
            # geçiş-başına artımlı kayıt (uzun koşum kesilse bile korunur)
            pth = os.path.join(BASE, "kmod_drivers", "paper_runs_results.json")
            dd = json.load(open(pth)) if os.path.exists(pth) else {}
            dd["bba_iter"] = {"bbeat": args.bbeat, "seed": args.seed,
                              "f_raw": f_raw, "passes": hist}
            json.dump(dd, open(pth, "w"), indent=1)

    print(f"\n  ham {f_raw/1e-9:.0f}× → {args.passes} geçiş sonrası "
          f"{hist[-1]['f']/1e-9:.2f}× hedef  (toplam bastırma {f_raw/max(hist[-1]['f'],1e-30):.0f}×)")
    path = os.path.join(BASE, "kmod_drivers", "paper_runs_results.json")
    data = json.load(open(path)) if os.path.exists(path) else {}
    data["bba_iter"] = {"bbeat": args.bbeat, "seed": args.seed, "f_raw": f_raw,
                        "passes": hist}
    json.dump(data, open(path, "w"), indent=1)
    print(f"  kaydedildi → [bba_iter]   [toplam {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
